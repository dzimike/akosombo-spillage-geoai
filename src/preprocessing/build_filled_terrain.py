from __future__ import annotations
"""
SUPERSEDED (kept for provenance): intermediate approach that filled SRTM voids
and applied the ORIGINAL model without retraining. The study now uses a
gap-free GLO-30 DEM and a retrained model (build_gapfree_dem.py).

Void-filled terrain predictors for full-domain flood probability inference.

The original DEM (data/processed/dem/elevation.tif) is raw SRTM clipped to the
AOI polygon and contains SRTM voids over open water and wetland, where most
of the mapped flood lies. Core terrain predictors are therefore NaN there and
predict_susceptibility.py (by design) returns no-data. This script rebuilds
the terrain bands without changing any pixel that was already valid:

  1. elevation = raw SRTM wherever valid (identical to the original DEM where
     both exist, and also recovers pixels masked only by the polygon clip);
     inside SRTM voids, NASADEM (Earth Engine, 30 m) shifted by the median
     NASADEM-SRTM offset over co-valid pixels.
  2. Every band keeps its ORIGINAL value wherever the original band was valid,
     so the model's training domain is unchanged. Only originally-invalid
     pixels receive newly computed values, using the SAME definitions the
     trained model saw: slope/aspect from np.gradient; "flow accumulation" is
     the rank-of-elevation proxy that preprocess_dem.py fell back to (pysheds
     failed), extended to new pixels by ranking against the original
     elevation distribution; TWI from that proxy; relative elevation =
     elevation minus the original floor.
  3. Bands aligned to the 10 m template exactly as terrain_features.py does.

Nothing under data/processed/dem or data/processed/features is overwritten.

Outputs
-------
data/processed/dem_filled/*.tif
data/processed/features_filled/terrain/*.tif   (8 predictor bands)
outputs/tables/dem_fill_diagnostics.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.preprocessing.preprocess_dem import NODATA_FILL, compute_slope_aspect, compute_twi
from src.features.terrain_features import write_feature
from src.utils.paths import get_project_root, get_output_path, ensure_dir
from src.utils.logging_utils import Timer, log_run
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GEE_PROJECT = "instigis"


def _on_grid(src_path: Path, tr, crs, shape, src_nodata) -> np.ndarray:
    out = np.full(shape, np.nan, np.float32)
    with rasterio.open(src_path) as s:
        reproject(rasterio.band(s, 1), out, dst_transform=tr, dst_crs=crs,
                  src_nodata=src_nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


def _fetch_nasadem(tr, shape, crs, dest: Path) -> None:
    import ee, requests
    ee.Initialize(project=GEE_PROJECT)
    H, W = shape
    img = ee.Image("NASA/NASADEM_HGT/001").select("elevation").resample("bilinear")
    region = ee.Geometry.Rectangle(
        [tr.c, tr.f + H * tr.e, tr.c + W * tr.a, tr.f], proj=str(crs), geodesic=False)
    url = img.getDownloadURL({"name": "nasadem", "crs": str(crs),
                              "crs_transform": [tr.a, 0, tr.c, 0, tr.e, tr.f],
                              "region": region, "format": "GEO_TIFF"})
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    dest.write_bytes(r.content)


def main() -> None:
    root = get_project_root()
    dem_old = root / "data/processed/dem"
    dem_new = root / "data/processed/dem_filled"
    ter_new = root / "data/processed/features_filled/terrain"
    for d in (dem_new, ter_new):
        ensure_dir(d)
    template = root / "data/processed/satellite/s2_ndwi_event.tif"
    nasadem_path = root / "data/raw/dem/nasadem_gee_utm30_30m.tif"

    with Timer() as t:
        with rasterio.open(dem_old / "elevation.tif") as s:
            tr, crs, shape, profile = s.transform, s.crs, s.shape, s.profile.copy()
            old_elev = s.read(1).astype(np.float32)
        old_elev[old_elev == NODATA_FILL] = np.nan
        with rasterio.open(dem_old / "relative_elevation.tif") as s:
            old_rel = s.read(1).astype(np.float32)
        old_rel[old_rel == NODATA_FILL] = np.nan
        floor = float(np.nanmedian(old_elev - old_rel))
        log.info("Original relative-elevation floor: %.3f m", floor)

        if not nasadem_path.exists():
            log.info("Fetching NASADEM from Earth Engine…")
            _fetch_nasadem(tr, shape, crs, nasadem_path)
        srtm = _on_grid(root / "data/raw/dem/srtm_37_11.tif", tr, crs, shape, -32768)
        nasa = _on_grid(nasadem_path, tr, crs, shape, -32768)

        both = np.isfinite(srtm) & np.isfinite(nasa)
        offset = float(np.median(nasa[both] - srtm[both]))
        log.info("NASADEM - SRTM median offset: %.3f m (n=%d)", offset, both.sum())

        def load_old(name):
            with rasterio.open(dem_old / f"{name}.tif") as s:
                a = s.read(1).astype(np.float32)
            a[a == NODATA_FILL] = np.nan
            return a

        elev = np.where(np.isfinite(old_elev), old_elev,
                        np.where(np.isfinite(srtm), srtm, nasa - offset)).astype(np.float32)
        assert np.isfinite(elev).all(), "residual voids after NASADEM fill"
        new_px = ~np.isfinite(old_elev)
        log.info("Newly filled elevation pixels: %d (%.1f%% of grid): from raw SRTM %d, from NASADEM %d",
                 new_px.sum(), 100 * new_px.mean(),
                 (new_px & np.isfinite(srtm)).sum(), (new_px & ~np.isfinite(srtm)).sum())

        slope_n, aspect_n = compute_slope_aspect(elev, 30.0, NODATA_FILL)
        slope_n = np.where(slope_n == NODATA_FILL, np.nan, slope_n)
        aspect_n = np.where(aspect_n == NODATA_FILL, np.nan, aspect_n)
        sorted_old = np.sort(old_elev[np.isfinite(old_elev)])
        fa_n = np.searchsorted(sorted_old, elev, side="left").astype(np.float32)
        old = {k: load_old(k) for k in ("slope", "aspect", "flow_accumulation", "twi", "relative_elevation")}
        slope = np.where(np.isfinite(old["slope"]), old["slope"], slope_n)
        aspect = np.where(np.isfinite(old["aspect"]), old["aspect"], aspect_n)
        fa = np.where(np.isfinite(old["flow_accumulation"]), old["flow_accumulation"], fa_n)
        twi_n = compute_twi(np.nan_to_num(fa), np.nan_to_num(slope), 30.0 ** 2, NODATA_FILL)
        twi = np.where(np.isfinite(old["twi"]), old["twi"], np.where(twi_n == NODATA_FILL, np.nan, twi_n))
        rel = np.where(np.isfinite(old["relative_elevation"]), old["relative_elevation"], elev - floor)

        profile.update(driver="GTiff", dtype="float32", nodata=NODATA_FILL, compress="lzw")
        final = {"elevation": elev, "slope": slope, "aspect": aspect,
                 "flow_accumulation": fa, "twi": twi, "relative_elevation": rel}
        for name, arr in final.items():
            with rasterio.open(dem_new / f"{name}.tif", "w", **profile) as dst:
                dst.write(np.where(np.isfinite(arr), arr, NODATA_FILL).astype(np.float32), 1)

        # compare distributions of newly-valid pixels with the original training domain
        rows = []
        olds = {"elevation": old_elev, **old}
        for name, arr in final.items():
            o = olds[name]; was = np.isfinite(o); new_only = np.isfinite(arr) & ~was
            ov = o[was]
            a = arr[new_only]
            rows.append({"band": name, "n_orig_valid": int(was.sum()), "n_newly_valid": int(new_only.sum()),
                         "orig_mean": float(ov.mean()), "new_mean": float(a.mean()) if a.size else np.nan,
                         "orig_p05": float(np.percentile(ov, 5)), "new_p05": float(np.percentile(a, 5)) if a.size else np.nan,
                         "orig_p95": float(np.percentile(ov, 95)), "new_p95": float(np.percentile(a, 95)) if a.size else np.nan,
                         "frac_new_outside_orig_range": float(((a < ov.min()) | (a > ov.max())).mean()) if a.size else np.nan,
                         "unchanged_on_orig_valid": bool(np.all(arr[was] == o[was]))})
            log.info("  %-20s new px %8d | mean orig %.3f new %.3f | outside orig range %.3f | orig unchanged=%s",
                     name, rows[-1]["n_newly_valid"], rows[-1]["orig_mean"], rows[-1]["new_mean"],
                     rows[-1]["frac_new_outside_orig_range"], rows[-1]["unchanged_on_orig_valid"])
        pd.DataFrame(rows).to_csv(get_output_path("tables/dem_fill_diagnostics.csv"), index=False)

        # align to 10 m template, mirroring terrain_features.py
        with rasterio.open(template) as s:
            tprofile = s.profile.copy(); tprofile.update(driver="GTiff")
        feats = {}
        for name in ("elevation", "slope", "aspect", "twi", "flow_accumulation", "relative_elevation"):
            a = read_aligned(str(dem_new / f"{name}.tif"), str(template), Resampling.bilinear)
            feats[name] = np.where(a < -9000, np.nan, a).astype(np.float32)
        rad = np.radians(feats.pop("aspect"))
        feats["aspect_sin"] = np.sin(rad).astype(np.float32)
        feats["aspect_cos"] = np.cos(rad).astype(np.float32)
        f = feats.pop("flow_accumulation")
        feats["flow_accumulation_log"] = np.where(np.isfinite(f), np.log1p(np.clip(f, 0, None)), np.nan).astype(np.float32)
        feats["hand_proxy"] = feats["relative_elevation"].copy()
        for name, arr in feats.items():
            write_feature(arr, tprofile, ter_new / f"{name}.tif", f"{name} (void-filled DEM)")
            log.info("  %-24s valid=%.1f%%", name, 100 * np.isfinite(arr).mean())

    log_run(script=__file__, input_paths=[str(dem_old / "elevation.tif"), str(nasadem_path)],
            output_paths=[str(ter_new / f"{n}.tif") for n in feats],
            parameters={"nasadem_srtm_offset_m": offset, "rel_elev_floor_m": floor},
            status="success", runtime_seconds=t.elapsed)
    log.info("Done in %.1f s", t.elapsed)


if __name__ == "__main__":
    main()
