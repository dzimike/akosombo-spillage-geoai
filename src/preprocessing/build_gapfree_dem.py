from __future__ import annotations
"""
Gap-free DEM and terrain derivatives for the full raster domain.

Replaces the SRTM-based DEM stage (preprocess_dem.py) for the study run. The
original DEM was raw SRTM clipped to the AOI polygon and contained SRTM voids
over open water and wetland, so terrain predictors were NaN across ~80 % of the
mapped flood area and those pixels were dropped from model training. Here the
elevation surface is Copernicus GLO-30 (30 m, void-free, water-flattened),
sampled on the same 30 m UTM-30N grid as the original DEM and NOT clipped to
the AOI polygon, so the whole 10 m analysis domain has terrain values.

Derivative DEFINITIONS are deliberately unchanged from preprocess_dem.py so
that the only experimental change is the elevation source:
  slope / aspect  : finite-difference gradient (np.gradient)
  flow_accumulation : rank of each pixel's elevation among all valid pixels
                      (the elevation-rank proxy preprocess_dem.py fell back to;
                      NOT a routed upslope contributing area)
  twi             : ln(a / tan(beta)) using that rank index for a
  relative_elevation : elevation minus the 5th-percentile elevation of the DEM

Outputs (data/processed/dem/, same filenames as before)
-------
elevation.tif, slope.tif, aspect.tif, flow_accumulation.tif, twi.tif,
relative_elevation.tif
data/raw/dem/glo30_gee_utm30_30m.tif  (raw download, cached)
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.preprocessing.preprocess_dem import (
    NODATA_FILL, compute_slope_aspect, compute_twi, compute_relative_elevation,
    _simple_flow_accum,
)
from src.utils.logging_utils import Timer, log_run
from src.utils.paths import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GEE_PROJECT = "instigis"
GRID_REF = "archive/v1_srtm_polygon_clip/data_dem/elevation.tif"   # 30 m UTM-30N grid


def _fetch_glo30(tr, shape, crs, dest: Path) -> None:
    import ee, requests, io
    ee.Initialize(project=GEE_PROJECT)
    coll = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    img = coll.mosaic().setDefaultProjection(coll.first().projection()).resample("bilinear")
    H, W = shape
    out = np.full(shape, np.nan, np.float32)
    n_tiles = 2
    rows = np.linspace(0, H, n_tiles + 1).astype(int)
    for a, b in zip(rows[:-1], rows[1:]):
        top = tr.f + a * tr.e
        bottom = tr.f + b * tr.e
        region = ee.Geometry.Rectangle([tr.c, bottom, tr.c + W * tr.a, top],
                                       proj=str(crs), geodesic=False)
        url = img.getDownloadURL({"name": "glo30", "crs": str(crs),
                                  "crs_transform": [tr.a, 0, tr.c, 0, tr.e, top],
                                  "region": region, "format": "GEO_TIFF"})
        r = requests.get(url, timeout=900)
        r.raise_for_status()
        with rasterio.MemoryFile(r.content) as mf, mf.open() as s:
            tile = np.full((b - a, W), np.nan, np.float32)
            reproject(rasterio.band(s, 1), tile, dst_transform=rasterio.Affine(tr.a, 0, tr.c, 0, tr.e, top),
                      dst_crs=crs, src_nodata=s.nodata, dst_nodata=np.nan, resampling=Resampling.nearest)
        out[a:b] = tile
        log.info("  tile rows %d-%d fetched (valid %.3f)", a, b, np.isfinite(tile).mean())
    prof = {"driver": "GTiff", "dtype": "float32", "count": 1, "height": H, "width": W,
            "crs": crs, "transform": tr, "nodata": np.nan, "compress": "lzw"}
    with rasterio.open(dest, "w", **prof) as dst:
        dst.write(out, 1)


def main() -> None:
    root = get_project_root()
    dem_dir = root / "data/processed/dem"
    raw = root / "data/raw/dem/glo30_gee_utm30_30m.tif"
    with rasterio.open(root / GRID_REF) as s:
        tr, crs, shape = s.transform, s.crs, s.shape
        profile = s.profile.copy()

    with Timer() as t:
        if not raw.exists():
            log.info("Fetching Copernicus GLO-30 from Earth Engine…")
            _fetch_glo30(tr, shape, crs, raw)
        with rasterio.open(raw) as s:
            elev = s.read(1).astype(np.float32)
        log.info("GLO-30 valid fraction: %.4f ; range %.1f to %.1f m",
                 np.isfinite(elev).mean(), np.nanmin(elev), np.nanmax(elev))
        assert np.isfinite(elev).all(), "GLO-30 grid has gaps"

        arr = elev.astype(np.float32)
        slope, aspect = compute_slope_aspect(arr, 30.0, NODATA_FILL)
        fa = _simple_flow_accum(arr, NODATA_FILL)
        twi = compute_twi(fa, slope, 30.0 ** 2, NODATA_FILL)
        rel = compute_relative_elevation(arr, NODATA_FILL)

        profile.update(driver="GTiff", dtype="float32", nodata=NODATA_FILL, compress="lzw")
        for name, a in {"elevation": arr, "slope": slope, "aspect": aspect,
                        "flow_accumulation": fa, "twi": twi, "relative_elevation": rel}.items():
            with rasterio.open(dem_dir / f"{name}.tif", "w", **profile) as dst:
                dst.write(a.astype(np.float32), 1)
            log.info("  wrote %-20s valid=%.4f", name, float((a != NODATA_FILL).mean()))
        log.info("relative-elevation floor (5th percentile): %.3f m", float(np.percentile(arr, 5)))

    log_run(script=__file__, input_paths=[str(raw)],
            output_paths=[str(dem_dir / f"{n}.tif") for n in
                          ("elevation", "slope", "aspect", "flow_accumulation", "twi", "relative_elevation")],
            parameters={"dem": "COPERNICUS/DEM/GLO30", "clip": "none", "resolution_m": 30},
            status="success", runtime_seconds=t.elapsed)


if __name__ == "__main__":
    main()
