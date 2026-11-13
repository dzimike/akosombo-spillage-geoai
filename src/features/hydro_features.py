from __future__ import annotations
"""
Phase 6 – Task 6.2: Hydrological Feature Engineering

Rasterizes OSM waterways and dam locations, then computes Euclidean
distance-transform proximity rasters at 10 m resolution.

Features produced
-----------------
dist_to_waterway   — distance (m) to nearest OSM waterway
dist_to_dam        — distance (m) to nearest dam (Akosombo / Kpong)
dist_to_volta      — distance (m) to main Volta River channel
                      (approximated from MNDWI pre-event water extent)

Notes
-----
Distance computation uses scipy.ndimage.distance_transform_edt which
runs in O(N) and is memory-efficient for large rasters.

Outputs
-------
data/processed/features/hydro/  (one GeoTIFF per feature)
"""

import argparse
import logging

import numpy as np
import rasterio
from rasterio.enums import Resampling
import rasterio.features
import geopandas as gpd
from scipy.ndimage import distance_transform_edt

from src.utils.paths import load_config, get_processed_path, ensure_dir, get_project_root
from src.utils.logging_utils import Timer, log_run
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEMPLATE_RASTER = "satellite/s2_ndwi_event.tif"


def rasterize_vector(gdf: gpd.GeoDataFrame, template_path: str,
                     burn_value: float = 1.0) -> np.ndarray:
    """Rasterize vector geometries to a binary grid matching the template."""
    with rasterio.open(template_path) as src:
        out = np.zeros((src.height, src.width), dtype=np.uint8)
        if gdf.empty:
            return out
        geoms = [(geom, burn_value) for geom in gdf.geometry]
        rasterio.features.rasterize(
            geoms, out=out,
            transform=src.transform,
            fill=0, all_touched=True,
        )
    return out


def distance_raster_m(binary: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """
    Euclidean distance (metres) from each 0-pixel to the nearest 1-pixel.
    Pixels already on features get distance = 0.
    """
    no_feature = (binary == 0).astype(bool)
    dist_px = distance_transform_edt(no_feature)
    return (dist_px * pixel_size_m).astype(np.float32)


def write_feature(arr: np.ndarray, profile: dict, path, description: str) -> None:
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=-1, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr, 1)
        dst.update_tags(description=description)


def main(config_path: str | None = None) -> None:
    config   = load_config(config_path)
    root     = get_project_root()
    sat_dir  = get_processed_path("satellite", config)
    feat_dir = get_processed_path("features/hydro", config)
    ensure_dir(feat_dir)

    template    = get_processed_path(TEMPLATE_RASTER, config)
    ww_path     = root / "data/raw/roads/osm_waterways.gpkg"
    dams_path   = get_processed_path("aoi/dam_locations.gpkg", config)
    mndwi_pre   = sat_dir / "s2_mndwi_pre.tif"

    with rasterio.open(template) as src:
        profile    = src.profile.copy()
        pixel_size = src.res[0]   # 10 m
        crs        = src.crs

    with Timer() as t:
        # ---- Distance to OSM waterways ------------------------------------
        log.info("Computing distance to OSM waterways…")
        if ww_path.exists():
            ww = gpd.read_file(str(ww_path)).to_crs(crs)
            ww_bin = rasterize_vector(ww, str(template))
            log.info("  Waterway pixels: %d", ww_bin.sum())
        else:
            log.warning("  Waterways not found — using zero raster")
            with rasterio.open(template) as src:
                ww_bin = np.zeros((src.height, src.width), dtype=np.uint8)

        dist_ww = distance_raster_m(ww_bin, pixel_size)
        out_ww = feat_dir / "dist_to_waterway.tif"
        write_feature(dist_ww, profile, out_ww,
                      "Distance (m) to nearest OSM waterway")
        log.info("  dist_to_waterway: range=[%.0f, %.0f] m",
                 float(dist_ww.min()), float(dist_ww.max()))

        # ---- Distance to dams ---------------------------------------------
        log.info("Computing distance to dams (Akosombo / Kpong)…")
        if dams_path.exists():
            dams = gpd.read_file(str(dams_path)).to_crs(crs)
            # Convert point GDF to use projected coordinates
            dams_bin = rasterize_vector(dams, str(template))
            log.info("  Dam pixels: %d", dams_bin.sum())
        else:
            log.warning("  Dams not found — using zero raster")
            with rasterio.open(template) as src:
                dams_bin = np.zeros((src.height, src.width), dtype=np.uint8)

        dist_dams = distance_raster_m(dams_bin, pixel_size)
        out_dams = feat_dir / "dist_to_dam.tif"
        write_feature(dist_dams, profile, out_dams,
                      "Distance (m) to nearest dam (Akosombo/Kpong)")
        log.info("  dist_to_dam: range=[%.0f, %.0f] m",
                 float(dist_dams.min()), float(dist_dams.max()))

        # ---- Distance to permanent water (Volta River extent) -------------
        log.info("Computing distance to Volta main channel (from MNDWI pre)…")
        if mndwi_pre.exists():
            mndwi_arr = read_aligned(str(mndwi_pre), str(template), Resampling.bilinear)
            # Main river = pixels with high pre-event MNDWI (MNDWI > 0.3 = open water)
            volta_bin = (np.isfinite(mndwi_arr) & (mndwi_arr > 0.3)).astype(np.uint8)
            log.info("  Volta water pixels: %d", int(volta_bin.sum()))
        else:
            log.warning("  MNDWI pre not found — using waterway raster as proxy")
            volta_bin = ww_bin.copy()

        dist_volta = distance_raster_m(volta_bin, pixel_size)
        out_volta = feat_dir / "dist_to_volta.tif"
        write_feature(dist_volta, profile, out_volta,
                      "Distance (m) to Volta River main channel (MNDWI pre proxy)")
        log.info("  dist_to_volta: range=[%.0f, %.0f] m",
                 float(dist_volta.min()), float(dist_volta.max()))

    log.info("Hydrological features complete in %.1f s", t.elapsed)

    outputs = [out_ww, out_dams, out_volta]
    log_run(
        script=__file__,
        input_paths=[str(ww_path), str(dams_path)],
        output_paths=[str(o) for o in outputs],
        parameters={"n_features": len(outputs), "pixel_size_m": pixel_size},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydrological feature engineering")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
