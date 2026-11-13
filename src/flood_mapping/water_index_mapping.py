from __future__ import annotations
"""
Phase 4 – Task 4.2: Optical Water Index Mapping

Uses Sentinel-2 NDWI and MNDWI event-period composites to map surface water.
High-confidence water requires BOTH indices to exceed their thresholds,
reducing commission errors from cloud shadows and dark surfaces.

Permanent water (JRC, open sea, pre-event optical water) is subtracted to retain only flood-event water.
The result complements SAR-based detection in cloud-free areas.

Usage
-----
    python src/flood_mapping/water_index_mapping.py --config config/config.yaml

Outputs
-------
data/processed/flood_extent/optical_water_mask.tif
data/processed/flood_extent/optical_water_extent.gpkg
data/processed/flood_extent/optical_threshold_report.csv
"""

import argparse
import csv
import logging

import numpy as np
import rasterio
from rasterio.enums import Resampling

from src.utils.paths import load_config, get_processed_path, ensure_dir
from src.utils.logging_utils import Timer, log_run
from src.utils.raster import read_aligned
from src.flood_mapping.water_masks import static_water

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NDWI_THRESHOLD  = 0.0
MNDWI_THRESHOLD = 0.0
MIN_MAPPING_UNIT_PIXELS = 10


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def detect_water_ndwi(
    ndwi: np.ndarray, threshold: float = NDWI_THRESHOLD
) -> np.ndarray:
    """Binary water mask from NDWI >= threshold (finite pixels only)."""
    return (np.isfinite(ndwi) & (ndwi >= threshold)).astype(np.uint8)


def detect_water_mndwi(
    mndwi: np.ndarray, threshold: float = MNDWI_THRESHOLD
) -> np.ndarray:
    """Binary water mask from MNDWI >= threshold (finite pixels only)."""
    return (np.isfinite(mndwi) & (mndwi >= threshold)).astype(np.uint8)


def combine_optical_masks(
    ndwi_mask: np.ndarray, mndwi_mask: np.ndarray
) -> np.ndarray:
    """Require agreement between NDWI and MNDWI for high-confidence water."""
    return ((ndwi_mask == 1) & (mndwi_mask == 1)).astype(np.uint8)


def remove_small_patches(
    mask: np.ndarray, min_pixels: int = MIN_MAPPING_UNIT_PIXELS
) -> np.ndarray:
    """Remove connected components smaller than the minimum mapping unit."""
    from scipy import ndimage
    labeled, _ = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, labeled.max() + 1))
    small_ids = [i + 1 for i, s in enumerate(sizes) if s < min_pixels]
    return np.where(np.isin(labeled, small_ids), 0, mask).astype(np.uint8)


def raster_to_polygon(mask: np.ndarray, transform, crs):
    """Convert binary raster to GeoDataFrame of polygons."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    polys = [shape(g) for g, v in shapes(mask, transform=transform) if v == 1]
    gdf = gpd.GeoDataFrame({"geometry": polys}, crs=crs)
    if not gdf.empty:
        gdf["area_ha"] = (gdf.geometry.area / 1e4).round(4)
    return gdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    config    = load_config(config_path)
    sat_dir   = get_processed_path("satellite", config)
    flood_dir = get_processed_path("flood_extent", config)
    ensure_dir(flood_dir)

    ndwi_event_path  = sat_dir / "s2_ndwi_event.tif"
    mndwi_event_path = sat_dir / "s2_mndwi_event.tif"
    jrc_path         = sat_dir / "jrc_permanent_water.tif"
    mndwi_pre_path   = sat_dir / "s2_mndwi_pre.tif"
    ndwi_pre_path    = sat_dir / "s2_ndwi_pre.tif"

    for p in [ndwi_event_path, mndwi_event_path]:
        if not p.exists():
            log.error("Required file missing: %s", p)
            return

    with Timer() as t:
        # ---- Load event-period water indices ------------------------------
        log.info("Loading S2 water index composites (event period)…")
        with rasterio.open(ndwi_event_path) as src:
            ndwi_event  = src.read(1).astype(np.float32)
            profile     = src.profile.copy()
            transform   = src.transform
            crs         = src.crs

        with rasterio.open(mndwi_event_path) as src:
            mndwi_event = src.read(1).astype(np.float32)

        valid_ndwi  = np.isfinite(ndwi_event).sum()
        valid_mndwi = np.isfinite(mndwi_event).sum()
        log.info("  Valid NDWI pixels:  %d / %d (%.1f%%)",
                 valid_ndwi, ndwi_event.size,
                 100 * valid_ndwi / ndwi_event.size)
        log.info("  Valid MNDWI pixels: %d / %d (%.1f%%)",
                 valid_mndwi, mndwi_event.size,
                 100 * valid_mndwi / mndwi_event.size)

        # ---- Apply index thresholds ---------------------------------------
        log.info("Applying water thresholds (NDWI >= %.1f, MNDWI >= %.1f)…",
                 NDWI_THRESHOLD, MNDWI_THRESHOLD)
        ndwi_water  = detect_water_ndwi(ndwi_event)
        mndwi_water = detect_water_mndwi(mndwi_event)
        water_mask  = combine_optical_masks(ndwi_water, mndwi_water)
        log.info("  NDWI water pixels:  %d", ndwi_water.sum())
        log.info("  MNDWI water pixels: %d", mndwi_water.sum())
        log.info("  Combined (AND):     %d", water_mask.sum())

        # ---- Permanent / pre-existing water mask ---------------------------
        log.info("Building static water mask (JRC + open sea + pre-event optical)…")
        comps = static_water(ndwi_event_path, sat_dir, flood_dir)
        for k, v in comps.items():
            log.info("  %-12s %.1f km2", k, v.sum() / 1e4)
        perm_water = comps["union"].astype(np.uint8)

        perm_pixels = int(perm_water.sum())
        water_mask  = np.where(perm_water.astype(bool), 0, water_mask).astype(np.uint8)
        log.info("  Permanent water pixels removed: %d", perm_pixels)
        log.info("  Flood-only water after mask:    %d", water_mask.sum())

        # ---- MMU filter ---------------------------------------------------
        log.info("Removing patches < %d pixels…", MIN_MAPPING_UNIT_PIXELS)
        water_mask = remove_small_patches(water_mask)
        n_water    = int(water_mask.sum())
        area_km2   = n_water * 100 / 1e6
        log.info("  Final optical water pixels: %d  Area: %.2f km²",
                 n_water, area_km2)

        # ---- Write binary raster ------------------------------------------
        out_tif = flood_dir / "optical_water_mask.tif"
        out_profile = profile.copy()
        out_profile.update(dtype="uint8", count=1, nodata=255, compress="lzw")
        with rasterio.open(out_tif, "w", **out_profile) as dst:
            dst.write(water_mask, 1)
        log.info("  Saved: %s", out_tif)

        # ---- Vectorise ----------------------------------------------------
        log.info("Vectorising optical water mask…")
        water_poly = raster_to_polygon(water_mask, transform, crs)
        n_polys    = len(water_poly)
        log.info("  Polygons: %d  (total %.2f ha)",
                 n_polys,
                 water_poly["area_ha"].sum() if not water_poly.empty else 0)

        out_gpkg = flood_dir / "optical_water_extent.gpkg"
        if not water_poly.empty:
            water_poly.to_file(str(out_gpkg), driver="GPKG")
            log.info("  Saved: %s", out_gpkg)

        # ---- Report -------------------------------------------------------
        report_path = flood_dir / "optical_threshold_report.csv"
        with open(report_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["ndwi_threshold",   NDWI_THRESHOLD])
            w.writerow(["mndwi_threshold",  MNDWI_THRESHOLD])
            w.writerow(["water_pixels_ndwi",  int(ndwi_water.sum())])
            w.writerow(["water_pixels_mndwi", int(mndwi_water.sum())])
            w.writerow(["water_pixels_combined", n_water])
            w.writerow(["water_area_km2",    round(area_km2, 4)])
            w.writerow(["water_polygons",    n_polys])
            w.writerow(["permanent_water_source", "JRC>=50 | GLO-30 sea mask | pre-event NDWI+MNDWI"])
            for k, v in comps.items():
                w.writerow([f"static_water_{k}_km2", round(float(v.sum()) / 1e4, 2)])
        log.info("  Saved: %s", report_path)

    log.info("Optical water mapping complete in %.1f s", t.elapsed)

    log_run(
        script=__file__,
        input_paths=[str(ndwi_event_path), str(mndwi_event_path)],
        output_paths=[str(out_tif), str(out_gpkg)],
        parameters={"ndwi_thresh": NDWI_THRESHOLD,
                    "mndwi_thresh": MNDWI_THRESHOLD,
                    "area_km2": round(area_km2, 2)},
        status="success",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical water index mapping")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
