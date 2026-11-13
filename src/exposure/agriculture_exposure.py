from __future__ import annotations
"""
Phase 5 – Task 5.4: Agricultural Exposure Analysis

Estimates cropland area exposed to flooding using NDVI-based cropland
identification and NDVI change (pre vs post event) as a vegetation stress proxy.

Methodology
-----------
1. Cropland mask: pixels where pre-event NDVI > 0.35 AND outside permanent water.
2. Exposed cropland: cropland pixels within the flood extent.
3. NDVI change (post − pre) within the flooded cropland → vegetation stress proxy.

IMPORTANT: Reports 'exposed cropland area' and 'NDVI change as proxy for
vegetation stress'. This is NOT a crop loss estimate; crop loss requires
field survey or yield calibration data.

Outputs
-------
data/processed/exposure/exposed_cropland.tif
data/processed/exposure/exposed_cropland.gpkg
outputs/tables/agricultural_exposure_summary.csv
outputs/tables/agricultural_exposure_by_district.csv
"""

import argparse
import csv
import logging

import numpy as np
import rasterio
from rasterio.enums import Resampling
import geopandas as gpd
import pandas as pd

from src.utils.paths import load_config, get_processed_path, get_output_path, ensure_dir
from src.utils.logging_utils import Timer, log_run
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NDVI_CROPLAND_THRESHOLD = 0.35   # min pre-event NDVI to consider as vegetation/cropland
NDVI_LOSS_THRESHOLD     = -0.10  # NDVI change below this → significant stress


def main(config_path: str | None = None) -> None:
    config     = load_config(config_path)
    exp_dir    = get_processed_path("exposure", config)
    sat_dir    = get_processed_path("satellite", config)
    tables_dir = get_output_path("tables", config)
    ensure_dir(exp_dir)
    ensure_dir(tables_dir)

    flood_path    = get_processed_path("flood_extent/fused_flood_confidence.tif", config)
    ndvi_pre_path  = sat_dir / "s2_ndvi_pre.tif"
    ndvi_post_path = sat_dir / "s2_ndvi_post.tif"
    ndvi_evt_path  = sat_dir / "s2_ndvi_event.tif"
    admin_path     = get_processed_path(
        "../../data/raw/admin/study_area_districts.gpkg", config
    )
    from src.utils.paths import get_project_root
    admin_path = get_project_root() / "data/raw/admin/study_area_districts.gpkg"
    comm_path  = get_processed_path("aoi/affected_communities.gpkg", config)

    for p in [flood_path, ndvi_pre_path, ndvi_post_path]:
        if not p.exists():
            log.error("Required file missing: %s", p); return

    with Timer() as t:
        # ---- Load flood extent (class band + confidence band) --------------
        log.info("Loading flood masks and NDVI rasters…")
        with rasterio.open(flood_path) as src:
            flood_class = src.read(1).astype(np.uint8)
            confidence  = src.read(2).astype(np.uint8)
            profile     = src.profile.copy()
            transform   = src.transform
            crs         = src.crs

        flooded_mask   = np.isin(flood_class, [1, 2, 3])   # any flood class
        perm_water_mask = flood_class == 4

        # ---- Load NDVI rasters (align to flood grid) ----------------------
        ndvi_pre  = read_aligned(str(ndvi_pre_path),  str(flood_path), Resampling.bilinear)
        ndvi_post = read_aligned(str(ndvi_post_path), str(flood_path), Resampling.bilinear)

        # Replace NaN with 0 (conservative — won't be counted as crop)
        ndvi_pre_valid  = np.where(np.isfinite(ndvi_pre),  ndvi_pre,  0.0).astype(np.float32)
        ndvi_post_valid = np.where(np.isfinite(ndvi_post), ndvi_post, 0.0).astype(np.float32)

        # ---- Cropland mask: high pre-event NDVI, not permanent water -------
        cropland_mask = (
            (ndvi_pre_valid >= NDVI_CROPLAND_THRESHOLD) &
            (~perm_water_mask)
        )
        log.info("  Cropland pixels (NDVI pre >= %.2f): %d  (%.2f km²)",
                 NDVI_CROPLAND_THRESHOLD, cropland_mask.sum(),
                 cropland_mask.sum() * 100 / 1e6)

        # ---- Exposed cropland = cropland ∩ flooded -------------------------
        exposed_crop_mask = cropland_mask & flooded_mask
        n_exposed = int(exposed_crop_mask.sum())
        exposed_km2 = n_exposed * 100 / 1e6
        log.info("  Exposed cropland: %d pixels  %.3f km²  (%.1f ha)",
                 n_exposed, exposed_km2, exposed_km2 * 100)

        # ---- NDVI change within exposed cropland --------------------------
        ndvi_change = ndvi_post_valid - ndvi_pre_valid
        ndvi_change_in_crop = np.where(exposed_crop_mask, ndvi_change, np.nan)

        valid_change = ndvi_change_in_crop[np.isfinite(ndvi_change_in_crop)]
        if valid_change.size > 0:
            mean_ndvi_loss  = float(np.nanmean(valid_change))
            n_stressed      = int((valid_change < NDVI_LOSS_THRESHOLD).sum())
            pct_stressed    = round(100 * n_stressed / len(valid_change), 1)
            log.info("  Mean NDVI change in exposed cropland: %.3f", mean_ndvi_loss)
            log.info("  Pixels with NDVI loss > %.2f (significant stress): %d  (%.1f%%)",
                     abs(NDVI_LOSS_THRESHOLD), n_stressed, pct_stressed)
        else:
            mean_ndvi_loss = 0.0
            n_stressed = 0
            pct_stressed = 0.0

        # ---- Write exposed cropland raster --------------------------------
        out_tif = exp_dir / "exposed_cropland.tif"
        out_profile = profile.copy()
        out_profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
        with rasterio.open(out_tif, "w", **out_profile) as dst:
            dst.write(exposed_crop_mask.astype(np.uint8), 1)
            dst.update_tags(description=(
                "Exposed cropland proxy: pixels with NDVI pre >= 0.35 "
                "within flood extent. NOT calibrated crop loss."))
        log.info("  Saved: %s", out_tif)

        # ---- Vectorise exposed cropland -----------------------------------
        from rasterio.features import shapes
        from shapely.geometry import shape
        polys = [shape(g) for g, v in
                 shapes(exposed_crop_mask.astype(np.uint8), transform=transform) if v == 1]
        crop_gdf = gpd.GeoDataFrame({"geometry": polys}, crs=crs)
        if not crop_gdf.empty:
            crop_gdf["area_ha"] = (crop_gdf.geometry.area / 1e4).round(2)
            out_gpkg = exp_dir / "exposed_cropland.gpkg"
            crop_gdf.to_file(str(out_gpkg), driver="GPKG")
            log.info("  Saved: %s", out_gpkg)
        else:
            out_gpkg = exp_dir / "exposed_cropland.gpkg"

        # ---- Summary CSV --------------------------------------------------
        summary_path = tables_dir / "agricultural_exposure_summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value", "note"])
            w.writerow(["cropland_mask_pixels",  int(cropland_mask.sum()),
                        f"NDVI pre >= {NDVI_CROPLAND_THRESHOLD}"])
            w.writerow(["cropland_area_km2",
                        round(cropland_mask.sum() * 100 / 1e6, 2), "proxy only"])
            w.writerow(["exposed_cropland_pixels", n_exposed, ""])
            w.writerow(["exposed_cropland_km2",    round(exposed_km2, 3), ""])
            w.writerow(["exposed_cropland_ha",     round(exposed_km2 * 100, 1), ""])
            w.writerow(["mean_ndvi_change_in_flood", round(mean_ndvi_loss, 4),
                        "post - pre NDVI; negative = vegetation stress"])
            w.writerow(["pixels_significant_stress", n_stressed,
                        f"NDVI change < {NDVI_LOSS_THRESHOLD}"])
            w.writerow(["pct_significant_stress", pct_stressed, ""])
            w.writerow(["note", "PROXY — not calibrated crop loss estimate", ""])
        log.info("  Saved summary: %s", summary_path)

        # ---- District-level breakdown (if admin polygons available) -------
        district_rows = []
        if admin_path.exists():
            try:
                admin = gpd.read_file(str(admin_path)).to_crs(crs)
                import rasterio.features as rfeat
                for _, row in admin.iterrows():
                    d_name = row.get("GID_2", row.get("NAME_2", "district"))
                    try:
                        d_mask = rfeat.geometry_mask(
                            [row.geometry.__geo_interface__],
                            transform=transform, invert=True,
                            out_shape=(profile["height"], profile["width"])
                        )
                        d_exp_km2 = float(
                            (exposed_crop_mask & d_mask).sum() * 100 / 1e6
                        )
                        if d_exp_km2 > 0:
                            district_rows.append({
                                "district": d_name,
                                "exposed_cropland_km2": round(d_exp_km2, 3),
                                "exposed_cropland_ha": round(d_exp_km2 * 100, 1),
                            })
                    except Exception:
                        pass
            except Exception as e:
                log.warning("District breakdown skipped: %s", e)

        if district_rows:
            dist_df = pd.DataFrame(district_rows).sort_values(
                "exposed_cropland_km2", ascending=False
            )
            dist_path = tables_dir / "agricultural_exposure_by_district.csv"
            dist_df.to_csv(str(dist_path), index=False)
            log.info("  Saved district breakdown: %s", dist_path)
            for _, r in dist_df.iterrows():
                log.info("    %-25s  %.1f ha exposed", r["district"],
                         r["exposed_cropland_ha"])

    log.info("Agricultural exposure complete in %.1f s", t.elapsed)

    log_run(
        script=__file__,
        input_paths=[str(flood_path), str(ndvi_pre_path), str(ndvi_post_path)],
        output_paths=[str(out_tif), str(summary_path)],
        parameters={"exposed_cropland_km2": round(exposed_km2, 3),
                    "mean_ndvi_change": round(mean_ndvi_loss, 4)},
        status="success",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agricultural exposure analysis")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
