from __future__ import annotations
"""
Phase 5 – Task 5.3: Population Exposure Analysis

Aligns WorldPop 2020 1 km raster to the flood confidence grid and sums
estimated resident population within each flood confidence class and per
community 10 km buffer zone.

IMPORTANT: Outputs are 'estimated exposed population', NOT confirmed
displaced or affected population. Displacement requires field verification.

Outputs
-------
data/processed/exposure/exposed_population_grid.tif
outputs/tables/population_exposure_summary.csv
outputs/tables/population_exposure_by_community.csv
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

COMMUNITY_BUFFER_M = 10_000   # 10 km buffer for community attribution
POP_NODATA         = -99999.0


def main(config_path: str | None = None) -> None:
    config    = load_config(config_path)
    exp_dir   = get_processed_path("exposure", config)
    tables_dir = get_output_path("tables", config)
    ensure_dir(exp_dir)
    ensure_dir(tables_dir)

    flood_path  = get_processed_path("flood_extent/fused_flood_confidence.tif", config)
    pop_path    = get_processed_path("population/population_grid_clipped.tif", config)
    comm_path   = get_processed_path("aoi/affected_communities.gpkg", config)

    for p in [flood_path, pop_path]:
        if not p.exists():
            log.error("Required file missing: %s", p); return

    with Timer() as t:
        # ---- Load flood confidence (band 2) aligned at 10 m ---------------
        log.info("Loading flood confidence raster…")
        with rasterio.open(flood_path) as src:
            confidence = src.read(2).astype(np.uint8)   # 0=none,1-3=low/med/high,4=perm
            profile    = src.profile.copy()
            transform  = src.transform
            crs        = src.crs

        flooded_mask = (confidence >= 1) & (confidence <= 3)

        # ---- Align WorldPop to flood grid ----------------------------------
        log.info("Aligning WorldPop raster to flood grid…")
        pop_aligned = read_aligned(str(pop_path), str(flood_path), Resampling.bilinear)
        # Mask nodata and negatives
        pop_valid = np.where(
            (pop_aligned > 0) & np.isfinite(pop_aligned), pop_aligned, 0.0
        ).astype(np.float32)

        # Each 10-m pixel covers 100 m²; WorldPop is people per km²
        # After alignment, each pixel value represents people per km² × (100/1e6) km² = people × 1e-4
        # So pop per pixel = value × (pixel_area_m2 / 1e6)
        pixel_area_km2 = 100 / 1e6   # 10 m × 10 m = 100 m²
        pop_per_pixel  = pop_valid * pixel_area_km2

        # ---- Overall flood exposure by confidence class --------------------
        conf_labels = {1: "low_confidence", 2: "medium_confidence", 3: "high_confidence"}
        totals = {}
        for code, label in conf_labels.items():
            mask = confidence == code
            totals[label] = float(np.nansum(pop_per_pixel[mask]))

        totals["all_flooded"] = float(np.nansum(pop_per_pixel[flooded_mask]))
        totals["total_aoi"]   = float(np.nansum(pop_per_pixel))

        log.info("  Estimated exposed population (all classes): %s",
                 f"{totals['all_flooded']:,.0f}")
        log.info("    High confidence:   %s", f"{totals['high_confidence']:,.0f}")
        log.info("    Medium confidence: %s", f"{totals['medium_confidence']:,.0f}")
        log.info("    Low confidence:    %s", f"{totals['low_confidence']:,.0f}")
        log.info("  Total AOI population estimate:              %s",
                 f"{totals['total_aoi']:,.0f}")

        # ---- Save exposed population grid ----------------------------------
        out_tif = exp_dir / "exposed_population_grid.tif"
        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1, nodata=-1, compress="lzw")
        exposed_pop_grid = np.where(flooded_mask, pop_per_pixel, -1).astype(np.float32)
        with rasterio.open(out_tif, "w", **out_profile) as dst:
            dst.write(exposed_pop_grid, 1)
            dst.update_tags(description=(
                "Estimated population per 10m pixel within flood extent. "
                "PROXY — WorldPop 2020 1km UN-adjusted resampled to 10m. "
                "NOT confirmed displaced population."))
        log.info("  Saved: %s", out_tif)

        # ---- Summary CSV ---------------------------------------------------
        summary_path = tables_dir / "population_exposure_summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value", "note"])
            w.writerow(["total_aoi_population",   round(totals["total_aoi"]),
                        "WorldPop 2020 1km UN-adjusted"])
            w.writerow(["exposed_pop_all_flood",  round(totals["all_flooded"]),
                        "ESTIMATED — not confirmed displaced"])
            w.writerow(["exposed_pop_high_conf",  round(totals["high_confidence"]), ""])
            w.writerow(["exposed_pop_medium_conf",round(totals["medium_confidence"]), ""])
            w.writerow(["exposed_pop_low_conf",   round(totals["low_confidence"]), ""])
            w.writerow(["pct_aoi_exposed",
                        round(100 * totals["all_flooded"] / max(totals["total_aoi"], 1), 2),
                        "% of total AOI population in flood zone"])
        log.info("  Saved: %s", summary_path)

        # ---- Community-level breakdown -------------------------------------
        comm_rows = []
        if comm_path.exists():
            communities = gpd.read_file(str(comm_path)).to_crs(crs)
            communities["geometry"] = communities.geometry.buffer(COMMUNITY_BUFFER_M)

            for _, row in communities.iterrows():
                name = row.get("community_name", "Unknown")
                # Burn community buffer to mask
                import rasterio.features as rfeat
                buf_mask = rfeat.geometry_mask(
                    [row.geometry.__geo_interface__],
                    transform=transform,
                    invert=True,
                    out_shape=(profile["height"], profile["width"]),
                )
                c_pop_total   = float(np.nansum(pop_per_pixel[buf_mask]))
                c_pop_exposed = float(np.nansum(pop_per_pixel[buf_mask & flooded_mask]))
                c_pct = round(100 * c_pop_exposed / max(c_pop_total, 1), 1)
                comm_rows.append({
                    "community": name,
                    "buffer_radius_km": COMMUNITY_BUFFER_M / 1000,
                    "total_pop_estimate": round(c_pop_total),
                    "exposed_pop_estimate": round(c_pop_exposed),
                    "pct_exposed": c_pct,
                    "note": "ESTIMATED — WorldPop 2020 1km resampled",
                })
            comm_df = pd.DataFrame(comm_rows)
            comm_path_out = tables_dir / "population_exposure_by_community.csv"
            comm_df.to_csv(str(comm_path_out), index=False)
            log.info("  Saved community breakdown: %s", comm_path_out)
            for r in comm_rows:
                log.info("    %-15s  exposed ≈ %6d  (%.1f%%)",
                         r["community"], r["exposed_pop_estimate"], r["pct_exposed"])

    log.info("Population exposure complete in %.1f s", t.elapsed)

    log_run(
        script=__file__,
        input_paths=[str(flood_path), str(pop_path)],
        output_paths=[str(out_tif), str(summary_path)],
        parameters={"exposed_pop": round(totals["all_flooded"]),
                    "pct_aoi": round(100*totals["all_flooded"]/max(totals["total_aoi"],1),2)},
        status="success",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population exposure analysis")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
