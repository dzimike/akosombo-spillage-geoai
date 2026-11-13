from __future__ import annotations
"""
Phase 5 – Task 5.6: Community Exposure Summary

Aggregates building, road, population, agricultural, and facility exposure
into a single community-level table and GeoPackage for downstream use
in ML feature stacks and dashboard visualisation.

Outputs
-------
data/processed/exposure/community_exposure_summary.gpkg
outputs/tables/community_exposure_summary.csv
"""

import argparse
import logging

import geopandas as gpd
import pandas as pd

from src.utils.paths import load_config, get_processed_path, get_output_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def safe_read_csv(path, default_cols: list[str]) -> pd.DataFrame:
    """Read CSV, return empty DataFrame with default columns if missing."""
    if path.exists():
        return pd.read_csv(str(path))
    log.warning("Missing table: %s — filling with zeros", path.name)
    return pd.DataFrame(columns=default_cols)


def main(config_path: str | None = None) -> None:
    config     = load_config(config_path)
    exp_dir    = get_processed_path("exposure", config)
    tables_dir = get_output_path("tables", config)
    ensure_dir(exp_dir)
    ensure_dir(tables_dir)

    comm_path = get_processed_path("aoi/affected_communities.gpkg", config)
    if not comm_path.exists():
        log.error("Community layer missing: %s", comm_path)
        return

    with Timer() as t:
        # ---- Load community geometries ------------------------------------
        log.info("Loading community geometries…")
        comms = gpd.read_file(str(comm_path)).to_crs("EPSG:32630")
        comms["geometry"] = comms.geometry.buffer(10_000)   # 10 km service zone
        log.info("  Communities: %d", len(comms))

        # ---- Load all component tables -----------------------------------
        pop_table  = safe_read_csv(tables_dir / "population_exposure_by_community.csv",
                                   ["community", "total_pop_estimate", "exposed_pop_estimate", "pct_exposed"])
        bldg_table = safe_read_csv(tables_dir / "building_exposure_by_community.csv",
                                   ["community_name", "total_buildings", "exposed_buildings", "pct_exposed"])
        road_table = safe_read_csv(tables_dir / "road_exposure_by_community.csv",
                                   ["community_name", "flooded_road_km"])
        fac_table  = safe_read_csv(tables_dir / "facility_exposure_summary.csv",
                                   ["facility_type", "total", "directly_flooded", "within_500m_flood"])

        # ---- Normalise community name column -----------------------------
        pop_table  = pop_table.rename(columns={"community": "community_name"}) \
                     if "community" in pop_table.columns else pop_table
        bldg_table = bldg_table.rename(columns={"pct_exposed": "pct_buildings_exposed"}) \
                     if "pct_exposed" in bldg_table.columns else bldg_table
        pop_table  = pop_table.rename(columns={"pct_exposed": "pct_pop_exposed"}) \
                     if "pct_exposed" in pop_table.columns else pop_table

        # ---- Build community pivot for facilities -------------------------
        if not fac_table.empty and "facility_type" in fac_table.columns:
            fac_pivot = fac_table.set_index("facility_type")[
                ["total", "directly_flooded"]
            ].add_prefix("fac_")
            # Collapse to single totals (not per-community breakdown available)
            fac_totals = {
                "total_facilities":          int(fac_table["total"].sum()),
                "flooded_facilities_direct": int(fac_table["directly_flooded"].sum()),
                "flooded_facilities_500m":   int(fac_table.get("within_500m_flood", 0).sum()
                                                 if "within_500m_flood" in fac_table.columns else 0),
                "schools_flooded":           int(fac_table.loc[fac_table["facility_type"] == "school",
                                                                "directly_flooded"].sum()
                                                 if "facility_type" in fac_table.columns else 0),
                "health_facilities_flooded": int(fac_table.loc[
                    fac_table["facility_type"].isin(["hospital", "clinic", "health_post"]),
                    "directly_flooded"].sum()
                    if "facility_type" in fac_table.columns else 0),
            }
        else:
            fac_totals = {}

        # ---- Merge tables on community_name ------------------------------
        log.info("Merging exposure tables…")
        summary = comms[["community_name", "district", "geometry"]].copy()

        for df, keys in [
            (pop_table,  ["community_name", "total_pop_estimate",
                          "exposed_pop_estimate", "pct_pop_exposed"]),
            (bldg_table, ["community_name", "total_buildings",
                          "exposed_buildings", "pct_buildings_exposed"]),
            (road_table, ["community_name", "flooded_road_km"]),
        ]:
            if df.empty:
                continue
            cols = [c for c in keys if c in df.columns]
            if "community_name" in cols:
                summary = summary.merge(df[cols], on="community_name", how="left")

        # ---- Fill nulls with 0 for numeric cols --------------------------
        num_cols = [c for c in summary.columns
                    if c not in ("community_name", "district", "geometry")]
        summary[num_cols] = summary[num_cols].fillna(0)

        # ---- Facility totals (AOI-wide, repeated per community) ----------
        for k, v in fac_totals.items():
            summary[k] = v

        # ---- Add flood area (km²) per community from fused raster --------
        flood_conf_path = get_processed_path(
            "flood_extent/fused_flood_confidence.tif", config
        )
        if flood_conf_path.exists():
            try:
                import rasterio
                import rasterio.features as rfeat
                import numpy as np
                with rasterio.open(flood_conf_path) as src:
                    flood_class = src.read(1).astype(np.uint8)
                    flood_xform = src.transform

                flood_km2_vals = []
                for _, row in summary.iterrows():
                    try:
                        mask = rfeat.geometry_mask(
                            [row.geometry.__geo_interface__],
                            transform=flood_xform, invert=True,
                            out_shape=flood_class.shape
                        )
                        km2 = float(np.isin(flood_class[mask], [1, 2, 3]).sum() * 100 / 1e6)
                    except Exception:
                        km2 = 0.0
                    flood_km2_vals.append(round(km2, 3))
                summary["flood_area_km2"] = flood_km2_vals
            except Exception as e:
                log.warning("Could not compute per-community flood area: %s", e)
                summary["flood_area_km2"] = 0.0

        # ---- Save GeoPackage ---------------------------------------------
        out_gpkg = exp_dir / "community_exposure_summary.gpkg"
        summary.to_file(str(out_gpkg), driver="GPKG")
        log.info("  Saved: %s", out_gpkg)

        # ---- Save flat CSV (drop geometry) --------------------------------
        out_csv = tables_dir / "community_exposure_summary.csv"
        summary.drop(columns=["geometry"]).to_csv(str(out_csv), index=False)
        log.info("  Saved: %s", out_csv)

        # ---- Pretty print ------------------------------------------------
        display_cols = [
            "community_name", "exposed_pop_estimate", "exposed_buildings",
            "flooded_road_km", "flood_area_km2"
        ]
        available = [c for c in display_cols if c in summary.columns]
        log.info("\n%s", summary[available].to_string(index=False))

    log.info("\nCommunity exposure summary complete in %.1f s", t.elapsed)

    log_run(
        script=__file__,
        input_paths=[str(comm_path)],
        output_paths=[str(out_gpkg), str(out_csv)],
        parameters={"n_communities": len(comms)},
        status="success",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Community exposure summary")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
