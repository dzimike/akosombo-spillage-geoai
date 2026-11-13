from __future__ import annotations
"""
Phase 5 – Task 5.5: Facility Exposure Analysis

Flags schools, health facilities, and markets that are directly within the
flood extent or within 500 m of it (access disruption risk).

Access disruption is assessed by checking whether the nearest road segment
is also flooded.

Outputs
-------
data/processed/exposure/exposed_facilities.gpkg
outputs/tables/facility_exposure_summary.csv
"""

import argparse
import csv
import logging

import geopandas as gpd
import pandas as pd

from src.utils.crs import to_projected
from src.utils.vector import fix_geometries, drop_empty_geometries
from src.utils.paths import load_config, get_processed_path, get_output_path, ensure_dir, get_project_root
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUFFER_NEAR_M = 500    # metres — "near-flood" threshold
FACILITY_TYPES = {"school", "hospital", "clinic", "health_post",
                  "pharmacy", "marketplace"}


def main(config_path: str | None = None) -> None:
    config     = load_config(config_path)
    exp_dir    = get_processed_path("exposure", config)
    tables_dir = get_output_path("tables", config)
    ensure_dir(exp_dir)
    ensure_dir(tables_dir)

    flood_path    = get_processed_path("flood_extent/fused_flood_extent.gpkg", config)
    fac_path      = get_project_root() / "data/raw/health_education/osm_facilities.gpkg"
    roads_exp_path = exp_dir / "exposed_roads.gpkg"
    comm_path     = get_processed_path("aoi/affected_communities.gpkg", config)

    for p in [flood_path, fac_path]:
        if not p.exists():
            log.error("Required file missing: %s", p); return

    with Timer() as t:
        # ---- Load and reproject -------------------------------------------
        log.info("Loading facilities and flood extent…")
        flood_raw = to_projected(
            drop_empty_geometries(fix_geometries(gpd.read_file(str(flood_path))))
        )
        fac = to_projected(
            drop_empty_geometries(fix_geometries(gpd.read_file(str(fac_path))))
        )

        # Ensure point geometry (use centroid for any polygons)
        fac = fac.copy()
        fac["geometry"] = fac.geometry.centroid

        # Normalise facility type
        fac["facility_type"] = "unknown"
        if "amenity" in fac.columns:
            fac["facility_type"] = fac["amenity"].fillna("unknown")
        if "healthcare" in fac.columns:
            mask = fac["healthcare"].notna() & (fac["facility_type"] == "unknown")
            fac.loc[mask, "facility_type"] = fac.loc[mask, "healthcare"]

        log.info("  Facilities: %d", len(fac))
        log.info("  Type breakdown: %s",
                 fac["facility_type"].value_counts().to_dict())

        # Simplify flood for fast overlay (keep topology)
        log.info("  Simplifying flood polygons for overlay…")
        flood_simple = flood_raw.copy()
        flood_simple["geometry"] = flood_raw.geometry.simplify(30, preserve_topology=True)
        flood_simple = gpd.GeoDataFrame(
            {"geometry": [flood_simple.geometry.buffer(0).union_all()]},
            crs=flood_raw.crs
        )

        # ---- Direct flood exposure via spatial join -----------------------
        fac_direct = gpd.sjoin(
            fac[["facility_type", "geometry"]],
            flood_simple, how="left", predicate="within"
        )
        fac_direct_dedup = fac_direct[~fac_direct.index.duplicated(keep="first")]
        fac["directly_flooded"] = fac_direct_dedup.reindex(fac.index)["index_right"].notna().values

        # ---- Near-flood (within 500 m) — buffer facilities, join to flood ---
        fac_buf = fac.copy()
        fac_buf["geometry"] = fac.geometry.buffer(BUFFER_NEAR_M)
        fac_near = gpd.sjoin(
            fac_buf[["facility_type", "geometry"]],
            flood_simple, how="left", predicate="intersects"
        )
        fac_near_dedup = fac_near[~fac_near.index.duplicated(keep="first")]
        fac["within_500m_flood"] = fac_near_dedup.reindex(fac.index)["index_right"].notna().values

        # ---- Access disruption: within 500m of a flooded road? -----------
        fac["access_road_flooded"] = False
        if roads_exp_path.exists():
            exp_roads = gpd.read_file(str(roads_exp_path))
            if not exp_roads.empty:
                fac_road = gpd.sjoin(
                    fac_buf[["facility_type", "geometry"]],
                    exp_roads[["geometry"]].reset_index(drop=True),
                    how="left", predicate="intersects"
                )
                fac_road_dedup = fac_road[~fac_road.index.duplicated(keep="first")]
                fac["access_road_flooded"] = (
                    fac_road_dedup.reindex(fac.index)["index_right"].notna().values
                )

        # ---- Assign to community ------------------------------------------
        fac["community_name"] = "Unknown"
        if comm_path.exists():
            comms = gpd.read_file(str(comm_path)).to_crs(fac.crs)
            comms["geometry"] = comms.geometry.buffer(10_000)
            joined = gpd.sjoin(
                fac[["facility_type", "directly_flooded", "within_500m_flood",
                      "access_road_flooded", "geometry"]],
                comms[["community_name", "geometry"]],
                how="left", predicate="within"
            )
            joined_dedup = joined[~joined.index.duplicated(keep="first")]
            fac["community_name"] = joined_dedup.reindex(fac.index)["community_name"].values

        # ---- Save exposed facilities layer --------------------------------
        out_gpkg = exp_dir / "exposed_facilities.gpkg"
        exposed = fac[fac["directly_flooded"] | fac["within_500m_flood"]].copy()
        exposed.to_file(str(out_gpkg), driver="GPKG")
        log.info("  Saved: %s", out_gpkg)
        log.info("  Directly flooded: %d", fac["directly_flooded"].sum())
        log.info("  Within 500 m of flood: %d", fac["within_500m_flood"].sum())
        log.info("  Access road flooded: %d", fac["access_road_flooded"].sum())

        # ---- Summary by facility type ------------------------------------
        summary_rows = []
        for ftype in sorted(fac["facility_type"].unique()):
            sub = fac[fac["facility_type"] == ftype]
            summary_rows.append({
                "facility_type":       ftype,
                "total":               len(sub),
                "directly_flooded":    int(sub["directly_flooded"].sum()),
                "within_500m_flood":   int(sub["within_500m_flood"].sum()),
                "access_road_flooded": int(sub["access_road_flooded"].sum()),
            })
            if sub["directly_flooded"].sum() > 0 or sub["within_500m_flood"].sum() > 0:
                log.info("    %-15s  flooded: %d  near-flood: %d  road: %d",
                         ftype, sub["directly_flooded"].sum(),
                         sub["within_500m_flood"].sum(),
                         sub["access_road_flooded"].sum())

        summary_df = pd.DataFrame(summary_rows)
        summary_path = tables_dir / "facility_exposure_summary.csv"
        summary_df.to_csv(str(summary_path), index=False)
        log.info("  Saved summary: %s", summary_path)

    log.info("Facility exposure complete in %.1f s", t.elapsed)

    log_run(
        script=__file__,
        input_paths=[str(flood_path), str(fac_path)],
        output_paths=[str(out_gpkg), str(summary_path)],
        parameters={"directly_flooded": int(fac["directly_flooded"].sum()),
                    "within_500m": int(fac["within_500m_flood"].sum())},
        status="success",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facility exposure analysis")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
