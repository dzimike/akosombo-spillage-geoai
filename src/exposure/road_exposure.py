from __future__ import annotations
"""
Phase 5 – Task 5.2: Road Exposure Analysis

Clips OSM road network to the fused flood extent and computes flooded road
length in km by highway class and community.

IMPORTANT: Reports road segments geometrically within the flood extent.
This is NOT a passability assessment; some flooded roads may remain
passable depending on water depth and road surface type.

Outputs
-------
data/processed/exposure/exposed_roads.gpkg
outputs/tables/road_exposure_by_class.csv
outputs/tables/road_exposure_by_community.csv
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

COMMUNITY_BUFFER_M = 10_000

# Road classes grouped for reporting
ROAD_CLASS_GROUPS = {
    "primary":      ["motorway", "trunk", "primary"],
    "secondary":    ["secondary", "tertiary"],
    "local":        ["unclassified", "residential", "service", "living_street"],
    "tracks_paths": ["track", "path", "footway", "cycleway", "bridleway"],
    "other":        [],
}


def classify_road(highway: str) -> str:
    for group, types in ROAD_CLASS_GROUPS.items():
        if highway in types:
            return group
    return "other"


def compute_road_exposure(roads: gpd.GeoDataFrame, flood: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add flooded_length_km: the length of each road segment inside the flood extent, in km (0.0 when outside).

    Same rule as main() (intersection with the dissolved flood extent); both inputs must share a projected CRS.
    """
    union = flood.geometry.union_all()
    out = roads.copy()
    out["flooded_length_km"] = (out.geometry.intersection(union).length / 1000).round(4)
    return out


def main(config_path: str | None = None) -> None:
    config     = load_config(config_path)
    exp_dir    = get_processed_path("exposure", config)
    tables_dir = get_output_path("tables", config)
    ensure_dir(exp_dir)
    ensure_dir(tables_dir)

    flood_path = get_processed_path("flood_extent/fused_flood_extent.gpkg", config)
    roads_path = get_project_root() / "data/raw/roads/osm_roads.gpkg"
    comm_path  = get_processed_path("aoi/affected_communities.gpkg", config)

    for p in [flood_path, roads_path]:
        if not p.exists():
            log.error("Required file missing: %s", p); return

    with Timer() as t:
        # ---- Load and reproject -------------------------------------------
        log.info("Loading roads and flood extent…")
        flood = to_projected(
            drop_empty_geometries(fix_geometries(gpd.read_file(str(flood_path))))
        )
        roads = to_projected(
            drop_empty_geometries(fix_geometries(gpd.read_file(str(roads_path))))
        )

        # Keep only line geometries
        roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
        roads["highway"] = roads["highway"].fillna("other").astype(str)
        roads["road_class"] = roads["highway"].apply(classify_road)
        roads["total_length_km"] = (roads.geometry.length / 1000).round(4)
        total_road_km = roads["total_length_km"].sum()

        log.info("  Roads: %d segments, %.1f km total", len(roads), total_road_km)

        # ---- Clip roads to flood extent -----------------------------------
        log.info("Clipping roads to flood extent…")
        flood_union = flood.geometry.union_all()
        try:
            flooded_roads = gpd.clip(roads, flood_union)
        except Exception as e:
            log.warning("gpd.clip failed (%s), using intersection fallback", e)
            flooded_roads = roads.copy()
            flooded_roads["geometry"] = roads.geometry.intersection(flood_union)
            flooded_roads = drop_empty_geometries(flooded_roads)

        flooded_roads = flooded_roads[
            flooded_roads.geometry.geom_type.isin(["LineString", "MultiLineString",
                                                    "GeometryCollection"])
        ].copy()
        flooded_roads["flooded_length_km"] = (flooded_roads.geometry.length / 1000).round(4)
        flooded_km = flooded_roads["flooded_length_km"].sum()

        log.info("  Flooded road segments: %d, %.2f km  (%.1f%% of network)",
                 len(flooded_roads), flooded_km,
                 100 * flooded_km / max(total_road_km, 1))

        # ---- Assign to community -----------------------------------------
        if comm_path.exists():
            comms = gpd.read_file(str(comm_path)).to_crs(roads.crs)
            comms["geometry"] = comms.geometry.buffer(COMMUNITY_BUFFER_M)
            road_comm = gpd.sjoin(
                flooded_roads[["road_class", "highway", "flooded_length_km", "geometry"]],
                comms[["community_name", "geometry"]],
                how="left", predicate="intersects"
            )
            road_comm_dedup = road_comm[~road_comm.index.duplicated(keep="first")]
            road_comm_dedup = road_comm_dedup.reindex(flooded_roads.index)
            road_comm_dedup["community_name"] = road_comm_dedup["community_name"].fillna("Other")
        else:
            road_comm_dedup = flooded_roads.copy()
            road_comm_dedup["community_name"] = "Unknown"

        flooded_roads["community_name"] = road_comm_dedup["community_name"].values

        # ---- Save exposed roads layer ------------------------------------
        out_gpkg = exp_dir / "exposed_roads.gpkg"
        flooded_roads.to_file(str(out_gpkg), driver="GPKG")
        log.info("  Saved: %s", out_gpkg)

        # ---- Summary by road class ----------------------------------------
        class_summary = (
            flooded_roads.groupby("road_class")
            .agg(
                n_segments         = ("flooded_length_km", "count"),
                flooded_length_km  = ("flooded_length_km", "sum"),
            )
            .reset_index()
        )
        # Add total road length per class for context
        total_per_class = (
            roads.groupby("road_class")["total_length_km"].sum().reset_index()
        )
        class_summary = class_summary.merge(total_per_class, on="road_class", how="left")
        class_summary["pct_flooded"] = (
            100 * class_summary["flooded_length_km"] /
            class_summary["total_length_km"].replace(0, 1)
        ).round(1)

        class_out = tables_dir / "road_exposure_by_class.csv"
        class_summary.to_csv(str(class_out), index=False)
        log.info("  Saved road class breakdown: %s", class_out)

        for _, r in class_summary.iterrows():
            log.info("    %-15s  %.1f km flooded / %.1f km total  (%.1f%%)",
                     r["road_class"], r["flooded_length_km"],
                     r["total_length_km"], r["pct_flooded"])

        # ---- Summary by community ----------------------------------------
        comm_road_summary = (
            flooded_roads.groupby("community_name")
            .agg(flooded_km = ("flooded_length_km", "sum"))
            .reset_index()
            .rename(columns={"flooded_km": "flooded_road_km"})
        )
        comm_out = tables_dir / "road_exposure_by_community.csv"
        comm_road_summary.to_csv(str(comm_out), index=False)
        log.info("  Saved community breakdown: %s", comm_out)

    log.info("Road exposure complete in %.1f s", t.elapsed)

    log_run(
        script=__file__,
        input_paths=[str(flood_path), str(roads_path)],
        output_paths=[str(out_gpkg), str(class_out)],
        parameters={"total_road_km": round(total_road_km, 1),
                    "flooded_road_km": round(flooded_km, 2)},
        status="success",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road exposure analysis")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
