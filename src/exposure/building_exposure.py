from __future__ import annotations
"""
Phase 5 – Task 5.1: Building Exposure Analysis

Spatially intersects OSM building footprints with the fused flood extent
to count exposed structures and their total footprint area.

Attribution to study communities uses a 10 km Voronoi/buffer zone.

Outputs
-------
data/processed/exposure/exposed_buildings.gpkg
outputs/tables/building_exposure_by_community.csv
outputs/tables/building_exposure_summary.csv
"""

import argparse
import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from src.utils.crs import to_projected
from src.utils.vector import fix_geometries, drop_empty_geometries
from src.utils.paths import load_config, get_processed_path, get_output_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COMMUNITY_BUFFER_M = 10_000


def load_and_prep(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    return to_projected(drop_empty_geometries(fix_geometries(gdf)))


def compute_building_exposure(buildings: gpd.GeoDataFrame, flood: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Flag each building 'exposed' when its footprint centroid lies inside the flood extent, else 'not_exposed'.

    Same rule as main() (centroid within the dissolved flood extent); both inputs must share a projected CRS.
    """
    union = flood.geometry.union_all()
    out = buildings.copy()
    out["flood_status"] = ["exposed" if c.within(union) else "not_exposed" for c in out.geometry.centroid]
    return out


def main(config_path: str | None = None) -> None:
    config     = load_config(config_path)
    exp_dir    = get_processed_path("exposure", config)
    tables_dir = get_output_path("tables", config)
    ensure_dir(exp_dir)
    ensure_dir(tables_dir)

    flood_path    = get_processed_path("flood_extent/fused_flood_extent.gpkg", config)
    buildings_path = get_processed_path("buildings/osm_buildings.gpkg", config)
    comm_path     = get_processed_path("aoi/affected_communities.gpkg", config)

    # The OSM buildings are in data/raw/buildings/
    if not buildings_path.exists():
        buildings_path = get_processed_path(
            "../../data/raw/buildings/osm_buildings.gpkg", config
        )
    # Fallback: find the raw path directly
    from src.utils.paths import get_project_root
    raw_buildings = get_project_root() / "data/raw/buildings/osm_buildings.gpkg"

    for p in [flood_path, raw_buildings]:
        if not p.exists():
            log.error("Required file missing: %s", p); return

    with Timer() as t:
        # ---- Load flood extent --------------------------------------------
        log.info("Loading fused flood extent…")
        flood = load_and_prep(str(flood_path))
        flood_union = flood.geometry.union_all()
        flood_area_km2 = flood.geometry.area.sum() / 1e6
        log.info("  Flood extent: %d polygons, %.2f km²", len(flood), flood_area_km2)

        # ---- Load buildings -----------------------------------------------
        log.info("Loading OSM buildings…")
        buildings = load_and_prep(str(raw_buildings))
        buildings["footprint_area_m2"] = buildings.geometry.area.round(2)
        total_bldg = len(buildings)
        log.info("  Buildings: %d", total_bldg)

        # ---- Spatial join: buildings intersecting flood --------------------
        log.info("Joining buildings with flood extent…")
        flood_simple = gpd.GeoDataFrame(
            {"geometry": [flood_union]}, crs=flood.crs
        )
        # Use centroid for polygon buildings (faster, avoids edge artefacts)
        bldg_pts = buildings.copy()
        bldg_pts["geometry"] = buildings.geometry.centroid

        joined = gpd.sjoin(bldg_pts, flood_simple, how="left", predicate="within")
        # Drop duplicates: keep one row per building (index = original building index)
        joined_dedup = joined[~joined.index.duplicated(keep="first")]
        # Re-index to match buildings
        buildings["flood_exposed"] = joined_dedup.reindex(buildings.index)["index_right"].notna()

        n_exposed = buildings["flood_exposed"].sum()
        exposed_area_m2 = buildings.loc[buildings["flood_exposed"], "footprint_area_m2"].sum()

        log.info("  Exposed buildings: %d / %d  (%.1f%%)",
                 n_exposed, total_bldg, 100 * n_exposed / max(total_bldg, 1))
        log.info("  Exposed footprint area: %.1f ha", exposed_area_m2 / 1e4)

        # ---- Assign to nearest community ----------------------------------
        if comm_path.exists():
            comms = gpd.read_file(str(comm_path)).to_crs(buildings.crs)
            comms_buf = comms.copy()
            comms_buf["geometry"] = comms.geometry.buffer(COMMUNITY_BUFFER_M)

            comm_join = gpd.sjoin(
                buildings[["flood_exposed", "footprint_area_m2", "geometry"]],
                comms_buf[["community_name", "geometry"]],
                how="left", predicate="within"
            )
            comm_join_dedup = comm_join[~comm_join.index.duplicated(keep="first")]
            buildings["community_name"] = comm_join_dedup.reindex(buildings.index)["community_name"].values
        else:
            buildings["community_name"] = "Unknown"

        # ---- Save exposed buildings layer ---------------------------------
        out_gpkg = exp_dir / "exposed_buildings.gpkg"
        exposed_gdf = buildings[buildings["flood_exposed"]].copy()
        exposed_gdf.to_file(str(out_gpkg), driver="GPKG")
        log.info("  Saved: %s", out_gpkg)

        # ---- Summary by community ----------------------------------------
        comm_summary = (
            buildings.groupby("community_name")
            .agg(
                total_buildings   = ("flood_exposed", "count"),
                exposed_buildings = ("flood_exposed", "sum"),
                exposed_area_m2   = ("footprint_area_m2",
                                     lambda x: x[buildings.loc[x.index, "flood_exposed"]].sum()),
            )
            .reset_index()
        )
        comm_summary["pct_exposed"] = (
            100 * comm_summary["exposed_buildings"] /
            comm_summary["total_buildings"].replace(0, 1)
        ).round(1)

        comm_out = tables_dir / "building_exposure_by_community.csv"
        comm_summary.to_csv(str(comm_out), index=False)
        log.info("  Saved community breakdown: %s", comm_out)

        # ---- Overall summary CSV -----------------------------------------
        building_types = {}
        if "building" in buildings.columns:
            building_types = (
                buildings[buildings["flood_exposed"]]["building"]
                .fillna("unknown").value_counts().head(10).to_dict()
            )

        summary_path = tables_dir / "building_exposure_summary.csv"
        with open(summary_path, "w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["total_buildings_in_aoi",   total_bldg])
            w.writerow(["exposed_buildings",         int(n_exposed)])
            w.writerow(["pct_buildings_exposed",
                        round(100 * n_exposed / max(total_bldg, 1), 1)])
            w.writerow(["exposed_footprint_area_ha", round(exposed_area_m2 / 1e4, 2)])
            w.writerow(["flood_extent_km2",          round(flood_area_km2, 2)])
            for btype, cnt in building_types.items():
                w.writerow([f"exposed_type_{btype}", int(cnt)])
        log.info("  Saved summary: %s", summary_path)

        for _, r in comm_summary.iterrows():
            log.info("    %-15s  exposed: %4d / %4d  (%.1f%%)",
                     r["community_name"], r["exposed_buildings"],
                     r["total_buildings"], r["pct_exposed"])

    log.info("Building exposure complete in %.1f s", t.elapsed)

    log_run(
        script=__file__,
        input_paths=[str(flood_path), str(raw_buildings)],
        output_paths=[str(out_gpkg), str(summary_path)],
        parameters={"exposed_buildings": int(n_exposed),
                    "pct_exposed": round(100 * n_exposed / max(total_bldg, 1), 1)},
        status="success",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building exposure analysis")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
