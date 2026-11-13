from __future__ import annotations
"""
Phase 1 – Task 1.1: AOI Construction

Builds the Lower Volta Basin Area of Interest by:
1. Loading Ghana district boundaries and affected community seed points.
2. Loading the Lower Volta River corridor downstream of the dams.
3. Buffering the river at 10 km, 20 km, and 30 km for sensitivity testing.
4. Exporting the default 20 km buffer AOI as GeoPackage and GeoJSON.
5. Generating an AOI overview map.

Outputs
-------
data/processed/aoi/lower_volta_aoi.gpkg
data/processed/aoi/lower_volta_aoi.geojson
data/processed/aoi/affected_communities.gpkg
data/processed/aoi/dam_locations.gpkg
outputs/maps/aoi_overview.png
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

from src.utils.crs import PROJ_CRS, GEO_CRS, to_projected
from src.utils.logging_utils import Timer, log_run
from src.utils.paths import load_config, get_processed_path, get_output_path, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Affected community seed points (lon, lat) – confirmed from public reporting.
# TODO: replace with official Ghana Statistical Service community coordinates.
AFFECTED_COMMUNITIES = [
    {"community_name": "Mepe", "district": "North Tongu", "region": "Volta", "lon": 0.4833, "lat": 6.1500},
    {"community_name": "Battor", "district": "North Tongu", "region": "Volta", "lon": 0.5167, "lat": 6.2167},
    {"community_name": "Sogakope", "district": "South Tongu", "region": "Volta", "lon": 0.5833, "lat": 5.9167},
    {"community_name": "Mafi Kumase", "district": "Central Tongu", "region": "Volta", "lon": 0.3667, "lat": 6.0833},
    {"community_name": "Adidome", "district": "Central Tongu", "region": "Volta", "lon": 0.3833, "lat": 6.1167},
    {"community_name": "Ada Foah", "district": "Ada East", "region": "Greater Accra", "lon": 0.6333, "lat": 5.7833},
]

# Dam locations – from public sources.
DAM_LOCATIONS = [
    {"name": "Akosombo Dam", "lon": 0.0537, "lat": 6.2988},
    {"name": "Kpong Dam", "lon": 0.0819, "lat": 6.1491},
]


def build_community_gdf() -> gpd.GeoDataFrame:
    df = pd.DataFrame(AFFECTED_COMMUNITIES)
    geometry = [Point(row.lon, row.lat) for row in df.itertuples()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=GEO_CRS)
    return gdf


def build_dam_gdf() -> gpd.GeoDataFrame:
    df = pd.DataFrame(DAM_LOCATIONS)
    geometry = [Point(row.lon, row.lat) for row in df.itertuples()]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=GEO_CRS)


def build_river_aoi(communities_proj: gpd.GeoDataFrame, buffer_km: float) -> gpd.GeoDataFrame:
    """
    Create a buffered AOI from community seed points.

    TODO: replace the convex-hull placeholder with actual Lower Volta River
    geometry loaded from HydroSHEDS / MERIT Hydro once data are available.
    """
    buffer_m = buffer_km * 1000
    buffered = communities_proj.copy()
    buffered["geometry"] = communities_proj.geometry.buffer(buffer_m)
    aoi = buffered.dissolve()
    aoi["buffer_km"] = buffer_km
    aoi["description"] = f"Lower Volta AOI – {buffer_km} km community buffer (placeholder)"
    log.warning(
        "AOI built from community point buffers. Replace with true river corridor "
        "geometry from HydroSHEDS or MERIT Hydro for final analysis."
    )
    return aoi[["buffer_km", "description", "geometry"]]


def make_aoi_map(
    aoi_gdf: gpd.GeoDataFrame,
    communities: gpd.GeoDataFrame,
    dams: gpd.GeoDataFrame,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    aoi_geo = aoi_gdf.to_crs(GEO_CRS)
    comm_geo = communities.to_crs(GEO_CRS)
    dam_geo = dams.to_crs(GEO_CRS)

    aoi_geo.plot(ax=ax, color="lightyellow", edgecolor="blue", linewidth=1.5, alpha=0.5, label="AOI (20 km)")
    comm_geo.plot(ax=ax, color="red", markersize=40, zorder=5, label="Affected Communities")
    dam_geo.plot(ax=ax, color="black", marker="^", markersize=80, zorder=6, label="Dam Locations")

    for _, row in comm_geo.iterrows():
        ax.annotate(row["community_name"], xy=(row.geometry.x, row.geometry.y),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    for _, row in dam_geo.iterrows():
        ax.annotate(row["name"], xy=(row.geometry.x, row.geometry.y),
                    xytext=(4, 4), textcoords="offset points", fontsize=7, fontweight="bold")

    ax.set_title("Lower Volta Basin – Akosombo Spillage Study Area\n(Placeholder AOI – replace with river corridor geometry)", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left", framealpha=0.8)
    ax.annotate("CRS: EPSG:4326 | Source: OpenStreetMap, public records | Date: 2024",
                xy=(0.01, 0.01), xycoords="axes fraction", fontsize=7, color="gray")

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("AOI map saved: %s", out_path)


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    aoi_dir = Path(get_processed_path("aoi", config))
    ensure_dir(aoi_dir)

    with Timer() as t:
        communities = build_community_gdf()
        communities_proj = to_projected(communities)
        dams = build_dam_gdf()

        aois = {}
        for km in [10, 20, 30]:
            aois[km] = build_river_aoi(communities_proj, buffer_km=km)

        default_aoi = aois[config["aoi"].get("default_buffer_km", 20)]

        # Save AOI
        aoi_gpkg = aoi_dir / "lower_volta_aoi.gpkg"
        aoi_geojson = aoi_dir / "lower_volta_aoi.geojson"
        default_aoi.to_crs(GEO_CRS).to_file(aoi_gpkg, driver="GPKG")
        default_aoi.to_crs(GEO_CRS).to_file(aoi_geojson, driver="GeoJSON")
        log.info("AOI saved: %s", aoi_gpkg)

        # Save communities
        comm_path = aoi_dir / "affected_communities.gpkg"
        communities.to_file(comm_path, driver="GPKG")

        # Save dam locations
        dam_path = aoi_dir / "dam_locations.gpkg"
        dams.to_file(dam_path, driver="GPKG")

        # Map
        map_path = get_output_path("maps/aoi_overview.png", config)
        make_aoi_map(default_aoi, communities, dams, map_path)

    log_run(
        script=__file__,
        input_paths=["config"],
        output_paths=[str(aoi_gpkg), str(comm_path), str(dam_path), str(map_path)],
        parameters={"default_buffer_km": 20},
        status="success",
        runtime_seconds=t.elapsed,
    )
    log.info("Phase 1 Task 1.1 complete in %.1f s", t.elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Lower Volta AOI")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
