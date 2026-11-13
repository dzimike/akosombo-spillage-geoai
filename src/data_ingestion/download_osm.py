from __future__ import annotations
"""
Phase 2 – Task 2.2: OSM Data Download

Downloads roads, waterways, buildings, schools, health facilities, markets,
bridges, ferries, and settlement points via OSMnx.

Usage
-----
    python src/data_ingestion/download_osm.py --config config/config.yaml

Outputs
-------
data/raw/roads/osm_roads.gpkg
data/raw/roads/osm_waterways.gpkg
data/raw/buildings/osm_buildings.gpkg
data/raw/health_education/osm_facilities.gpkg
"""

import argparse
import logging

import geopandas as gpd
from shapely.geometry import box

from src.utils.logging_utils import Timer, log_run
from src.utils.paths import load_config, get_data_path, ensure_dir, get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_aoi_polygon():
    """Load the study AOI polygon in EPSG:4326."""
    aoi_path = get_project_root() / "data/processed/aoi/lower_volta_aoi.gpkg"
    gdf = gpd.read_file(str(aoi_path))
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf.union_all()


def download_roads(polygon, output_path) -> gpd.GeoDataFrame:
    """Download road and waterway network within the AOI."""
    import osmnx as ox
    log.info("Downloading road network...")
    G = ox.graph_from_polygon(polygon, network_type="all", retain_all=True)
    _, gdf_edges = ox.graph_to_gdfs(G)
    keep_cols = [c for c in ["name", "highway", "bridge", "tunnel", "oneway", "length", "geometry"] if c in gdf_edges.columns]
    gdf_edges = gdf_edges[keep_cols].reset_index(drop=True)
    gdf_edges.to_file(str(output_path), driver="GPKG")
    log.info("Roads saved: %d features → %s", len(gdf_edges), output_path.name)
    return gdf_edges


def download_features(polygon, tags: dict, output_path, label: str) -> gpd.GeoDataFrame:
    """Download OSM features by tag dict within the AOI polygon."""
    import osmnx as ox
    log.info("Downloading %s (tags=%s)...", label, tags)
    try:
        gdf = ox.features_from_polygon(polygon, tags=tags)
        gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)
        # Keep only serialisable columns (drop list/dict columns that break GPKG)
        safe_cols = []
        for col in gdf.columns:
            if col == "geometry":
                safe_cols.append(col)
                continue
            try:
                sample = gdf[col].dropna()
                if len(sample) == 0 or not isinstance(sample.iloc[0], (list, dict)):
                    safe_cols.append(col)
            except Exception:
                pass
        gdf = gdf[safe_cols]
        gdf.to_file(str(output_path), driver="GPKG")
        log.info("%s saved: %d features → %s", label, len(gdf), output_path.name)
    except Exception as exc:
        log.warning("Could not download %s: %s", label, exc)
        gdf = gpd.GeoDataFrame()
    return gdf


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)

    roads_dir     = get_data_path("roads", config)
    buildings_dir = get_data_path("buildings", config)
    facilities_dir = get_data_path("health_education", config)
    for d in [roads_dir, buildings_dir, facilities_dir]:
        ensure_dir(d)

    log.info("Loading AOI polygon...")
    aoi_polygon = load_aoi_polygon()
    log.info("AOI bounds: %s", aoi_polygon.bounds)

    with Timer() as t:
        download_roads(aoi_polygon, roads_dir / "osm_roads.gpkg")

        download_features(
            aoi_polygon,
            tags={"waterway": ["river", "stream", "canal", "drain"]},
            output_path=roads_dir / "osm_waterways.gpkg",
            label="waterways",
        )
        download_features(
            aoi_polygon,
            tags={"amenity": ["school", "hospital", "clinic", "pharmacy", "marketplace"],
                  "healthcare": True},
            output_path=facilities_dir / "osm_facilities.gpkg",
            label="facilities",
        )
        download_features(
            aoi_polygon,
            tags={"building": True},
            output_path=buildings_dir / "osm_buildings.gpkg",
            label="buildings",
        )

    output_paths = [
        str(roads_dir / "osm_roads.gpkg"),
        str(roads_dir / "osm_waterways.gpkg"),
        str(facilities_dir / "osm_facilities.gpkg"),
        str(buildings_dir / "osm_buildings.gpkg"),
    ]
    log_run(
        script=__file__,
        input_paths=["data/processed/aoi/lower_volta_aoi.gpkg"],
        output_paths=output_paths,
        parameters={"source": "OpenStreetMap via OSMnx"},
        status="success",
        runtime_seconds=t.elapsed,
    )
    log.info("OSM download complete in %.1f s", t.elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OSM data for the Lower Volta study area")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
