from __future__ import annotations
"""
Phase 2 – Task 2.1: Google Earth Engine Imagery Download

Authenticates with GEE, loads the AOI, pulls Sentinel-1 and Sentinel-2
image collections by the configured event windows, and exports composites
to Google Drive or local storage.

Usage
-----
    python src/data_ingestion/download_gee_imagery.py --config config/gee_config.yaml

Outputs
-------
data/processed/tables/satellite_scene_catalog.csv
exports queued to Google Drive folder defined in gee_config.yaml
"""

import argparse
import csv
import logging
from pathlib import Path

import yaml

from src.utils.logging_utils import Timer, log_run
from src.utils.paths import load_config, get_processed_path, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CATALOG_COLUMNS = [
    "scene_id", "sensor", "date", "orbit", "polarization",
    "cloud_cover", "date_window", "aoi_coverage_percent", "export_path", "notes",
]


def authenticate_gee(gee_config: dict) -> None:
    """Authenticate with Earth Engine using project ID or service account."""
    try:
        import ee
    except ImportError:
        raise ImportError("earthengine-api not installed. Run: pip install earthengine-api")

    if gee_config.get("service_account") and gee_config.get("credentials_path"):
        credentials = ee.ServiceAccountCredentials(
            gee_config["service_account"], gee_config["credentials_path"]
        )
        ee.Initialize(credentials, project=gee_config.get("project"))
    else:
        ee.Authenticate()
        ee.Initialize(project=gee_config.get("project"))
    log.info("GEE authenticated and initialized")


def get_s1_collection(ee, aoi_geom, date_start: str, date_end: str, gee_config: dict):
    """Return a filtered Sentinel-1 image collection for the given window."""
    cfg = gee_config["sentinel1"]
    collection = (
        ee.ImageCollection(cfg["collection"])
        .filterBounds(aoi_geom)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.eq("instrumentMode", cfg["instrument_mode"]))
        .select(cfg["polarizations"])
    )
    return collection


def get_s2_collection(ee, aoi_geom, date_start: str, date_end: str, gee_config: dict):
    """Return a cloud-filtered Sentinel-2 SR image collection."""
    cfg = gee_config["sentinel2"]
    collection = (
        ee.ImageCollection(cfg["collection"])
        .filterBounds(aoi_geom)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg["cloud_cover_max"]))
    )
    return collection


def export_image(ee, image, description: str, folder: str, scale: int, aoi_geom) -> None:
    """Queue a GEE export task to Google Drive."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        scale=scale,
        region=aoi_geom,
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
    )
    task.start()
    log.info("Export task queued: %s", description)


def main(config_path: str | None = None, gee_config_path: str | None = None) -> None:
    config = load_config(config_path)
    if gee_config_path is None:
        from src.utils.paths import get_project_root
        gee_config_path = get_project_root() / "config" / "gee_config.yaml"
    with open(gee_config_path) as f:
        gee_cfg = yaml.safe_load(f)

    # TODO: load AOI from data/processed/aoi/lower_volta_aoi.gpkg once built
    # and convert to ee.Geometry for filtering
    authenticate_gee(gee_cfg)

    try:
        import ee
    except ImportError:
        raise ImportError("earthengine-api not installed")

    # TODO: replace with actual AOI geometry loaded from GeoPackage
    log.warning("AOI geometry placeholder – replace with actual lower_volta_aoi.gpkg")
    # aoi_geom = geemap.geopandas_to_ee(gpd.read_file(...))

    windows = config["event_windows"]
    catalog_rows = []

    for window_name, window in windows.items():
        log.info("Processing window: %s (%s to %s)", window_name, window["start"], window["end"])
        # TODO: uncomment and complete when AOI geometry is available
        # s1 = get_s1_collection(ee, aoi_geom, window["start"], window["end"], gee_cfg)
        # s1_count = s1.size().getInfo()
        # log.info("S1 scenes in %s: %d", window_name, s1_count)
        catalog_rows.append({
            "scene_id": f"TODO_{window_name}_S1",
            "sensor": "Sentinel-1",
            "date": window["start"],
            "orbit": "TODO",
            "polarization": "VV,VH",
            "cloud_cover": "N/A",
            "date_window": window_name,
            "aoi_coverage_percent": "TODO",
            "export_path": f"TODO/{window_name}_s1_composite.tif",
            "notes": "Placeholder – run after AOI geometry is built",
        })

    catalog_path = get_processed_path("tables/satellite_scene_catalog.csv", config)
    ensure_dir(catalog_path.parent)
    with open(catalog_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        writer.writerows(catalog_rows)
    log.info("Scene catalog written: %s", catalog_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GEE imagery")
    parser.add_argument("--config", default=None)
    parser.add_argument("--gee-config", default=None)
    args = parser.parse_args()
    main(args.config, args.gee_config)
