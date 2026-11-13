from __future__ import annotations
"""
Phase 3 – Task 3.2: Sentinel-2 MSI Preprocessing via Google Earth Engine

Queues GEE export tasks to Google Drive for cloud-masked spectral index
composites (NDVI, NDWI, MNDWI, NDBI) and RGB images.

Indices
-------
NDVI  = (B8 − B4)  / (B8 + B4)
NDWI  = (B3 − B8)  / (B3 + B8)
MNDWI = (B3 − B11) / (B3 + B11)
NDBI  = (B11 − B8) / (B11 + B8)

After export completes
----------------------
Copy TIFs from Google Drive / akosombo_geoai_exports/ to:
  data/processed/satellite/

Usage
-----
    python src/preprocessing/preprocess_sentinel2.py
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.utils.paths import load_config, get_processed_path, get_project_root, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET_CRS = "EPSG:32630"
RESOLUTION = 10


def init_gee(gee_cfg: dict):
    import ee
    _gee    = gee_cfg.get("gee", gee_cfg)
    project = _gee.get("project") or None
    creds   = _gee.get("credentials_path") or None
    svc     = _gee.get("service_account") or None
    if creds and svc:
        ee.Initialize(ee.ServiceAccountCredentials(svc, creds), project=project)
    else:
        ee.Initialize(project=project)
    log.info("GEE initialised (project=%s)", project)
    return ee


def aoi_to_ee_geometry(ee, aoi_path: Path):
    gdf = gpd.read_file(str(aoi_path)).to_crs("EPSG:4326")
    return ee.Geometry(gdf.geometry.union_all().__geo_interface__)


def mask_s2_clouds(image):
    """SCL-based cloud/shadow mask for Sentinel-2 SR."""
    scl = image.select("SCL")
    mask = (scl.neq(3)    # cloud shadow
              .And(scl.neq(8))    # medium cloud
              .And(scl.neq(9))    # high cloud
              .And(scl.neq(10)))  # thin cirrus
    return image.updateMask(mask)


def add_indices(image):
    """Add NDVI, NDWI, MNDWI, NDBI to image."""
    ndvi  = image.normalizedDifference(["B8",  "B4"]).rename("NDVI")
    ndwi  = image.normalizedDifference(["B3",  "B8"]).rename("NDWI")
    mndwi = image.normalizedDifference(["B3",  "B11"]).rename("MNDWI")
    ndbi  = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return image.addBands([ndvi, ndwi, mndwi, ndbi])


def get_s2_composite(ee, aoi_geom, date_start: str, date_end: str, gee_cfg: dict):
    """Return cloud-masked, index-enriched median S2 composite."""
    max_cloud = gee_cfg.get("sentinel2", {}).get("cloud_cover_max", 20)
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_geom)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(mask_s2_clouds)
        .map(add_indices)
    )
    count = col.size().getInfo()
    log.info("  %s – %s: %d S2 scenes (cloud < %d%%)", date_start, date_end, count, max_cloud)
    if count == 0:
        log.warning("  No scenes — relaxing cloud threshold to 80%%")
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi_geom)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
            .map(mask_s2_clouds)
            .map(add_indices)
        )
        count = col.size().getInfo()
        log.info("  Relaxed threshold: %d scenes", count)
    return col.median(), count


def queue_export(ee, image, name: str, aoi_geom, folder: str, scale: int):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=folder,
        fileNamePrefix=name,
        scale=scale,
        region=aoi_geom,
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
        crs=TARGET_CRS,
    )
    task.start()
    log.info("  Queued: %s → Drive/%s/", name, folder)
    return task


def main(config_path: str | None = None) -> None:
    import yaml
    config = load_config(config_path)
    root   = get_project_root()

    sat_dir    = get_processed_path("satellite", config)
    tables_dir = get_processed_path("tables", config)
    ensure_dir(sat_dir)
    ensure_dir(tables_dir)

    with open(root / "config" / "gee_config.yaml") as f:
        gee_cfg = yaml.safe_load(f)

    try:
        ee = init_gee(gee_cfg)
    except Exception as exc:
        log.error("GEE init failed: %s", exc)
        return

    aoi_geom = aoi_to_ee_geometry(ee, root / "data/processed/aoi/lower_volta_aoi.gpkg")
    windows  = config["event_windows"]
    scale    = gee_cfg["export"]["scale_meters"]
    folder   = gee_cfg["export"]["drive_folder"]

    with Timer() as t:
        log.info("Building S2 composites…")
        pre_comp,   n_pre   = get_s2_composite(ee, aoi_geom, windows["pre_event"]["start"],        windows["pre_event"]["end"],        gee_cfg)
        event_comp, n_event = get_s2_composite(ee, aoi_geom, windows["event"]["start"],            windows["event"]["end"],            gee_cfg)
        post_comp,  n_post  = get_s2_composite(ee, aoi_geom, windows["post_event_early"]["start"], windows["post_event_early"]["end"], gee_cfg)

        exports = [
            # Indices
            ("s2_ndvi_pre",    pre_comp.select("NDVI")),
            ("s2_ndwi_pre",    pre_comp.select("NDWI")),
            ("s2_mndwi_pre",   pre_comp.select("MNDWI")),
            ("s2_ndvi_event",  event_comp.select("NDVI")),
            ("s2_ndwi_event",  event_comp.select("NDWI")),
            ("s2_mndwi_event", event_comp.select("MNDWI")),
            ("s2_ndbi_event",  event_comp.select("NDBI")),
            ("s2_ndvi_post",   post_comp.select("NDVI")),
            ("s2_mndwi_post",  post_comp.select("MNDWI")),
            # True colour RGB
            ("s2_rgb_pre",     pre_comp.select(["B4", "B3", "B2"])),
            ("s2_rgb_event",   event_comp.select(["B4", "B3", "B2"])),
            ("s2_rgb_post",    post_comp.select(["B4", "B3", "B2"])),
        ]

        log.info("Queueing %d GEE export tasks…", len(exports))
        for name, img in exports:
            queue_export(ee, img, name, aoi_geom, folder, scale)

        # Scene catalog stub
        catalog_path = tables_dir / "satellite_scene_catalog.csv"
        rows = [{"scene_id": name, "sensor": "Sentinel-2 L2A",
                 "date_window": "pre" if "pre" in name else ("event" if "event" in name else "post"),
                 "polarization": None, "notes": "GEE export queued"}
                for name, _ in exports]
        df = pd.DataFrame(rows)
        if catalog_path.exists():
            existing = pd.read_csv(catalog_path)
            df = pd.concat([existing, df], ignore_index=True).drop_duplicates("scene_id")
        df.to_csv(str(catalog_path), index=False)

    log.info(
        "\nAll S2 exports queued in %.1f s."
        "\nMonitor progress: https://code.earthengine.google.com/tasks"
        "\nOnce complete, copy TIFs from Google Drive/%s/ to data/processed/satellite/",
        t.elapsed, folder,
    )
    log_run(
        script=__file__,
        input_paths=["data/processed/aoi/lower_volta_aoi.gpkg"],
        output_paths=[f"data/processed/satellite/{n}.tif" for n, _ in exports],
        parameters={"source": "GEE S2-SR", "scale_m": scale, "n_pre": n_pre, "n_event": n_event},
        status="exports_queued",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Sentinel-2 via GEE")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
