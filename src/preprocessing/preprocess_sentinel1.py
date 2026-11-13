from __future__ import annotations
"""
Phase 3 – Task 3.1: Sentinel-1 SAR Preprocessing via Google Earth Engine

Queues GEE export tasks to Google Drive for:
  - Pre-event VV/VH median composites
  - Event-period VV/VH median composites
  - VV/VH dB change (event − pre)
  - VV linear ratio

Processing chain
----------------
1. Filter S1-GRD to AOI, date window, IW mode, VV+VH.
2. Convert to dB (10 * log10).
3. Apply 7×7 focal-mean speckle filter.
4. Compute median composite per window.
5. Compute change (dB difference) and ratio images.
6. Export GeoTIFFs to Google Drive folder.

After export completes
----------------------
Copy TIFs from Google Drive / akosombo_geoai_exports/ to:
  data/processed/satellite/

Usage
-----
    python src/preprocessing/preprocess_sentinel1.py
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
    import ee, yaml
    _gee    = gee_cfg.get("gee", gee_cfg)   # handle both nested and flat config
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


def to_db(image):
    """Convert S1 linear power to dB."""
    return image.log10().multiply(10)


def speckle_filter(image):
    """7×7 focal mean speckle filter."""
    return image.focal_mean(radius=3.5, kernelType="square", units="pixels")


def get_s1_composite(ee, aoi_geom, date_start: str, date_end: str, gee_cfg: dict):
    """Return dB median composite for one date window."""
    cfg = gee_cfg.get("sentinel1", {})
    col = (
        ee.ImageCollection(cfg.get("collection", "COPERNICUS/S1_GRD"))
        .filterBounds(aoi_geom)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )
    count = col.size().getInfo()
    log.info("  %s – %s: %d S1 scenes", date_start, date_end, count)
    composite = col.map(lambda img: speckle_filter(to_db(img))).median()
    return composite, count


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
        log.info("Building S1 composites…")
        pre_comp,   n_pre   = get_s1_composite(ee, aoi_geom, windows["pre_event"]["start"],        windows["pre_event"]["end"],        gee_cfg)
        event_comp, n_event = get_s1_composite(ee, aoi_geom, windows["event"]["start"],            windows["event"]["end"],            gee_cfg)
        post_comp,  n_post  = get_s1_composite(ee, aoi_geom, windows["post_event_early"]["start"], windows["post_event_early"]["end"], gee_cfg)

        vv_change = event_comp.select("VV").subtract(pre_comp.select("VV")).rename("VV_change")
        vh_change = event_comp.select("VH").subtract(pre_comp.select("VH")).rename("VH_change")
        vv_ratio  = event_comp.select("VV").divide(pre_comp.select("VV")).rename("VV_ratio")

        exports = [
            ("s1_pre_event_vv",  pre_comp.select("VV")),
            ("s1_pre_event_vh",  pre_comp.select("VH")),
            ("s1_event_vv",      event_comp.select("VV")),
            ("s1_event_vh",      event_comp.select("VH")),
            ("s1_post_event_vv", post_comp.select("VV")),
            ("s1_vv_change",     vv_change),
            ("s1_vh_change",     vh_change),
            ("s1_vv_ratio",      vv_ratio),
        ]

        log.info("Queueing %d GEE export tasks…", len(exports))
        for name, img in exports:
            queue_export(ee, img, name, aoi_geom, folder, scale)

        # Scene catalog stub
        catalog_path = tables_dir / "satellite_scene_catalog.csv"
        rows = [{"scene_id": name, "sensor": "Sentinel-1 GRD",
                 "date_window": name.split("_")[1] if "_" in name else "",
                 "polarization": "VV+VH", "notes": "GEE export queued"}
                for name, _ in exports]
        df = pd.DataFrame(rows)
        if catalog_path.exists():
            existing = pd.read_csv(catalog_path)
            df = pd.concat([existing, df], ignore_index=True).drop_duplicates("scene_id")
        df.to_csv(str(catalog_path), index=False)

    log.info(
        "\nAll S1 exports queued in %.1f s."
        "\nMonitor progress: https://code.earthengine.google.com/tasks"
        "\nOnce complete, copy TIFs from Google Drive/%s/ to data/processed/satellite/",
        t.elapsed, folder,
    )
    log_run(
        script=__file__,
        input_paths=["data/processed/aoi/lower_volta_aoi.gpkg"],
        output_paths=[f"data/processed/satellite/{n}.tif" for n, _ in exports],
        parameters={"source": "GEE S1-GRD", "scale_m": scale, "n_pre": n_pre, "n_event": n_event},
        status="exports_queued",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Sentinel-1 via GEE")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
