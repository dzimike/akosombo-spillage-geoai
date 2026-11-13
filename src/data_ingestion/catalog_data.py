from __future__ import annotations
"""
Phase 2 – Task 2 (extra): Data Catalog Generator

Scans the data/ directory, detects raster/vector/table files, records CRS,
geometry type, raster resolution, date modified, and writes a CSV catalog.

Usage
-----
    python src/data_ingestion/catalog_data.py --config config/config.yaml

Outputs
-------
data/processed/tables/data_catalog.csv
"""

import argparse
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import rasterio
from pyproj import CRS

from src.utils.paths import load_config, get_project_root, get_processed_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CATALOG_COLUMNS = [
    "file_path", "file_type", "file_size_mb", "date_modified",
    "crs_epsg", "geometry_type", "feature_count",
    "raster_bands", "raster_resolution_m", "raster_nodata",
    "notes",
]

VECTOR_EXTS = {".gpkg", ".shp", ".geojson", ".json"}
RASTER_EXTS = {".tif", ".tiff", ".vrt"}
TABLE_EXTS = {".csv", ".parquet", ".xlsx"}


def catalog_vector(path: Path) -> dict:
    try:
        gdf = gpd.read_file(path)
        epsg = gdf.crs.to_epsg() if gdf.crs else None
        geom_types = gdf.geometry.geom_type.unique().tolist() if not gdf.empty else []
        return {"crs_epsg": epsg, "geometry_type": "|".join(str(g) for g in geom_types),
                "feature_count": len(gdf), "notes": ""}
    except Exception as e:
        return {"crs_epsg": None, "geometry_type": None, "feature_count": None, "notes": str(e)}


def catalog_raster(path: Path) -> dict:
    try:
        with rasterio.open(path) as src:
            epsg = CRS(src.crs).to_epsg() if src.crs else None
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            return {"crs_epsg": epsg, "geometry_type": "raster",
                    "feature_count": None, "raster_bands": src.count,
                    "raster_resolution_m": f"{res_x:.1f}x{res_y:.1f}",
                    "raster_nodata": src.nodata, "notes": ""}
    except Exception as e:
        return {"crs_epsg": None, "geometry_type": "raster", "feature_count": None,
                "raster_bands": None, "raster_resolution_m": None,
                "raster_nodata": None, "notes": str(e)}


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    root = get_project_root()
    data_root = root / "data"

    rows = []
    with Timer() as t:
        for fpath in sorted(data_root.rglob("*")):
            if not fpath.is_file():
                continue
            ext = fpath.suffix.lower()
            rel = str(fpath.relative_to(root))
            size_mb = fpath.stat().st_size / 1e6
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime).isoformat()

            base = {"file_path": rel, "file_size_mb": round(size_mb, 3),
                    "date_modified": mtime, "raster_bands": None,
                    "raster_resolution_m": None, "raster_nodata": None,
                    "feature_count": None, "crs_epsg": None, "geometry_type": None, "notes": ""}

            if ext in VECTOR_EXTS:
                base["file_type"] = "vector"
                base.update(catalog_vector(fpath))
            elif ext in RASTER_EXTS:
                base["file_type"] = "raster"
                base.update(catalog_raster(fpath))
            elif ext in TABLE_EXTS:
                base["file_type"] = "table"
            else:
                continue

            rows.append(base)

    catalog_path = get_processed_path("tables/data_catalog.csv", config)
    ensure_dir(catalog_path.parent)
    with open(catalog_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Data catalog written: %s (%d files)", catalog_path, len(rows))
    log_run(
        script=__file__,
        input_paths=["data/"],
        output_paths=[str(catalog_path)],
        parameters={},
        status="success",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Catalog data directory")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
