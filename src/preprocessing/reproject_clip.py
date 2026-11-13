"""
Reproject and clip raster/vector layers to the AOI and target CRS.

Utility script for standardizing all intermediate layers before stacking.
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd

from src.utils.crs import PROJ_CRS
from src.utils.raster import reproject_raster, align_raster
from src.utils.paths import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def reproject_vector(input_path: str, output_path: str, crs: str = PROJ_CRS) -> None:
    gdf = gpd.read_file(input_path).to_crs(crs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG")
    log.info("Reprojected vector: %s -> %s", input_path, output_path)


def clip_vector_to_aoi(input_path: str, aoi_path: str, output_path: str) -> None:
    aoi = gpd.read_file(aoi_path)
    gdf = gpd.read_file(input_path)
    if gdf.crs != aoi.crs:
        gdf = gdf.to_crs(aoi.crs)
    clipped = gpd.clip(gdf, aoi)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(output_path, driver="GPKG")
    log.info("Clipped %d features -> %s", len(clipped), output_path)
