from __future__ import annotations
"""
Phase 6 – Task 6.3: Settlement & Infrastructure Feature Engineering

Produces building density, road presence, and population density rasters
at 10 m resolution aligned to the S2 reference grid.

Features produced
-----------------
building_density_500m  — fraction of 500 m radius containing buildings
road_density_1km       — fraction of 1 km radius with road pixels
population_density     — WorldPop 2020 1km → 10m resampled (persons/km²)
dist_to_road           — Euclidean distance (m) to nearest road

Outputs
-------
data/processed/features/settlement/  (one GeoTIFF per feature)
"""

import argparse
import logging

import numpy as np
import rasterio
import rasterio.features
from rasterio.enums import Resampling
import geopandas as gpd
from scipy.ndimage import uniform_filter, distance_transform_edt

from src.utils.paths import load_config, get_processed_path, ensure_dir, get_project_root
from src.utils.logging_utils import Timer, log_run
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEMPLATE_RASTER   = "satellite/s2_ndwi_event.tif"
BUILDING_RADIUS_M = 500    # density kernel radius
ROAD_RADIUS_M     = 1000   # density kernel radius


def rasterize_vector(gdf: gpd.GeoDataFrame, template_path: str) -> np.ndarray:
    with rasterio.open(template_path) as src:
        out = np.zeros((src.height, src.width), dtype=np.uint8)
        if gdf.empty:
            return out
        geoms = [(g, 1) for g in gdf.geometry]
        rasterio.features.rasterize(
            geoms, out=out, transform=src.transform, fill=0, all_touched=True,
        )
    return out


def kernel_density(binary: np.ndarray, pixel_size_m: float,
                   radius_m: float) -> np.ndarray:
    """Mean fraction of pixels with feature in a circular window (approx by uniform filter)."""
    size_px = max(3, int(2 * radius_m / pixel_size_m) | 1)   # odd kernel size
    density = uniform_filter(binary.astype(np.float32), size=size_px, mode="constant")
    return density.astype(np.float32)


def write_feature(arr: np.ndarray, profile: dict, path, description: str) -> None:
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=-1, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr, 1)
        dst.update_tags(description=description)


def main(config_path: str | None = None) -> None:
    config   = load_config(config_path)
    root     = get_project_root()
    pop_dir  = get_processed_path("population", config)
    feat_dir = get_processed_path("features/settlement", config)
    ensure_dir(feat_dir)

    template      = get_processed_path(TEMPLATE_RASTER, config)
    bldg_path     = root / "data/raw/buildings/osm_buildings.gpkg"
    roads_path    = root / "data/raw/roads/osm_roads.gpkg"
    pop_path      = pop_dir / "population_grid_clipped.tif"

    with rasterio.open(template) as src:
        profile    = src.profile.copy()
        pixel_size = src.res[0]
        crs        = src.crs

    with Timer() as t:
        # ---- Building density ---------------------------------------------
        log.info("Computing building density (radius=%d m)…", BUILDING_RADIUS_M)
        if bldg_path.exists():
            bldg = gpd.read_file(str(bldg_path)).to_crs(crs)
            # Use centroids for fast rasterization
            bldg_pts = bldg.copy()
            bldg_pts["geometry"] = bldg.geometry.centroid
            bldg_bin = rasterize_vector(bldg_pts, str(template))
            log.info("  Building pixels: %d", int(bldg_bin.sum()))
        else:
            log.warning("  Buildings not found")
            with rasterio.open(template) as s:
                bldg_bin = np.zeros((s.height, s.width), dtype=np.uint8)

        bldg_density = kernel_density(bldg_bin, pixel_size, BUILDING_RADIUS_M)
        out_bldg = feat_dir / "building_density_500m.tif"
        write_feature(bldg_density, profile, out_bldg,
                      f"Building density: fraction of buildings in {BUILDING_RADIUS_M}m window")
        log.info("  building_density_500m: max=%.4f", float(bldg_density.max()))

        # ---- Road density -------------------------------------------------
        log.info("Computing road density (radius=%d m)…", ROAD_RADIUS_M)
        if roads_path.exists():
            roads = gpd.read_file(str(roads_path)).to_crs(crs)
            roads = roads[roads.geometry.geom_type.isin(
                ["LineString", "MultiLineString"])].copy()
            road_bin = rasterize_vector(roads, str(template))
            log.info("  Road pixels: %d", int(road_bin.sum()))
        else:
            log.warning("  Roads not found")
            with rasterio.open(template) as s:
                road_bin = np.zeros((s.height, s.width), dtype=np.uint8)

        road_density = kernel_density(road_bin, pixel_size, ROAD_RADIUS_M)
        out_road = feat_dir / "road_density_1km.tif"
        write_feature(road_density, profile, out_road,
                      f"Road density: fraction of road pixels in {ROAD_RADIUS_M}m window")
        log.info("  road_density_1km: max=%.4f", float(road_density.max()))

        # ---- Distance to road ---------------------------------------------
        log.info("Computing distance to nearest road…")
        dist_road = (distance_transform_edt(road_bin == 0) * pixel_size).astype(np.float32)
        out_dist_road = feat_dir / "dist_to_road.tif"
        write_feature(dist_road, profile, out_dist_road,
                      "Distance (m) to nearest road pixel (OSM)")
        log.info("  dist_to_road: range=[%.0f, %.0f] m",
                 float(dist_road.min()), float(dist_road.max()))

        # ---- Population density -------------------------------------------
        log.info("Aligning population density to 10 m grid…")
        if pop_path.exists():
            pop = read_aligned(str(pop_path), str(template), Resampling.bilinear)
            # WorldPop values are people/km²; nodata = -99999
            pop = np.where((pop > 0) & np.isfinite(pop), pop, 0.0).astype(np.float32)
        else:
            log.warning("  Population grid not found")
            with rasterio.open(template) as s:
                pop = np.zeros((s.height, s.width), dtype=np.float32)

        out_pop = feat_dir / "population_density.tif"
        write_feature(pop, profile, out_pop,
                      "Population density (persons/km²), WorldPop 2020 1km → 10m")
        log.info("  population_density: max=%.1f", float(pop.max()))

    log.info("Settlement features complete in %.1f s", t.elapsed)
    log_run(
        script=__file__,
        input_paths=[str(bldg_path), str(roads_path), str(pop_path)],
        output_paths=[str(out_bldg), str(out_road), str(out_pop)],
        parameters={"building_radius_m": BUILDING_RADIUS_M,
                    "road_radius_m": ROAD_RADIUS_M},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settlement feature engineering")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
