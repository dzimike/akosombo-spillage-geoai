"""
Expanded community exposure summary — all 24 Dam Spill Communities
===================================================================
Extends the six-community analysis in community_summary.py to all 24
communities in data/raw/communities/Dam Spill Communities.shp.

For each community a 10 km service buffer is used for population and
facility aggregation; flood extent and road/building exposure are
computed by direct spatial intersection within the buffer.

Outputs
-------
outputs/tables/community_exposure_summary_24.csv
outputs/tables/community_flood_validation_24.csv  — community × flood class
data/processed/exposure/community_exposure_24.gpkg

Usage
-----
    python -m src.exposure.community_summary_24
"""

from __future__ import annotations
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

from src.utils.paths import get_project_root, get_processed_path, get_output_path, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUFFER_M       = 10_000   # 10 km service zone
FLOOD_CLASSES  = {0: "Not flooded", 1: "SAR only", 2: "Optical only",
                  3: "SAR+optical", 4: "Perm water", 255: "Outside AOI"}
FLOOD_POSITIVE = {1, 2, 3}


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------

def raster_pixel_count_in_buffer(raster_path: Path, buffer_geom,
                                  target_values: set[int]) -> int:
    """Count pixels matching target_values inside buffer_geom."""
    if not raster_path.exists():
        return 0
    try:
        with rasterio.open(raster_path) as src:
            geom_reproj = _reproject_geom(buffer_geom, "EPSG:32630", src.crs.to_string())
            out, _ = rio_mask(src, [mapping(geom_reproj)], crop=True, nodata=255)
            arr = out[0]
            return int(np.isin(arr, list(target_values)).sum())
    except Exception as exc:
        log.warning("  raster_pixel_count failed: %s", exc)
        return 0


def raster_mean_in_buffer(raster_path: Path, buffer_geom,
                           nodata_val: float = -9999.0) -> float:
    """Mean of valid raster values inside buffer_geom."""
    if not raster_path.exists():
        return float("nan")
    try:
        with rasterio.open(raster_path) as src:
            geom_reproj = _reproject_geom(buffer_geom, "EPSG:32630", src.crs.to_string())
            out, _ = rio_mask(src, [mapping(geom_reproj)], crop=True, nodata=nodata_val)
            arr = out[0].astype(float)
            arr[arr == nodata_val] = np.nan
            valid = arr[np.isfinite(arr)]
            return float(valid.mean()) if len(valid) > 0 else float("nan")
    except Exception as exc:
        log.warning("  raster_mean failed: %s", exc)
        return float("nan")


def _reproject_geom(geom, src_crs: str, dst_crs: str):
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    if src_crs == dst_crs:
        return geom
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(transformer.transform, geom)


def sample_raster_at_point(raster_path: Path, x: float, y: float) -> float:
    """Sample a single raster value at projected (x, y)."""
    if not raster_path.exists():
        return float("nan")
    try:
        with rasterio.open(raster_path) as src:
            val = list(src.sample([(x, y)]))[0][0]
            return float(val)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def count_features_in_buffer(gdf: gpd.GeoDataFrame,
                               buffer: gpd.GeoDataFrame) -> int:
    if gdf is None or gdf.empty:
        return 0
    clipped = gpd.clip(gdf, buffer)
    return len(clipped)


def sum_length_in_buffer(lines: gpd.GeoDataFrame,
                          buffer: gpd.GeoDataFrame) -> float:
    if lines is None or lines.empty:
        return 0.0
    clipped = gpd.clip(lines, buffer)
    return float(clipped.geometry.length.sum() / 1000)   # km


def pop_in_buffer(pop_raster: Path, buffer_geom, pixel_area_m2: float = 100*100) -> float:
    """Sum WorldPop values (people per pixel) within buffer."""
    if not pop_raster.exists():
        return float("nan")
    try:
        with rasterio.open(pop_raster) as src:
            geom_reproj = _reproject_geom(buffer_geom, "EPSG:32630", src.crs.to_string())
            out, _ = rio_mask(src, [mapping(geom_reproj)], crop=True, nodata=-99999)
            arr = out[0].astype(float)
            arr[arr < 0] = 0
            return float(arr.sum())
    except Exception as exc:
        log.warning("  pop_in_buffer failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = get_project_root()
    ensure_dir(root / "outputs/tables")
    ensure_dir(root / "data/processed/exposure")

    # ---- Paths -------------------------------------------------------
    comm_path     = root / "data/processed/aoi/affected_communities.gpkg"
    flood_path    = root / "data/processed/flood_extent/fused_flood_confidence.tif"
    pop_path      = root / "data/processed/population/population_grid_clipped.tif"
    bldg_path     = root / "data/raw/buildings/osm_buildings.gpkg"
    roads_path    = root / "data/raw/roads/osm_roads.gpkg"

    # ---- Load communities --------------------------------------------
    log.info("Loading 24 communities…")
    comm = gpd.read_file(str(comm_path)).to_crs("EPSG:32630")
    comm["buffer_geom"] = comm.geometry.buffer(BUFFER_M)
    log.info("  %d communities loaded", len(comm))

    # ---- Load vector layers (optional) --------------------------------
    buildings = gpd.read_file(str(bldg_path)).to_crs("EPSG:32630") if bldg_path.exists() else None
    roads     = gpd.read_file(str(roads_path)).to_crs("EPSG:32630") if roads_path.exists() else None

    if buildings is None:
        log.warning("Buildings layer not found — building counts will be 0")
    if roads is None:
        log.warning("Roads layer not found — road lengths will be 0")

    # ---- Load flood extent as vector for intersection ----------------
    # fused_flood_extent.gpkg is produced by flood_extent_fusion.py already
    # restricted to flood-positive classes {1,2,3} (see raster_to_polygon()),
    # so no further class filtering is applied here.
    flood_vec_path = root / "data/processed/flood_extent/fused_flood_extent.gpkg"
    flood_vec = None
    if flood_vec_path.exists():
        flood_vec = gpd.read_file(str(flood_vec_path)).to_crs("EPSG:32630")
    else:
        log.warning("Flood extent vector not found at %s — exposed_buildings, "
                    "exposed_pop, and flooded_road_km will be 0/NaN", flood_vec_path)

    records = []

    for _, row in comm.iterrows():
        name    = row["community_name"]
        district = row["district"]
        region  = row["region"]
        pt_x, pt_y = row["geometry"].x, row["geometry"].y
        buf_geom = row["buffer_geom"]
        buf_gdf  = gpd.GeoDataFrame(geometry=[buf_geom], crs="EPSG:32630")

        log.info("Processing: %s", name)

        # Point-level flood class
        flood_cls_val = sample_raster_at_point(flood_path, pt_x, pt_y)
        flood_cls     = int(flood_cls_val) if np.isfinite(flood_cls_val) else 255
        flood_cls_name = FLOOD_CLASSES.get(flood_cls, "Unknown")
        flooded_point  = 1 if flood_cls in FLOOD_POSITIVE else 0

        # Flood pixel count within buffer
        flood_px = raster_pixel_count_in_buffer(flood_path, buf_geom, FLOOD_POSITIVE)
        flood_km2 = flood_px * (10 * 10) / 1e6   # 10 m pixels → km²

        # Population
        total_pop    = pop_in_buffer(pop_path, buf_geom)
        exposed_pop  = float("nan")
        if flood_vec is not None and not flood_vec.empty:
            flood_buf = gpd.clip(flood_vec, buf_gdf)
            if not flood_buf.empty:
                flood_pop = pop_in_buffer(pop_path, flood_buf.union_all())
                exposed_pop = flood_pop

        # Buildings
        total_bldg   = count_features_in_buffer(buildings, buf_gdf)
        exposed_bldg = 0
        if buildings is not None and flood_vec is not None and not flood_vec.empty:
            flood_buf = gpd.clip(flood_vec, buf_gdf)
            if not flood_buf.empty:
                flood_buf_gdf = gpd.GeoDataFrame(geometry=[flood_buf.union_all()], crs="EPSG:32630")
                exposed_bldg  = count_features_in_buffer(buildings, flood_buf_gdf)

        # Roads
        total_road_km   = sum_length_in_buffer(roads, buf_gdf)
        flooded_road_km = 0.0
        if roads is not None and flood_vec is not None and not flood_vec.empty:
            flood_buf = gpd.clip(flood_vec, buf_gdf)
            if not flood_buf.empty:
                flood_buf_gdf = gpd.GeoDataFrame(geometry=[flood_buf.union_all()], crs="EPSG:32630")
                flooded_road_km = sum_length_in_buffer(roads, flood_buf_gdf)

        # Buffer-based flood classification: >1 km² within 10 km buffer = flooded
        flooded_buffer = 1 if flood_km2 >= 1.0 else 0

        records.append({
            "community_name":   name,
            "district":         district,
            "region":           region,
            "flood_class_pt":   flood_cls,
            "flood_class_name": flood_cls_name,
            "flooded_point":    flooded_point,
            "flood_area_km2":   round(flood_km2, 2),
            "flooded_buffer":   flooded_buffer,
            "total_pop":        round(total_pop) if np.isfinite(total_pop) else None,
            "exposed_pop":      round(exposed_pop) if np.isfinite(exposed_pop) else None,
            "total_buildings":  total_bldg,
            "exposed_buildings":exposed_bldg,
            "total_road_km":    round(total_road_km, 1),
            "flooded_road_km":  round(flooded_road_km, 1),
        })

    df = pd.DataFrame(records)
    df = df.sort_values("flood_area_km2", ascending=False).reset_index(drop=True)

    # ---- Save CSV outputs -------------------------------------------
    csv_path = root / "outputs/tables/community_exposure_summary_24.csv"
    df.to_csv(str(csv_path), index=False)
    log.info("Saved: %s", csv_path)

    val_path = root / "outputs/validation/community_flood_validation.csv"
    df[["community_name","district","region","flood_class_pt",
        "flood_class_name","flooded_point","flood_area_km2"]].to_csv(str(val_path), index=False)
    log.info("Saved: %s", val_path)

    # ---- Save GeoPackage -------------------------------------------
    comm_out = comm[["community_name","district","region","geometry"]].copy()
    comm_out = comm_out.merge(df, on="community_name", how="left")
    gpkg_path = root / "data/processed/exposure/community_exposure_24.gpkg"
    comm_out.to_file(str(gpkg_path), driver="GPKG", layer="community_exposure")
    log.info("Saved: %s", gpkg_path)

    # ---- Summary print ----------------------------------------------
    log.info("")
    log.info("=== Summary ===")
    log.info("Total communities:  %d", len(df))
    log.info("Flooded (point):    %d", df["flooded_point"].sum())
    log.info("Total flood area:   %.1f km²", df["flood_area_km2"].sum())
    flooded_only = df[df["flooded_point"] == 1]
    if len(flooded_only):
        log.info("\nFlooded communities:")
        for _, r in flooded_only.iterrows():
            log.info("  %-20s  %.2f km²  class=%s",
                     r["community_name"], r["flood_area_km2"], r["flood_class_name"])


if __name__ == "__main__":
    main()
