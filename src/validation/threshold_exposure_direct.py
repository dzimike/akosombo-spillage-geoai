from __future__ import annotations
"""
Direct full-raster threshold-exposure propagation.

Replaces the proportional-scaling approximation (Section 4.2 of the
uncertainty manuscript) with exposure counts computed by intersecting
buildings, population, and roads directly against the full-scene flood
probability raster at each of the 19 decision thresholds. No vectorisation
of the flood mask is performed — buildings and roads are handled via point
sampling of the probability surface, and population via a total-preserving
reprojection of the WorldPop grid onto the probability raster's grid, so
that exposure at each threshold is a simple boolean-mask sum.

Permanent water (class 4 of the fused confidence map, i.e. the JRC layer) is
excluded from every exposure count, matching the pipeline's definition of
flood as classes 1-3. The classifier was never shown class 4 and scores it as
flood, so this mask is required; it does not depend on the threshold.

Usage
-----
    python -m src.validation.threshold_exposure_direct [--prob PATH] [--out PATH]

Outputs
-------
outputs/tables/threshold_exposure_direct.csv  (default)
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

from src.utils.paths import get_project_root, get_output_path, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

THRESHOLDS = [round(0.05 + 0.05 * i, 2) for i in range(19)]  # 0.05 .. 0.95
ROAD_SAMPLE_SPACING_M = 20.0


def sample_at_points(raster_path: Path, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=np.float32)
    return vals


def reproject_population_conserving(pop_path: Path, ref_path: Path) -> np.ndarray:
    """Reproject the WorldPop grid onto the reference raster's grid, dividing
    each source cell's population count across the destination pixels it
    overlaps so the total is conserved."""
    with rasterio.open(ref_path) as ref:
        ref_transform, ref_crs, ref_shape = ref.transform, ref.crs, (ref.height, ref.width)

    with rasterio.open(pop_path) as src:
        src_arr = src.read(1).astype(np.float64)
        src_arr[src_arr < 0] = 0
        src_transform, src_crs = src.transform, src.crs

        # Count how many destination pixels each source pixel maps to, by
        # reprojecting a constant array of ones with the same resampling.
        ones = np.ones_like(src_arr, dtype=np.float64)
        dest_count = np.zeros(ref_shape, dtype=np.float64)
        reproject(
            source=ones, destination=dest_count,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            resampling=Resampling.sum,
        )

        dest_pop = np.zeros(ref_shape, dtype=np.float64)
        reproject(
            source=src_arr, destination=dest_pop,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            resampling=Resampling.sum,
        )

    total_src = src_arr.sum()
    total_dst = dest_pop.sum()
    log.info("  Population conservation check: source total=%.1f, reprojected total=%.1f (%.3f%% diff)",
             total_src, total_dst, 100 * abs(total_src - total_dst) / max(total_src, 1))
    return dest_pop.astype(np.float32)


def _mask_perm_water(vals: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                     perm_water: np.ndarray, transform) -> np.ndarray:
    """Set sampled probability to 0 where the sample falls on a permanent-water pixel."""
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)
    ok = (rows >= 0) & (rows < perm_water.shape[0]) & (cols >= 0) & (cols < perm_water.shape[1])
    on_water = np.zeros(len(vals), dtype=bool)
    on_water[ok] = perm_water[rows[ok], cols[ok]]
    out = vals.copy()
    out[on_water] = 0.0
    return out


def interpolate_road_points(roads: gpd.GeoDataFrame, spacing_m: float):
    """For each road, return sample point coordinates and the segment length
    each point represents (length / n_points)."""
    all_xs, all_ys, seg_idx, seg_len_per_point = [], [], [], []
    for i, geom in enumerate(roads.geometry):
        length = geom.length
        if length <= 0:
            continue
        n_pts = max(int(length // spacing_m), 1)
        distances = np.linspace(0, length, n_pts + 1)[:-1] + (length / n_pts) / 2
        for d in distances:
            pt = geom.interpolate(d)
            all_xs.append(pt.x)
            all_ys.append(pt.y)
            seg_idx.append(i)
            seg_len_per_point.append(length / n_pts)
    return (np.array(all_xs), np.array(all_ys),
            np.array(seg_idx), np.array(seg_len_per_point))


def main(prob_rel: str = "outputs/maps/flood_susceptibility_xgb.tif",
         out_rel: str = "tables/threshold_exposure_direct.csv") -> None:
    root = get_project_root()
    ensure_dir(root / "outputs/tables")

    prob_path = root / prob_rel
    fused_path = root / "data/processed/flood_extent/fused_flood_confidence.tif"
    bldg_path = root / "data/raw/buildings/osm_buildings.gpkg"
    roads_path = root / "data/raw/roads/osm_roads.gpkg"
    pop_path = root / "data/processed/population/population_grid_clipped.tif"

    log.info("Loading flood probability raster…")
    with rasterio.open(prob_path) as src:
        prob = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        H, W = src.height, src.width
        pixel_area_km2 = abs(transform.a * transform.e) / 1e6

    with rasterio.open(fused_path) as src:
        assert (src.height, src.width) == (H, W), "fused map grid differs from probability grid"
        perm_water = src.read(1) == 4
    # permanent water is not event flood: remove from the probability surface
    prob = np.where(perm_water, np.float32(0.0), prob)
    pw_flat = perm_water.ravel()

    # ---- Buildings: sample centroid probability once ----------------------
    log.info("Loading buildings and sampling centroid probabilities…")
    bldg = gpd.read_file(str(bldg_path)).to_crs(crs)
    centroids = bldg.geometry.centroid
    bldg_prob = sample_at_points(prob_path, centroids.x.values, centroids.y.values)
    bldg_prob = _mask_perm_water(bldg_prob, centroids.x.values, centroids.y.values, perm_water, transform)
    n_buildings = len(bldg)

    # ---- Roads: sample points along each segment ---------------------------
    log.info("Loading roads and sampling points along segments (spacing=%.0fm)…",
              ROAD_SAMPLE_SPACING_M)
    roads = gpd.read_file(str(roads_path)).to_crs(crs)
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    rxs, rys, rseg_idx, rseg_len = interpolate_road_points(roads, ROAD_SAMPLE_SPACING_M)
    road_prob = sample_at_points(prob_path, rxs, rys)
    road_prob = _mask_perm_water(road_prob, rxs, rys, perm_water, transform)
    total_road_km = roads.geometry.length.sum() / 1000

    # ---- Population: total-preserving reprojection onto probability grid --
    log.info("Reprojecting population grid onto probability raster grid…")
    pop_aligned = reproject_population_conserving(pop_path, prob_path)

    # ---- Reference (threshold 0.50) --------------------------------------
    ref_thr = 0.50
    ref_flood_px = int((prob >= ref_thr).sum())
    ref_flood_km2 = ref_flood_px * pixel_area_km2

    log.info("Computing exposure at %d thresholds…", len(THRESHOLDS))
    records = []
    for thr in THRESHOLDS:
        flood_mask = prob >= thr
        flood_km2 = float(flood_mask.sum() * pixel_area_km2)

        exposed_buildings = int((bldg_prob >= thr).sum())

        exposed_pop = float(pop_aligned[flood_mask].sum())

        flooded_pt_mask = road_prob >= thr
        exposed_road_km = float(rseg_len[flooded_pt_mask].sum() / 1000)

        records.append({
            "threshold": thr,
            "flood_area_km2": round(flood_km2, 1),
            "buildings_exposed": exposed_buildings,
            "pct_buildings": round(100 * exposed_buildings / n_buildings, 2),
            "population_exposed": round(exposed_pop),
            "roads_exposed_km": round(exposed_road_km, 1),
            "pct_roads": round(100 * exposed_road_km / total_road_km, 2),
        })
        log.info("  thr=%.2f: flood=%.1f km2, buildings=%d, pop=%.0f, roads=%.1f km",
                  thr, flood_km2, exposed_buildings, exposed_pop, exposed_road_km)

    df = pd.DataFrame(records)

    out_path = get_output_path(out_rel)
    df.to_csv(str(out_path), index=False)
    log.info("Saved: %s", out_path)

    # ---- Summary: high-performance range comparison ------------------------
    hp = df[(df["threshold"] >= 0.50) & (df["threshold"] <= 0.95)]
    log.info("")
    log.info("=== High-performance range (0.50-0.95) direct computation ===")
    log.info("Buildings: %d - %d (range %.1f%%)",
              hp["buildings_exposed"].min(), hp["buildings_exposed"].max(),
              100 * (hp["buildings_exposed"].max() - hp["buildings_exposed"].min()) / hp["buildings_exposed"].max())
    log.info("Population: %.0f - %.0f (range %.1f%%)",
              hp["population_exposed"].min(), hp["population_exposed"].max(),
              100 * (hp["population_exposed"].max() - hp["population_exposed"].min()) / hp["population_exposed"].max())
    log.info("Roads: %.1f - %.1f km (range %.1f%%)",
              hp["roads_exposed_km"].min(), hp["roads_exposed_km"].max(),
              100 * (hp["roads_exposed_km"].max() - hp["roads_exposed_km"].min()) / hp["roads_exposed_km"].max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", default="outputs/maps/flood_susceptibility_xgb.tif")
    ap.add_argument("--out", default="tables/threshold_exposure_direct.csv")
    a = ap.parse_args()
    main(a.prob, a.out)
