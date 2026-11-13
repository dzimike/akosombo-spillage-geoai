from __future__ import annotations
"""
Per-community direct threshold-exposure propagation.

Complements threshold_exposure_direct.py: for each of the 24 designated
communities, counts buildings, population, and road length inside the
10 km service buffer that fall on pixels with flood probability >= each of
the 19 thresholds (permanent water excluded), so that the stability of the
between-community exposure ranking across thresholds is measured rather than
assumed.

Outputs
-------
outputs/tables/threshold_exposure_direct_by_community.csv   (long format)
outputs/tables/threshold_exposure_rank_stability.csv         (summary)
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr

from src.utils.paths import get_project_root, get_output_path
from src.validation.threshold_exposure_direct import (
    THRESHOLDS, ROAD_SAMPLE_SPACING_M, _mask_perm_water,
    interpolate_road_points, reproject_population_conserving, sample_at_points,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUFFER_M = 10_000
POP_SCALE = 366077 / 426499   # clipped WorldPop grid -> paper's AOI total


def main() -> None:
    root = get_project_root()
    prob_path = root / "outputs/maps/flood_susceptibility_xgb.tif"
    with rasterio.open(prob_path) as src:
        prob = src.read(1).astype(np.float32)
        tr, crs = src.transform, src.crs
    with rasterio.open(root / "data/processed/flood_extent/fused_flood_confidence.tif") as src:
        perm = src.read(1) == 4
    prob = np.where(perm, np.float32(0.0), prob)

    comm = gpd.read_file(root / "data/processed/exposure/community_exposure_24.gpkg").to_crs(crs)
    comm = comm[["community_name", "geometry"]].reset_index(drop=True)
    cx, cy = comm.geometry.x.values, comm.geometry.y.values

    bldg = gpd.read_file(root / "data/raw/buildings/osm_buildings.gpkg").to_crs(crs)
    cen = bldg.geometry.centroid
    bx, by = cen.x.values, cen.y.values
    bprob = _mask_perm_water(sample_at_points(prob_path, bx, by), bx, by, perm, tr)

    roads = gpd.read_file(root / "data/raw/roads/osm_roads.gpkg").to_crs(crs)
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    rx, ry, _, rlen = interpolate_road_points(roads, ROAD_SAMPLE_SPACING_M)
    rprob = _mask_perm_water(sample_at_points(prob_path, rx, ry), rx, ry, perm, tr)

    pop = reproject_population_conserving(
        root / "data/processed/population/population_grid_clipped.tif", prob_path)

    rows = []
    for i, name in enumerate(comm.community_name):
        in_b = np.hypot(bx - cx[i], by - cy[i]) <= BUFFER_M
        in_r = np.hypot(rx - cx[i], ry - cy[i]) <= BUFFER_M
        # raster window covering the buffer
        c0, r0 = ~tr * (cx[i] - BUFFER_M, cy[i] + BUFFER_M)
        c1, r1 = ~tr * (cx[i] + BUFFER_M, cy[i] - BUFFER_M)
        r0, r1 = max(int(r0), 0), min(int(r1) + 1, prob.shape[0])
        c0, c1 = max(int(c0), 0), min(int(c1) + 1, prob.shape[1])
        rr, cc = np.mgrid[r0:r1, c0:c1]
        px_x, px_y = tr * (cc + 0.5, rr + 0.5)
        in_px = np.hypot(px_x - cx[i], px_y - cy[i]) <= BUFFER_M
        pw, pp = prob[r0:r1, c0:c1], pop[r0:r1, c0:c1]
        for thr in THRESHOLDS:
            m = in_px & np.isfinite(pw) & (pw >= thr)
            rows.append({"community": name, "threshold": thr,
                         "buildings_exposed": int((bprob[in_b] >= thr).sum()),
                         "population_exposed": float(pp[m].sum() * POP_SCALE),
                         "roads_exposed_km": float(rlen[in_r][rprob[in_r] >= thr].sum() / 1000),
                         "flood_area_km2": float(m.sum() * abs(tr.a * tr.e) / 1e6)})
        log.info("  %s done", name)

    df = pd.DataFrame(rows)
    df.to_csv(get_output_path("tables/threshold_exposure_direct_by_community.csv"), index=False)

    summ = []
    for metric in ("buildings_exposed", "population_exposed", "roads_exposed_km", "flood_area_km2"):
        wide = df.pivot(index="community", columns="threshold", values=metric)
        ref_rank = wide[0.50].rank(ascending=False, method="average")
        top5_ref = set(wide[0.50].nlargest(5).index)
        for thr in THRESHOLDS:
            rho = spearmanr(wide[0.50], wide[thr]).statistic
            top5 = set(wide[thr].nlargest(5).index)
            summ.append({"metric": metric, "threshold": thr, "spearman_vs_0.50": rho,
                         "top5_overlap_with_0.50": len(top5 & top5_ref),
                         "max_abs_rank_shift": float((wide[thr].rank(ascending=False) - ref_rank).abs().max())})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(get_output_path("tables/threshold_exposure_rank_stability.csv"), index=False)
    hp = sdf[(sdf.threshold >= 0.50)]
    log.info("Rank stability, thresholds 0.50-0.95:\n%s",
             hp.groupby("metric").agg(min_rho=("spearman_vs_0.50", "min"),
                                      min_top5=("top5_overlap_with_0.50", "min"),
                                      max_rank_shift=("max_abs_rank_shift", "max")).round(3))
    full = sdf.groupby("metric").agg(min_rho=("spearman_vs_0.50", "min"),
                                     min_top5=("top5_overlap_with_0.50", "min"),
                                     max_rank_shift=("max_abs_rank_shift", "max")).round(3)
    log.info("Rank stability, full 0.05-0.95:\n%s", full)


if __name__ == "__main__":
    main()
