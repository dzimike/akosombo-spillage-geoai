from __future__ import annotations
"""
Threshold sweep of exposure by administrative district (non-overlapping).

Replaces the community-buffer version (threshold_exposure_by_community.py), whose 10 km
buffers overlap. For each of the 19 probability thresholds, flood area, buildings, people
(rescaled to the 366,077 total) and road length are tallied per GADM level-2 district on the
full-scene XGBoost probability surface (permanent water excluded). The ordering of districts
at each threshold is compared with the ordering at 0.50 (Spearman correlation, top-3 overlap).

Outputs: outputs/tables/threshold_exposure_by_district.csv, threshold_exposure_district_rank_stability.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.stats import spearmanr
from shapely.geometry import box

from src.utils.paths import get_project_root, get_output_path
from src.validation.threshold_exposure_direct import (
    THRESHOLDS, ROAD_SAMPLE_SPACING_M, interpolate_road_points, reproject_population_conserving,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
POP_SCALE = 366077 / 426499


def main(corrected: bool = False) -> None:
    root = get_project_root()
    sfx = "_prior_corrected" if corrected else ""
    prob_p = root / f"outputs/maps/flood_susceptibility_xgb{sfx}.tif"
    with rasterio.open(prob_p) as s:
        prob, tr, crs, shape = s.read(1), s.transform, s.crs, s.shape
    with rasterio.open(root / "data/processed/flood_extent/fused_flood_confidence.tif") as s:
        perm = s.read(1) == 4
    prob = np.where(perm, np.nan, prob)
    g = gpd.read_file(root / "data/raw/admin/gadm41_GHA.gpkg", layer="ADM_ADM_2").to_crs(crs)
    g = g[g.intersects(box(tr.c, tr.f + shape[0] * tr.e, tr.c + shape[1] * tr.a, tr.f))].reset_index(drop=True)
    g["gid"] = np.arange(1, len(g) + 1)
    ids = rasterize(((geom, int(i)) for geom, i in zip(g.geometry, g.gid)), out_shape=shape, transform=tr, fill=0, dtype="int32")
    pop = reproject_population_conserving(root / "data/processed/population/population_grid_clipped.tif", prob_p) * POP_SCALE

    bld = gpd.read_file(root / "data/raw/buildings/osm_buildings.gpkg").to_crs(crs)
    c = bld.geometry.centroid
    br, bc = rasterio.transform.rowcol(tr, c.x.values, c.y.values); br, bc = np.array(br), np.array(bc)
    okb = (br >= 0) & (br < shape[0]) & (bc >= 0) & (bc < shape[1])
    b_id = np.zeros(len(bld), int); b_id[okb] = ids[br[okb], bc[okb]]
    b_p = np.full(len(bld), np.nan, np.float32); b_p[okb] = prob[br[okb], bc[okb]]
    roads = gpd.read_file(root / "data/raw/roads/osm_roads.gpkg").to_crs(crs)
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].explode(index_parts=False).reset_index(drop=True)
    rx, ry, _, rlen = interpolate_road_points(roads, ROAD_SAMPLE_SPACING_M)
    rr, rc = rasterio.transform.rowcol(tr, rx, ry); rr, rc = np.array(rr), np.array(rc)
    okr = (rr >= 0) & (rr < shape[0]) & (rc >= 0) & (rc < shape[1])
    r_id = np.zeros(len(rx), int); r_id[okr] = ids[rr[okr], rc[okr]]
    r_p = np.full(len(rx), np.nan, np.float32); r_p[okr] = prob[rr[okr], rc[okr]]

    rows = []
    for _, d in g.iterrows():
        m = ids == d.gid
        pv, popv = prob[m], pop[m]
        bmask = b_id == d.gid; rmask = r_id == d.gid
        for t in THRESHOLDS:
            rows.append({"district": d.NAME_2, "threshold": t,
                         "flood_km2": float(np.nansum(pv >= t) / 1e4),
                         "buildings": int((b_p[bmask] >= t).sum()),
                         "people": float(popv[np.nan_to_num(pv, nan=0.0) >= t].sum()),
                         "roads_km": float(rlen[rmask][r_p[rmask] >= t].sum() / 1000)})
    df = pd.DataFrame(rows)
    keep = df[(df.threshold == 0.5) & (df.flood_km2 >= 1.0)].district
    df = df[df.district.isin(keep)]
    df.to_csv(get_output_path(f"tables/threshold_exposure_by_district{sfx}.csv"), index=False)
    summ = []
    for metric in ("flood_km2", "buildings", "people", "roads_km"):
        w = df.pivot(index="district", columns="threshold", values=metric)
        top3 = set(w[0.5].nlargest(3).index)
        for t in THRESHOLDS:
            rho = spearmanr(w[0.5], w[t]).statistic if w[0.5].nunique() > 1 and w[t].nunique() > 1 else np.nan
            summ.append({"metric": metric, "threshold": t, "spearman_vs_0.50": rho,
                         "top3_overlap_with_0.50": len(top3 & set(w[t].nlargest(3).index))})
    sd = pd.DataFrame(summ)
    sd.to_csv(get_output_path(f"tables/threshold_exposure_district_rank_stability{sfx}.csv"), index=False)
    log.info("districts kept: %s", list(keep))
    log.info("\n%s", sd.groupby("metric").agg(min_rho=("spearman_vs_0.50", "min"), min_top3=("top3_overlap_with_0.50", "min")).round(3))
    log.info("0.50: %s", df[df.threshold == 0.5][["district", "flood_km2", "buildings", "people", "roads_km"]].round(1).to_string(index=False))


if __name__ == "__main__":
    import sys
    main("--prior-corrected" in sys.argv)
