from __future__ import annotations
"""
Agreement between the XGBoost probability surface and the fused flood map over the
whole scene, and a check of the point-sampling exposure method against the
polygon-intersection reference.

* agreement: pixels outside permanent water (class 4) and with a model score;
  model flood = probability >= 0.50 (and >= the validation-selected threshold),
  reference = fused classes 1-3.
* method check: the identical point-sampling procedure of threshold_exposure_direct.py
  (building centroids, road points at 20 m, resampled population) applied to the fused
  flood mask instead of the probability raster.

Outputs: outputs/tables/model_vs_fused_agreement.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.utils.paths import get_project_root, get_output_path
from src.validation.threshold_exposure_direct import (
    ROAD_SAMPLE_SPACING_M, interpolate_road_points, reproject_population_conserving,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(corrected: bool = False) -> None:
    root = get_project_root()
    sfx = "_prior_corrected" if corrected else ""
    prob_path = root / f"outputs/maps/flood_susceptibility_xgb{sfx}.tif"
    with rasterio.open(prob_path) as s:
        prob, tr, crs = s.read(1), s.transform, s.crs
    with rasterio.open(root / "data/processed/flood_extent/fused_flood_confidence.tif") as s:
        fused = s.read(1)
    ref = np.isin(fused, [1, 2, 3])
    dom = (fused != 4) & np.isfinite(prob)
    rows = []
    sel = pd.read_csv(root / "outputs/tables/uncertainty_summary.csv")
    thr_sel = float(sel.loc[sel.metric == "optimal_threshold", "value"].iloc[0])
    for thr in ((0.5, 0.25, 0.75) if corrected else (0.5, thr_sel)):
        m = (prob >= thr) & dom
        r = ref & dom
        tp, fp, fn = (m & r).sum(), (m & ~r).sum(), (~m & r).sum()
        rows.append({"item": f"model_vs_fused@{thr:.2f}", "model_km2": m.sum() / 1e4, "fused_km2": r.sum() / 1e4,
                     "precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
                     "iou": tp / (tp + fp + fn), "scored_domain_km2": dom.sum() / 1e4})
    # method check on the fused map
    bldg = gpd.read_file(root / "data/raw/buildings/osm_buildings.gpkg").to_crs(crs)
    c = bldg.geometry.centroid
    rr, cc = rasterio.transform.rowcol(tr, c.x.values, c.y.values)
    rr, cc = np.array(rr), np.array(cc)
    ok = (rr >= 0) & (rr < ref.shape[0]) & (cc >= 0) & (cc < ref.shape[1])
    nb = int(ref[rr[ok], cc[ok]].sum())
    roads = gpd.read_file(root / "data/raw/roads/osm_roads.gpkg").to_crs(crs)
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].explode(index_parts=False).reset_index(drop=True)
    rx, ry, _, rlen = interpolate_road_points(roads, ROAD_SAMPLE_SPACING_M)
    r2, c2 = rasterio.transform.rowcol(tr, rx, ry); r2, c2 = np.array(r2), np.array(c2)
    ok2 = (r2 >= 0) & (r2 < ref.shape[0]) & (c2 >= 0) & (c2 < ref.shape[1])
    km = float(rlen[ok2][ref[r2[ok2], c2[ok2]]].sum() / 1000)
    pop = reproject_population_conserving(root / "data/processed/population/population_grid_clipped.tif", prob_path)
    pp = float(pop[ref].sum())
    rows.append({"item": "fused_map_point_sampling", "buildings": nb, "roads_km": km, "population_unscaled": pp,
                 "population_scaled_0.858": pp * 366077 / 426499, "fused_km2": ref.sum() / 1e4})
    df = pd.DataFrame(rows)
    df.to_csv(get_output_path(f"tables/model_vs_fused_agreement{sfx}.csv"), index=False)
    log.info("\n%s", df.round(4).to_string(index=False))


if __name__ == "__main__":
    import sys
    main("--prior-corrected" in sys.argv)
