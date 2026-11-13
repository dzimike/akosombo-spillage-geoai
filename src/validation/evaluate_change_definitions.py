from __future__ import annotations
"""
Score candidate change-based flood definitions against two independent references:
  (1) the UNOSAT FL20231018GHA FloodExtent (Sentinel-1, 23 Oct 2023), pixel agreement in the common domain;
  (2) the Charter-844 WorldView-2 Mepe map (18 Oct 2023): share of the OSM buildings that the analyst flagged
      as affected that fall inside the candidate extent (recall), and the share of unflagged OSM buildings in
      the same frame that fall inside it (false-positive rate).

Candidate = (event dates with dVV and dVH < -k dB) >= m, outside permanent / pre-existing water (JRC, sea, pre-event
optical water, pre-event SAR open water), slope <= 5 deg, 10-pixel minimum mapping unit.

Output: outputs/tables/change_definition_candidates.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize

from src.flood_mapping.sar_change_detection import remove_small_patches
from src.flood_mapping.water_masks import static_water
from src.utils.paths import get_project_root, get_output_path
from src.utils.raster import read_aligned
from src.validation.compare_unosat import UN, metrics, rd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    root = get_project_root()
    sat, fl = root / "data/processed/satellite", root / "data/processed/flood_extent"
    ref = sat / "s1_pre_event_vv.tif"
    with rasterio.open(ref) as s:
        tr, crs, shape = s.transform, s.crs, s.shape
    from src.flood_mapping.change_counts import count_flags
    cnt = {k: count_flags(sat, k)[0] for k in (2, 3, 4)}
    nobs = count_flags(sat, 3)[1].astype(float)
    static = static_water(ref, sat, fl)["union"]
    pre_w = rd(fl / "sar_pre_water_mask.tif") == 1
    pre_exist = static | pre_w
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), str(ref), Resampling.bilinear)
    flat = ~(np.isfinite(slope) & (slope > 5))
    sea = rd(fl / "sea_mask.tif") == 1
    g = gpd.read_file(root / UN.format("AnalysisExtent")).to_crs(crs)
    dom = rasterize(((geom, 1) for geom in g.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool) & ~sea
    g = gpd.read_file(root / UN.format("FloodExtent")).to_crs(crs)
    uf = rasterize(((geom, 1) for geom in g.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool) & dom
    # Mepe buildings
    mb = gpd.read_file(root / "data/processed/validation/charter_mepe_flooded_buildings.gpkg").to_crs(crs)
    cen = mb.geometry.centroid
    rr, cc = rasterio.transform.rowcol(tr, cen.x.values, cen.y.values); rr, cc = np.array(rr), np.array(cc)
    ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    flag = mb.charter_affected.values
    log.info("coverage: n_obs>0 on %.1f%% of domain", 100 * np.isfinite(nobs).mean())
    rows = []
    for k in (2, 3, 4):
        for m in (1, 2, 3):
            cand = (cnt[k] >= m) & ~pre_exist & flat
            cand = remove_small_patches(np.nan_to_num(cand).astype(np.uint8)) == 1
            r = metrics(cand, uf, dom); r.update(k_dB=k, persistence_dates=m, total_km2=cand.sum() / 1e4)
            inside = np.zeros(len(mb), bool); inside[ok] = cand[rr[ok], cc[ok]]
            r["mepe_flagged_recall"] = float(inside[flag].mean()); r["mepe_unflagged_fpr"] = float(inside[~flag].mean())
            rows.append(r)
    # reference rows: previous open-water primary extent
    prev = np.isin(rd(root / "archive/v5_open_water_labels/fused_flood_confidence.tif"), [1, 2, 3])
    r = metrics(prev, uf, dom); r.update(k_dB=np.nan, persistence_dates=np.nan, total_km2=prev.sum() / 1e4)
    inside = np.zeros(len(mb), bool); inside[ok] = prev[rr[ok], cc[ok]]
    r["mepe_flagged_recall"] = float(inside[flag].mean()); r["mepe_unflagged_fpr"] = float(inside[~flag].mean()); r["definition"] = "previous open-water extent"
    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(get_output_path("tables/change_definition_candidates.csv"), index=False)
    log.info("\n%s", df[["definition", "k_dB", "persistence_dates", "total_km2", "ours_km2", "overlap_km2", "precision_vs_unosat", "recall_vs_unosat", "f1", "mepe_flagged_recall", "mepe_unflagged_fpr"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
