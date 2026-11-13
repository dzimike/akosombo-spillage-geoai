from __future__ import annotations
"""
Optical-window test: how much of the fused extent depends on Sentinel-2 data after the release ended (30 Oct 2023)?

The fused map's optical component uses the 15 Sep-15 Nov median composite. Here the optical rule (NDWI >= 0 and MNDWI >= 0, outside pre-existing
water, ten-pixel minimum mapping unit; src/flood_mapping/water_index_mapping.py) is re-applied to two composites pulled the same way by
src/flood_mapping/s2_release_window.py: 15 Sep-15 Nov (to check that the re-pull reproduces the published optical mask) and 15 Sep-30 Oct. The SAR
component and pre-existing water are unchanged. Output: outputs/tables/optical_window_test.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.flood_mapping.water_index_mapping import remove_small_patches
from src.exposure.district_exposure import POP_SCALE
from src.utils.paths import get_project_root
from src.validation.threshold_exposure_direct import ROAD_SAMPLE_SPACING_M, interpolate_road_points, reproject_population_conserving

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
KM2 = 1e-4  # 10 m pixels to km2


def main() -> None:
    root = get_project_root()
    fl, sat = root / "data/processed/flood_extent", root / "data/processed/satellite"
    fused = rasterio.open(fl / "fused_flood_confidence.tif").read(1)
    opt_pub = rasterio.open(fl / "optical_water_mask.tif").read(1) == 1
    with rasterio.open(sat / "s2_water_indices_windows.tif") as s:
        idx = s.read()
        nod = s.nodata
    pre = fused == 4
    valid_area = fused != 255
    sar = np.isin(fused, [1, 3])
    flood_pub = np.isin(fused, [1, 2, 3])
    tr = rasterio.open(fl / "fused_flood_confidence.tif").transform
    H, W = fused.shape
    pop = reproject_population_conserving(root / "data/processed/population/population_grid_clipped.tif", fl / "fused_flood_confidence.tif") * POP_SCALE
    b = gpd.read_file(root / "data/raw/buildings/osm_buildings.gpkg").to_crs(32630).geometry.centroid
    br, bc = ((tr.f - b.y.values) // 10).astype(int), ((b.x.values - tr.c) // 10).astype(int)
    ok = (br >= 0) & (br < H) & (bc >= 0) & (bc < W); br, bc = br[ok], bc[ok]
    roads = gpd.read_file(root / "data/raw/roads/osm_roads.gpkg").to_crs(32630)
    rx, ry, _, rlen = interpolate_road_points(roads, ROAD_SAMPLE_SPACING_M)
    rr, rc = ((tr.f - ry) // 10).astype(int), ((rx - tr.c) // 10).astype(int)
    okr = (rr >= 0) & (rr < H) & (rc >= 0) & (rc < W); rr, rc, rlen = rr[okr], rc[okr], rlen[okr]

    def exposure(m):
        return {"buildings": int(m[br, bc].sum()), "population": round(float(pop[m].sum())), "roads_km": round(float(rlen[m[rr, rc]].sum()) / 1000, 1)}

    rows = []
    for k, (b_ndwi, b_mndwi) in {"event (15 Sep-15 Nov)": (0, 1), "release (15 Sep-30 Oct)": (2, 3)}.items():
        nd, mn = idx[b_ndwi], idx[b_mndwi]
        valid = (nd != nod) & (mn != nod)
        water = valid & (nd >= 0) & (mn >= 0) & ~pre & valid_area
        opt = remove_small_patches(water.astype(np.uint8)).astype(bool)
        flood = (sar | opt) & ~pre & valid_area
        cls2 = opt & ~sar & flood
        cls3 = opt & sar & flood
        rows.append({
            "composite": k,
            "optical_valid_pct_of_domain": round(100 * (valid & valid_area).sum() / valid_area.sum(), 1),
            "optical_water_km2": round(opt.sum() * KM2, 2),
            "optical_only_km2": round(cls2.sum() * KM2, 2),
            "agreement_km2": round(cls3.sum() * KM2, 2),
            "sar_only_km2": round((sar & ~opt & flood).sum() * KM2, 2),
            "flood_extent_km2": round(flood.sum() * KM2, 2),
            "optical_IoU_with_published_mask_outside_pre_water": round((opt & opt_pub & ~pre).sum() / max((opt | (opt_pub & ~pre)).sum(), 1), 3),
            "published_extent_lost_km2": round((flood_pub & ~flood).sum() * KM2, 2),
            "lost_where_no_valid_optical_km2": round((flood_pub & ~flood & ~valid).sum() * KM2, 2),
            "lost_where_observed_dry_km2": round((flood_pub & ~flood & valid).sum() * KM2, 2),
            "extent_gained_km2": round((flood & ~flood_pub).sum() * KM2, 2),
        } | exposure(flood))
    pub = {"composite": "published map", "optical_water_km2": round(opt_pub.sum() * KM2, 2),
           "optical_only_km2": round((fused == 2).sum() * KM2, 2), "agreement_km2": round((fused == 3).sum() * KM2, 2),
           "sar_only_km2": round((fused == 1).sum() * KM2, 2), "flood_extent_km2": round(flood_pub.sum() * KM2, 2)} | exposure(flood_pub)
    df = pd.DataFrame([pub] + rows)
    out = root / "outputs/tables/optical_window_test.csv"
    df.to_csv(out, index=False)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
