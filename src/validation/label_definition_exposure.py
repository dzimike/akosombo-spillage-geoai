from __future__ import annotations
"""
Exposure under alternative definitions of the flood extent.

The classifier-threshold sweep (threshold_exposure_direct.py) varies only the
probability cut-off. This script varies the definition of the flood label itself,
so the two uncertainty sources can be compared on the same exposure layers:

  primary             fused classes 1-3 (SAR change K=3 dB on >=2 dates, or optical water; outside pre-existing water)
  sar_change_only     SAR change mask alone
  optical_only_mask   optical water mask alone (outside pre-existing water)
  k2_m2, k4_m2        change threshold 2 / 4 dB, 2 dates
  k3_m1, k3_m3        change threshold 3 dB, 1 / 3 dates
  k4_m1               4 dB, 1 date (best F1 against UNOSAT in the candidate grid)
  open_water_rule     previous definition: event open water (VH < -26.7 dB, VV < -15.2 dB) or optical water,
                      outside static water only (SAR pre-event water NOT removed)
  new_open_water_only previous definition restricted to pixels not SAR-dark pre-event

Exposure is computed by the same point-sampling procedure as the threshold sweep
(building centroids, road points every 20 m, total-preserving resampled population,
rescaled to the 366,077 AOI total). Change variants use s1_change_by_date.tif (dates inside the release period 15 Sep - 30 Oct with dVV and dVH below -K dB\nagainst the orbit-matched pre-event median) and include the 10-pixel minimum mapping unit and the slope <= 5 deg mask.

Output: outputs/tables/label_definition_exposure.csv
"""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

from src.flood_mapping.sar_change_detection import remove_small_patches
from src.flood_mapping.water_masks import static_water
from src.utils.paths import get_project_root, get_output_path
from src.utils.raster import read_aligned
from src.validation.threshold_exposure_direct import (
    ROAD_SAMPLE_SPACING_M, interpolate_road_points, reproject_population_conserving,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
POP_SCALE = 366077 / 426499


def rd(p):
    with rasterio.open(p) as s:
        return s.read(1)


def main() -> None:
    root = get_project_root()
    fl, sat = root / "data/processed/flood_extent", root / "data/processed/satellite"
    ref = sat / "s1_pre_event_vv.tif"
    fused = rd(fl / "fused_flood_confidence.tif")
    opt = rd(fl / "optical_water_mask.tif") == 1
    sar_pre = rd(fl / "sar_pre_water_mask.tif") == 1
    static = static_water(ref, sat, fl)["union"]
    pre = fused == 4                                     # pre-existing water (static | SAR pre-event open water)
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), str(ref), Resampling.bilinear)
    flat = ~(np.isfinite(slope) & (slope > 5))
    from src.flood_mapping.change_counts import count_flags
    cnt = {k: count_flags(sat, k)[0] for k in (2, 3, 4)}
    ow = rd(fl / "sar_openwater_flood_mask.tif") == 1

    def change(k, m):
        return remove_small_patches(((cnt[k] >= m) & ~pre & flat).astype(np.uint8)) == 1

    primary = np.isin(fused, [1, 2, 3])
    sar0 = rd(fl / "sar_flood_mask_binary.tif") == 1
    opt_np = opt & ~pre
    variants = {
        "primary": primary,
        "sar_change_only": sar0,
        "optical_only_mask": opt_np,
        "k2_m2": change(2, 2) | opt_np,
        "k4_m2": change(4, 2) | opt_np,
        "k3_m1": change(3, 1) | opt_np,
        "k3_m3": change(3, 3) | opt_np,
        "k4_m1": change(4, 1) | opt_np,
        "open_water_rule": ow | (opt & ~static),
        "new_open_water_only": (ow | (opt & ~static)) & ~sar_pre,
    }
    with rasterio.open(ref) as s:
        tr, crs = s.transform, s.crs
    bldg = gpd.read_file(root / "data/raw/buildings/osm_buildings.gpkg").to_crs(crs)
    c = bldg.geometry.centroid
    br, bc = rasterio.transform.rowcol(tr, c.x.values, c.y.values); br, bc = np.array(br), np.array(bc)
    roads = gpd.read_file(root / "data/raw/roads/osm_roads.gpkg").to_crs(crs)
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].explode(index_parts=False).reset_index(drop=True)
    rx, ry, _, rlen = interpolate_road_points(roads, ROAD_SAMPLE_SPACING_M)
    rr, rc = rasterio.transform.rowcol(tr, rx, ry); rr, rc = np.array(rr), np.array(rc)
    pop = reproject_population_conserving(root / "data/processed/population/population_grid_clipped.tif", ref)
    H, W = primary.shape
    okb = (br >= 0) & (br < H) & (bc >= 0) & (bc < W)
    okr = (rr >= 0) & (rr < H) & (rc >= 0) & (rc < W)
    rows = []
    for name, m in variants.items():
        rows.append({"definition": name, "flood_km2": m.sum() / 1e4,
                     "buildings": int(m[br[okb], bc[okb]].sum()),
                     "roads_km": float(rlen[okr][m[rr[okr], rc[okr]]].sum() / 1000),
                     "population": float(pop[m].sum() * POP_SCALE)})
        log.info("%s", rows[-1])
    df = pd.DataFrame(rows)
    p0 = df.iloc[0]
    for k in ("flood_km2", "buildings", "roads_km", "population"):
        df[f"{k}_vs_primary_pct"] = 100 * (df[k] / p0[k] - 1)
    df.to_csv(get_output_path("tables/label_definition_exposure.csv"), index=False)
    log.info("\n%s", df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
