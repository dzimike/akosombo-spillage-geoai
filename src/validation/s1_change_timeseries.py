from __future__ import annotations
"""
Per-date change time series (orbits 74 and 147, 1 July 2023 - 31 January 2024).

From s1_change_by_date.tif (bit 2 = dVV and dVH < -3 dB against the orbit-matched pre-event median) the area flagged
on each acquisition date is summed, outside pre-existing water and on slope <= 5 deg, for:
  AOI           whole domain land
  A/B/C         the three 15 km reach windows of the open-water series (Mepe-Battor, Sogakope-Sokpoe, Ada-Songor)
  final_flood   pixels of the final fused flood extent (classes 1-3): share flagged on each date
  groups        patch groups of flood_patch_diagnostics.py (river/lagoon-adjacent, isolated, Ningo-Prampram isolated,
                Ada West adjacent) - share flagged on each date

The valid area of every date (bit 0) is reported because individual dates cover only part of the domain (frames).
Output: outputs/tables/s1_change_timeseries.csv
"""
import logging

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

from src.utils.paths import get_project_root, get_output_path
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
WINDOWS = {"A_Mepe_Battor": (872000, 663000), "B_Sogakope_Sokpoe": (888000, 655000), "C_Ada_Songor": (878000, 640600)}
SIZE_M = 15000
GROUPS = {"adjacent_ge2ha": 1, "adjacent_lt2ha": 2, "isolated_ge2ha": 3, "isolated_lt2ha": 4,
          "NingoPrampram_isolated": 5, "AdaWest_adjacent": 6}


def main() -> None:
    root = get_project_root()
    sat, fl = root / "data/processed/satellite", root / "data/processed/flood_extent"
    bands = pd.read_csv(sat / "s1_change_by_date_bands.csv")
    ref = str(sat / "s1_pre_event_vv.tif")
    with rasterio.open(ref) as s:
        tr, H, W = s.transform, s.height, s.width
    fu = rasterio.open(fl / "fused_flood_confidence.tif").read(1)
    pre = fu == 4
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), ref, Resampling.bilinear)
    land = ~pre & ~(np.isfinite(slope) & (slope > 5))
    final = np.isin(fu, [1, 2, 3])
    grp = rasterio.open(fl / "flood_patch_groups.tif").read(1)
    regions = {"AOI_land": land}
    for name, (x0, y0) in WINDOWS.items():
        c0 = int(round((x0 - tr.c) / tr.a)); r0 = int(round((tr.f - (y0 + SIZE_M)) / -tr.e)); n = SIZE_M // 10
        m = np.zeros((H, W), bool); m[r0:r0 + n, c0:c0 + n] = True
        regions[name] = land & m
    regions["final_flood"] = final
    for g, v in GROUPS.items():
        regions[f"patches_{g}"] = grp == v
    rows = []
    with rasterio.open(sat / "s1_change_by_date.tif") as src:
        for i, (orbit, date) in enumerate(zip(bands.orbit, bands.date), 1):
            code = src.read(i)
            valid = (code & 1) > 0
            flag3 = (code & 4) > 0
            for name, m in regions.items():
                v = valid & m
                rows.append({"orbit": orbit, "date": date, "region": name, "valid_km2": v.sum() / 1e4,
                             "region_km2": m.sum() / 1e4, "flagged_km2": (flag3 & v).sum() / 1e4,
                             "flagged_share_of_valid": (flag3 & v).sum() / max(v.sum(), 1),
                             "valid_share_of_region": v.sum() / max(m.sum(), 1)})
            if i % 6 == 0:
                log.info("%d / %d dates", i, len(bands))
    df = pd.DataFrame(rows)
    df.to_csv(get_output_path("tables/s1_change_timeseries.csv"), index=False)
    p = df[df.region.isin(["final_flood"] + [f"patches_{g}" for g in GROUPS])].pivot_table(
        index=["date", "orbit"], columns="region", values="flagged_share_of_valid")
    log.info("\n%s", p.round(3).to_string())


if __name__ == "__main__":
    main()
