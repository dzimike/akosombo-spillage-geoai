from __future__ import annotations
"""
Agreement of the change-based flood extent (and the previous open-water extent, UNOSAT scene) with the
Charter 844 TanDEM-X flood map (20 Oct 2023), digitised by charter_tandemx_extent.py.

The published map layers sit ~0.4-0.6 km from the project grid (the permanent-water layer overlaps JRC water
at IoU 0.12 as published and 0.34 after one rigid translation). The translation is estimated from the
PERMANENT-WATER layer only (never from flood pixels), then applied to the flood layer. Both variants are written.
Output: outputs/tables/charter_tandemx_agreement.csv
"""
import logging

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from src.utils.paths import get_project_root, get_output_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def best_shift(blue, jrc, valid):
    best = (0.0, 0, 0)
    for dy in range(-80, 21, 4):
        for dx in range(0, 121, 4):
            bb = np.roll(np.roll(blue, dy, 0), dx, 1)
            u = ((bb | jrc) & valid).sum()
            iou = (bb & jrc & valid).sum() / u if u else 0
            if iou > best[0]:
                best = (iou, dy, dx)
    return best


def main() -> None:
    root = get_project_root()
    fl = root / "data/processed/flood_extent"
    rd = lambda p: rasterio.open(p).read(1)
    t = rd(root / "data/processed/validation/charter_tandemx_20231020.tif")
    jrc = rd(root / "data/processed/satellite/jrc_permanent_water.tif") >= 50
    fu = rd(fl / "fused_flood_confidence.tif")
    cand = {
        "fused change-based (classes 1-3)": np.isin(fu, [1, 2, 3]),
        "SAR change (K=3 dB, 2 dates)": rd(fl / "sar_flood_mask_binary.tif") == 1,
        "SAR open water (previous rule)": rd(fl / "sar_openwater_flood_mask.tif") == 1,
        "open-water rule on the 23 Oct scene": rd(fl / "sar_flood_scene_20231023.tif") == 1,
        "single scene 23 Oct, change rule (dVV, dVH < -3 dB)": rd(fl / "sar_flood_scene_change_20231023.tif") == 1,
    }
    ins = t != 255
    rows_ = np.where(ins.any(1))[0]; cols_ = np.where(ins.any(0))[0]
    sl = (slice(rows_[0], rows_[-1] + 1), slice(cols_[0], cols_[-1] + 1))
    iou, dy, dx = best_shift(t[sl] == 2, jrc[sl], ins[sl])
    b0 = t[sl] == 2
    iou0 = (b0 & jrc[sl] & ins[sl]).sum() / ((b0 | jrc[sl]) & ins[sl]).sum()
    log.info("translation from permanent water: dy %d px, dx %d px (IoU %.3f; as published %.3f)", dy, dx, iou, iou0)
    red0 = t == 1
    red1 = np.roll(np.roll(red0, dy, 0), dx, 1) & ins
    prex = fu == 4
    out = []
    for variant, red in (("as published", red0), (f"shifted ({-dy * 10} m N, {dx * 10} m E)", red1)):
        red_buf = ndimage.binary_dilation(red, iterations=5)               # 50 m tolerance
        for n, a in cand.items():
            a = a & ins
            ab = ndimage.binary_dilation(a, iterations=5)
            out.append({"charter_layer": variant, "extent": n,
                        "ours_in_charter_aoi_km2": a.sum() / 1e4,
                        "charter_flooded_km2": red.sum() / 1e4,
                        "overlap_km2": (a & red).sum() / 1e4,
                        "recall_of_charter": (a & red).sum() / max(red.sum(), 1),
                        "recall_50m_tolerance": (red & ab).sum() / max(red.sum(), 1),
                        "precision": (a & red).sum() / max(a.sum(), 1),
                        "precision_50m_tolerance": (a & red_buf).sum() / max(a.sum(), 1),
                        "charter_flooded_in_our_class4_km2": (red & prex).sum() / 1e4})
    df = pd.DataFrame(out)
    df.to_csv(get_output_path("tables/charter_tandemx_agreement.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
