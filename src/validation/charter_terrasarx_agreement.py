from __future__ import annotations
"""
Agreement of the change-based flood extent (and alternatives) with the UN-SPIDER / Charter 844 TerraSAR-X flood map
of 16 October 2023 ('potentially flooded area'), digitised by charter_terrasarx_extent.py.

The map's JRC water layer overlaps our JRC >= 50 % mask at an intersection over union of 0.51 as published and the best
translation within +-90 m is zero, so no coregistration shift is applied. The map frame covers 238 km2 of our raster
(north-western corner). Output: outputs/tables/charter_terrasarx_agreement.csv
"""
import logging

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from src.utils.paths import get_project_root, get_output_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    root = get_project_root()
    fl = root / "data/processed/flood_extent"
    rd = lambda p: rasterio.open(p).read(1)
    t = rd(root / "data/processed/validation/charter_terrasarx_20231016.tif")
    jrc = rd(root / "data/processed/satellite/jrc_permanent_water.tif") >= 50
    fu = rd(fl / "fused_flood_confidence.tif")
    ins = t != 255
    red = t == 1
    rows = np.where(ins.any(1))[0]; cols = np.where(ins.any(0))[0]
    sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    best = max(((((np.roll(np.roll(t[sl] == 2, dy, 0), dx, 1) & jrc[sl] & ins[sl]).sum()
                  / ((np.roll(np.roll(t[sl] == 2, dy, 0), dx, 1) | jrc[sl]) & ins[sl]).sum()), dy, dx)
                for dy in range(-30, 31, 6) for dx in range(-30, 31, 6)))
    iou0 = ((t[sl] == 2) & jrc[sl] & ins[sl]).sum() / (((t[sl] == 2) | jrc[sl]) & ins[sl]).sum()
    log.info("map water vs JRC>=50: IoU %.3f as published; best shift %s", iou0, best)
    cand = {
        "fused change-based (classes 1-3)": np.isin(fu, [1, 2, 3]),
        "SAR change (K=3 dB, 2 dates)": rd(fl / "sar_flood_mask_binary.tif") == 1,
        "optical water outside pre-existing water": rd(fl / "optical_water_mask.tif") == 1,
        "SAR open water (previous rule)": rd(fl / "sar_openwater_flood_mask.tif") == 1,
        "open-water rule on the 23 Oct scene": rd(fl / "sar_flood_scene_20231023.tif") == 1,
        "single scene 23 Oct, change rule (dVV, dVH < -3 dB)": rd(fl / "sar_flood_scene_change_20231023.tif") == 1,
    }
    cand["optical water outside pre-existing water"] &= fu != 4
    red_buf = ndimage.binary_dilation(red, iterations=5)
    rows_ = []
    for n, a in cand.items():
        a = a & ins & (fu != 4) if n.startswith("optical") else a & ins
        ab = ndimage.binary_dilation(a, iterations=5)
        tp = (a & red).sum()
        rows_.append({"extent": n, "ours_in_frame_km2": a.sum() / 1e4, "terrasarx_flooded_km2": red.sum() / 1e4,
                      "overlap_km2": tp / 1e4, "precision": tp / max(a.sum(), 1), "recall": tp / max(red.sum(), 1),
                      "f1": 2 * tp / max(a.sum() + red.sum(), 1), "recall_50m": (red & ab).sum() / max(red.sum(), 1),
                      "precision_50m": (a & red_buf).sum() / max(a.sum(), 1)})
    # TanDEM-X (20 Oct) map in the common frame, as shifted in charter_tandemx_agreement
    tdx_p = root / "data/processed/validation/charter_tandemx_20231020.tif"
    if tdx_p.exists():
        td = rd(tdx_p)
        td = np.roll(np.roll(td == 1, -36, 0), 56, 1) & (np.roll(np.roll(td != 255, -36, 0), 56, 1))
        common = ins & (np.roll(np.roll(rd(tdx_p) != 255, -36, 0), 56, 1))
        a, b = red & common, td & common
        tp = (a & b).sum()
        rows_.append({"extent": "TanDEM-X 20 Oct flooded (shifted) vs TerraSAR-X 16 Oct, common frame",
                      "ours_in_frame_km2": b.sum() / 1e4, "terrasarx_flooded_km2": a.sum() / 1e4, "overlap_km2": tp / 1e4,
                      "precision": tp / max(b.sum(), 1), "recall": tp / max(a.sum(), 1),
                      "f1": 2 * tp / max(a.sum() + b.sum(), 1), "recall_50m": np.nan, "precision_50m": np.nan})
    rows_.append({"extent": "map flooded inside our pre-existing water (class 4), km2", "ours_in_frame_km2": (red & (fu == 4)).sum() / 1e4})
    df = pd.DataFrame(rows_)
    df.to_csv(get_output_path("tables/charter_terrasarx_agreement.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
