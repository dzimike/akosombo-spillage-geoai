from __future__ import annotations
"""
Protocol-matched comparison of the XGBoost and U-Net flood models.

Both model families are held out on the same spatial test blocks, but the
published per-model metrics are not directly comparable: XGBoost is scored on
a stratified pixel sample (~19-40 % flood) and the U-Nets on natural-prevalence
patches that contain at least 64 flood pixels. Here every model's full-scene
probability raster is scored on ALL pixels of the test blocks against the
fused flood-confidence labels:

  positive = classes 1-3 (flood); class 4 (permanent water, incl. the open sea)
  and no-data (9) pixels are excluded, as in both models' training.

All models are scored on the same pixels: those where the XGBoost model has a
score (event-period Sentinel-2 valid, ~96 % of the domain), because the U-Nets
produce a value everywhere (missing optical input is set to the training mean).

Output: outputs/tables/model_comparison_test_blocks.csv
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.models.train_unet import get_spatial_splits, BLOCK_PX
from src.utils.paths import get_project_root, get_output_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS = {
    "XGBoost (raw score)": "outputs/maps/flood_susceptibility_xgb.tif",
    "XGBoost (prior-corrected)": "outputs/maps/flood_susceptibility_xgb_prior_corrected.tif",
    "U-Net full": "outputs/maps/unet_full_flood_prob.tif",
    "U-Net SAR only": "outputs/maps/unet_sar_only_flood_prob.tif",
    "U-Net optical only": "outputs/maps/unet_optical_only_flood_prob.tif",
    "U-Net terrain only": "outputs/maps/unet_terrain_only_flood_prob.tif",
}


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12), tp / max(tp + fp + fn, 1)


def main(thr: float = 0.5) -> None:
    root = get_project_root()
    _, _, test_blocks = get_spatial_splits(root / "data/processed/features/model_training_samples.parquet")
    label_path = root / "data/processed/flood_extent/fused_flood_confidence.tif"
    rows = []
    paths = {n: root / r for n, r in MODELS.items() if (root / r).exists()}
    for n in set(MODELS) - set(paths):
        log.warning("skip %s (missing)", n)
    acc = {n: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "blk": []} for n in paths}
    srcs = {n: rasterio.open(p) for n, p in paths.items()}
    with rasterio.open(label_path) as ls:
        for b in sorted(test_blocks):
            r, c = map(int, b.split("_"))
            win = Window(c * BLOCK_PX, r * BLOCK_PX,
                         min(BLOCK_PX, ls.width - c * BLOCK_PX), min(BLOCK_PX, ls.height - r * BLOCK_PX))
            lbl = ls.read(1, window=win)
            probs = {n: s_.read(1, window=win).astype(np.float32) for n, s_ in srcs.items()}
            valid = (lbl != 9) & (lbl != 4) & np.isfinite(probs["XGBoost (raw score)"])
            pos = (lbl >= 1) & (lbl <= 3)
            for n, prob in probs.items():
                pred = np.nan_to_num(prob, nan=0.0) >= thr
                tp = int((pos & pred & valid).sum()); fp = int((~pos & pred & valid).sum())
                fn = int((pos & ~pred & valid).sum()); tn = int((~pos & ~pred & valid).sum())
                a = acc[n]; a["tp"] += tp; a["fp"] += fp; a["fn"] += fn; a["tn"] += tn
                if tp + fn > 0:
                    a["blk"].append(prf(tp, fp, fn)[2])
    for n, a in acc.items():
        p, rc, f1, iou = prf(a["tp"], a["fp"], a["fn"])
        npx = a["tp"] + a["fp"] + a["fn"] + a["tn"]
        rows.append({"model": n, "n_test_blocks": len(test_blocks), "n_pixels": npx,
                     "flood_pct": 100 * (a["tp"] + a["fn"]) / max(npx, 1), "precision": p, "recall": rc, "f1": f1, "iou": iou,
                     "predicted_flood_km2": (a["tp"] + a["fp"]) / 1e4, "reference_flood_km2": (a["tp"] + a["fn"]) / 1e4,
                     "block_f1_mean": float(np.mean(a["blk"])), "block_f1_min": float(np.min(a["blk"])),
                     "block_f1_max": float(np.max(a["blk"])), "blocks_with_flood": len(a["blk"])})
        log.info("%-20s P %.4f R %.4f F1 %.4f IoU %.4f | block F1 mean %.4f min %.4f (n=%d)",
                 n, p, rc, f1, iou, rows[-1]["block_f1_mean"], rows[-1]["block_f1_min"], len(a["blk"]))
    pd.DataFrame(rows).to_csv(get_output_path("tables/model_comparison_test_blocks.csv"), index=False)


if __name__ == "__main__":
    main()
