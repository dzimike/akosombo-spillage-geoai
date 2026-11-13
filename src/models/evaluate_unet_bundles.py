from __future__ import annotations
"""
Rebuild the complete U-Net metrics table from the saved model bundles.

The four U-Net experiments were trained as separate processes (two ran in
parallel), and each process writes its own copy of unet_metrics.csv, so the
shared file is not authoritative. This script re-evaluates every saved bundle
on the held-out test blocks (pooled patch-level metrics at threshold 0.5, the
same definition as train_unet.evaluate) and combines them with training
history from the bundle, per-patch statistics from unet_<name>_spatial_cv.csv,
and wall-clock training time parsed from the run logs.

Output: outputs/tables/unet_metrics_complete.csv
"""
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.train_unet import (
    FloodPatchDataset, evaluate, get_spatial_splits, BATCH_SIZE, WORKERS,
    POS_WEIGHT, DICE_WEIGHT,
)
from src.models.unet_model import UNet, DiceBCELoss
from src.utils.paths import get_project_root, get_output_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXPERIMENTS = ["full", "sar_only", "optical_only", "terrain_only"]
LOGS = {"full": "logs/rerun/unet_full_sar.log", "sar_only": "logs/rerun/unet_full_sar.log",
        "optical_only": "logs/rerun/unet_opt_terrain.log", "terrain_only": "logs/rerun/unet_opt_terrain.log"}
TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),")


def training_minutes(root: Path, name: str) -> float:
    start = end = None
    inside = False
    for line in open(root / LOGS[name]):
        if f"Experiment: {name} " in line:
            inside = True
            m = TS.match(line); start = datetime.fromisoformat(m.group(1))
        elif inside and f"Saved: " in line and f"unet_{name}.pt" in line:
            m = TS.match(line); end = datetime.fromisoformat(m.group(1)); break
    return round((end - start).total_seconds() / 60, 1) if start and end else float("nan")


def main() -> None:
    root = get_project_root()
    feat = root / "data/processed/features/flood_model_feature_stack.tif"
    label = root / "data/processed/flood_extent/fused_flood_confidence.tif"
    _, _, test_blocks = get_spatial_splits(root / "data/processed/features/model_training_samples.parquet")
    device = torch.device("cpu")
    rows = []
    for name in EXPERIMENTS:
        pt = root / f"outputs/models/unet_{name}.pt"
        if not pt.exists():
            log.warning("missing bundle %s", pt); continue
        b = torch.load(pt, map_location="cpu", weights_only=False)
        bands = b["band_indices"]
        model = UNet(in_channels=len(bands), base_filters=b.get("base_filters", 16))
        model.load_state_dict(b["model_state"]); model.eval()
        crit = DiceBCELoss(dice_weight=DICE_WEIGHT, pos_weight=POS_WEIGHT)
        crit.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]), reduction="none")
        ds = FloodPatchDataset(feat, label, test_blocks, bands, (b["stats_mean"], b["stats_std"]), augment=False)
        tm = evaluate(model, DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=WORKERS), crit, device)
        hist = b["history"]; vl = [h["val_loss"] for h in hist]
        cvp = root / f"outputs/tables/unet_{name}_spatial_cv.csv"
        pat = pd.read_csv(cvp) if cvp.exists() else pd.DataFrame(columns=["f1"])
        f1 = pat["f1"].astype(float)
        rows.append({
            "experiment": name, "n_bands": len(bands), "n_params": UNet.param_count(model),
            "best_epoch": int(np.argmin(vl)) + 1, "epochs_run": len(hist),
            "best_val_loss": float(np.min(vl)), "max_val_loss": float(np.max(vl)),
            "best_val_f1": float(hist[int(np.argmin(vl))]["f1"]),
            "train_minutes": training_minutes(root, name),
            "test_patches": len(ds), **{f"test_{k}": round(v, 4) for k, v in tm.items()},
            "spatial_cv_patches": len(pat),
            "spatial_cv_mean_f1": round(f1.mean(), 4) if len(f1) else np.nan,
            "spatial_cv_min_f1": round(f1.min(), 4) if len(f1) else np.nan,
            "spatial_cv_max_f1": round(f1.max(), 4) if len(f1) else np.nan,
            "spatial_cv_cv": round(f1.std() / f1.mean(), 3) if len(f1) and f1.mean() > 0 else np.nan,
            "zero_f1_patches": int((f1 < 1e-6).sum()) if len(f1) else 0,
        })
        log.info("%-13s epochs %d best %d | test F1 %.4f IoU %.4f P %.4f R %.4f | patches %d mean F1 %.4f",
                 name, len(hist), rows[-1]["best_epoch"], tm["f1"], tm["iou"], tm["precision"], tm["recall"],
                 len(pat), rows[-1]["spatial_cv_mean_f1"])
    pd.DataFrame(rows).to_csv(get_output_path("tables/unet_metrics_complete.csv"), index=False)


if __name__ == "__main__":
    main()
