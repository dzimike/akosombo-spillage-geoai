from __future__ import annotations
"""
Label-leakage ablation for the XGBoost flood classifier.

The training labels come from Otsu-thresholded SAR change detection and dual
NDWI/MNDWI thresholding, and those same signals are predictors. This script
retrains the classifier (identical configuration and spatial split) with the
label-defining predictors removed, to measure how much of the reported
accuracy depends on them.

Variants
--------
full                      all 28 features (the deployed model)
no_optical_label_bands    drop event-period NDWI and MNDWI
no_label_signals          drop event-period NDWI, MNDWI and all SAR bands
terrain_settlement_only   drop every event-period spectral and SAR band

Evaluation: test blocks at threshold 0.50, plus true leave-one-block-out CV
summary (mean/median/min block ROC-AUC, mean block F1).

Output: outputs/tables/xgb_ablation_label_bands.csv
"""
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.models.train_random_forest import NON_FEATURE_COLS, add_spatial_blocks, spatial_train_test_split
from src.utils.paths import get_project_root, get_output_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EVENT_OPTICAL = ["ndvi_event", "ndwi_event", "mndwi_event", "ndbi_event"]
SAR = ["sar_vv_pre", "sar_vv_event", "sar_vv_change", "sar_vh_pre", "sar_vh_event"]
VARIANTS = {
    "full": [],
    "no_optical_label_bands": ["ndwi_event", "mndwi_event"],
    "no_label_signals": ["ndwi_event", "mndwi_event"] + SAR,
    "terrain_settlement_only": EVENT_OPTICAL + SAR,
}


def fit(Xtr, ytr, Xva, yva, cfg, seed):
    m = XGBClassifier(
        n_estimators=cfg.get("n_estimators", 500), learning_rate=cfg.get("learning_rate", 0.03),
        max_depth=cfg.get("max_depth", 6), subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        scale_pos_weight=float((ytr == 0).sum() / max((ytr == 1).sum(), 1)),
        random_state=seed, eval_metric="logloss", early_stopping_rounds=20, n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def main() -> None:
    root = get_project_root()
    cfg_all = yaml.safe_load(open(root / "config/model_config.yaml"))
    cfg, seed = cfg_all["models"]["xgboost"], cfg_all.get("random_seed", 42)
    df = add_spatial_blocks(pd.read_parquet(root / "data/processed/features/model_training_samples.parquet"), 1000)
    all_feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    tr, va, te = spatial_train_test_split(df, 0.70, 0.15, seed)
    rows = []
    for name, drop in VARIANTS.items():
        feats = [c for c in all_feats if c not in drop]
        m = fit(tr[feats].values, tr.flood_label.values, va[feats].values, va.flood_label.values, cfg, seed)
        p = m.predict_proba(te[feats].values)[:, 1]; y = te.flood_label.values; yh = (p >= 0.5).astype(int)
        # leave-one-block-out
        aucs, f1s = [], []
        n_trees = int(m.best_iteration) + 1
        for b in sorted(df.block_id.unique()):
            blk = df[df.block_id == b]; yb = blk.flood_label.values
            if len(blk) < 10 or yb.sum() in (0, len(yb)):
                continue
            rest = df[df.block_id != b]; yr = rest.flood_label.values
            f = XGBClassifier(n_estimators=n_trees, learning_rate=cfg.get("learning_rate", 0.03),
                              max_depth=cfg.get("max_depth", 6), subsample=cfg.get("subsample", 0.8),
                              colsample_bytree=cfg.get("colsample_bytree", 0.8),
                              scale_pos_weight=float((yr == 0).sum() / max((yr == 1).sum(), 1)),
                              random_state=seed, n_jobs=-1)
            f.fit(rest[feats].values, yr)
            pb = f.predict_proba(blk[feats].values)[:, 1]
            aucs.append(roc_auc_score(yb, pb)); f1s.append(f1_score(yb, (pb >= 0.5).astype(int), zero_division=0))
        rows.append({"variant": name, "n_features": len(feats), "dropped": ";".join(drop),
                     "test_roc_auc": roc_auc_score(y, p), "test_f1": f1_score(y, yh),
                     "test_precision": precision_score(y, yh), "test_recall": recall_score(y, yh),
                     "lobo_blocks": len(aucs), "lobo_auc_mean": float(np.mean(aucs)),
                     "lobo_auc_median": float(np.median(aucs)), "lobo_auc_min": float(np.min(aucs)),
                     "lobo_f1_mean": float(np.mean(f1s)), "lobo_f1_min": float(np.min(f1s))})
        log.info("%-24s n=%2d test AUC %.4f F1 %.4f | LOBO AUC mean %.4f median %.4f min %.4f | F1 mean %.4f",
                 name, len(feats), rows[-1]["test_roc_auc"], rows[-1]["test_f1"],
                 rows[-1]["lobo_auc_mean"], rows[-1]["lobo_auc_median"], rows[-1]["lobo_auc_min"], rows[-1]["lobo_f1_mean"])
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("tables/xgb_ablation_label_bands.csv"), index=False)


if __name__ == "__main__":
    main()
