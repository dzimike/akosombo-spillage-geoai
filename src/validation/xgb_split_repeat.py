from __future__ import annotations
"""
Repeat the pixel-level XGBoost scoring of the model comparison over ten spatial block splits.

model_comparison_test_blocks.csv scores the deployed XGBoost model on all pixels of the eight test blocks of one block split (seed 42). That split is
one draw: which blocks are held out changes the flood share of the test pixels and every score computed on them. For each of ten seeds this script
retrains the classifier exactly as train_xgboost.py does (same hyperparameters, case-control sample, early stopping on the validation blocks, training
medians for the imputed bands), scores it on all valid pixels of that seed's test blocks at threshold 0.50 for the raw score and for the prior-shift
corrected probability (prior_correct_probability.py), and writes one row per seed and score.

The U-Nets are not retrained: a model trained on the seed-42 training blocks has seen blocks that are test blocks under other seeds, so a U-Net
comparison across splits would leak. Only the XGBoost side is repeated; the U-Net values in the paper are for the seed-42 split.

Output: outputs/tables/xgb_split_repeat.csv, outputs/tables/xgb_split_repeat_summary.csv
"""
import logging

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.windows import Window
from xgboost import XGBClassifier

from src.models.train_random_forest import IMPUTE_PREFIXES, NON_FEATURE_COLS, add_spatial_blocks, spatial_train_test_split
from src.utils.paths import get_output_path, get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
BLOCK_PX = 1000


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12), tp / max(tp + fp + fn, 1)


def main() -> None:
    root = get_project_root()
    cfg_all = yaml.safe_load(open(root / "config/model_config.yaml")); xc = cfg_all["models"]["xgboost"]
    df = add_spatial_blocks(pd.read_parquet(root / "data/processed/features/model_training_samples.parquet"), BLOCK_PX)
    feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    stack = root / "data/processed/features/flood_model_feature_stack.tif"
    label_path = root / "data/processed/flood_extent/fused_flood_confidence.tif"
    with rasterio.open(label_path) as ls:
        fu = ls.read(1); H, W = ls.height, ls.width
    nf = int(np.isin(fu, [1, 2, 3]).sum()); n0 = int((fu == 0).sum()); pi = nf / (nf + n0)
    with rasterio.open(stack) as s:
        tags = {s.tags(i).get("name"): i for i in range(1, s.count + 1)}
    bands = [tags[c] for c in feats]
    rows = []
    cache: dict[str, tuple] = {}

    def block_data(b):
        if b not in cache:
            r, c = map(int, b.split("_"))
            win = Window(c * BLOCK_PX, r * BLOCK_PX, min(BLOCK_PX, W - c * BLOCK_PX), min(BLOCK_PX, H - r * BLOCK_PX))
            with rasterio.open(stack) as s:
                X = np.stack([s.read(k, window=win).astype(np.float32) for k in bands], axis=0)
            cache[b] = (X.reshape(len(feats), -1).T, fu[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width].reshape(-1))
        return cache[b]

    for seed in SEEDS:
        tr, va, te = spatial_train_test_split(df, 0.70, 0.15, seed)
        med = {c: float(tr[c].median()) for c in feats if any(c.startswith(p) for p in IMPUTE_PREFIXES)}
        w = float((tr.flood_label == 0).sum() / max((tr.flood_label == 1).sum(), 1))
        m = XGBClassifier(n_estimators=xc.get("n_estimators", 500), learning_rate=xc.get("learning_rate", 0.03), max_depth=xc.get("max_depth", 6),
                          subsample=xc.get("subsample", 0.8), colsample_bytree=xc.get("colsample_bytree", 0.8), scale_pos_weight=w, random_state=seed,
                          eval_metric="logloss", early_stopping_rounds=20, n_jobs=-1)
        m.fit(tr[feats].values, tr.flood_label.values, eval_set=[(va[feats].values, va.flood_label.values)], verbose=False)
        pi_t = float(tr.flood_label.mean()); pi_s = w * pi_t / (w * pi_t + 1 - pi_t)
        shift = np.log(pi / (1 - pi)) - np.log(pi_s / (1 - pi_s))
        acc = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "blk": []} for k in ("raw", "corrected")}
        for b in sorted(te.block_id.unique()):
            X, lbl = block_data(b); X = X.copy()
            for j, c in enumerate(feats):
                if c in med:
                    bad = ~np.isfinite(X[:, j]); X[bad, j] = med[c]
            core = np.ones(len(X), bool)
            for j, c in enumerate(feats):
                if c not in med:
                    core &= np.isfinite(X[:, j])
            p = np.full(len(X), np.nan, np.float32)
            if core.any():
                p[core] = m.predict_proba(X[core])[:, 1]
            pc = np.clip(p, 1e-6, 1 - 1e-6); q = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + shift)))
            valid = (lbl != 9) & (lbl != 4) & np.isfinite(p); pos = (lbl >= 1) & (lbl <= 3)
            for k, prob in (("raw", p), ("corrected", q)):
                pred = np.nan_to_num(prob, nan=0.0) >= 0.5
                tp = int((pos & pred & valid).sum()); fp = int((~pos & pred & valid).sum()); fn = int((pos & ~pred & valid).sum()); tn = int((~pos & ~pred & valid).sum())
                a = acc[k]; a["tp"] += tp; a["fp"] += fp; a["fn"] += fn; a["tn"] += tn
                if tp + fn > 0:
                    a["blk"].append(prf(tp, fp, fn)[2])
        for k, a in acc.items():
            pr, rc, f1, iou = prf(a["tp"], a["fp"], a["fn"]); npx = a["tp"] + a["fp"] + a["fn"] + a["tn"]
            rows.append({"seed": seed, "score": k, "n_test_blocks": te.block_id.nunique(), "n_pixels": npx, "flood_pct": 100 * (a["tp"] + a["fn"]) / npx,
                         "precision": pr, "recall": rc, "f1": f1, "iou": iou, "predicted_flood_km2": (a["tp"] + a["fp"]) / 1e4,
                         "reference_flood_km2": (a["tp"] + a["fn"]) / 1e4, "block_f1_mean": float(np.mean(a["blk"])) if a["blk"] else np.nan,
                         "blocks_with_flood": len(a["blk"]), "best_iteration": int(m.best_iteration), "logit_shift": shift})
            log.info("seed %2d %-9s P %.3f R %.3f F1 %.4f | flood %.2f%% | pred %.1f ref %.1f km2", seed, k, pr, rc, f1, rows[-1]["flood_pct"],
                     rows[-1]["predicted_flood_km2"], rows[-1]["reference_flood_km2"])
        pd.DataFrame(rows).to_csv(get_output_path("tables/xgb_split_repeat.csv"), index=False)
    d = pd.DataFrame(rows)
    s = d.groupby("score").agg(f1_mean=("f1", "mean"), f1_sd=("f1", "std"), f1_min=("f1", "min"), f1_max=("f1", "max"), precision_mean=("precision", "mean"),
                              recall_mean=("recall", "mean"), predicted_km2_mean=("predicted_flood_km2", "mean"), flood_pct_min=("flood_pct", "min"),
                              flood_pct_max=("flood_pct", "max")).reset_index()
    s.to_csv(get_output_path("tables/xgb_split_repeat_summary.csv"), index=False)
    log.info("\n%s", s.round(3).to_string())


if __name__ == "__main__":
    main()
