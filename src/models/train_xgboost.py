from __future__ import annotations
"""
Phase 7 – Task 7.3: XGBoost Flood Susceptibility Model

Trains an XGBoost gradient-boosted classifier on the assembled feature stack
with the same spatial-block split used for Random Forest.

Requires: xgboost + libomp  (run via /opt/miniconda3/envs/flood/bin/python)

Outputs
-------
outputs/models/xgb_flood_model.joblib    — model bundle (model + meta)
outputs/tables/xgb_metrics.csv
outputs/figures/xgb_feature_importance.png
outputs/figures/xgb_roc_curve.png
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.utils.paths import get_project_root, get_output_path, ensure_dir
from src.utils.logging_utils import Timer, log_run
from src.models.train_random_forest import (
    NON_FEATURE_COLS, IMPUTE_PREFIXES,
    add_spatial_blocks, spatial_train_test_split,
    evaluate_split, plot_roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def plot_feature_importance(importances: np.ndarray, names: list[str],
                            path: Path) -> None:
    import matplotlib.pyplot as plt
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.32)))
    ax.barh([names[i] for i in idx], importances[idx], color="#2E7D32")
    ax.set_xlabel("Gain-based Importance")
    ax.set_title("XGBoost — Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Feature importance: %s", path)


def main(config_path: str | None = None) -> None:
    proj_root   = get_project_root()
    model_cfg_p = config_path or str(proj_root / "config" / "model_config.yaml")
    with open(model_cfg_p) as f:
        cfg = yaml.safe_load(f)

    for d in ["models", "tables", "figures"]:
        ensure_dir(get_output_path(d))

    sample_path = proj_root / "data/processed/features/model_training_samples.parquet"
    if not sample_path.exists():
        log.error("Training samples not found: %s", sample_path); return

    # ---- Load & prepare --------------------------------------------------
    df = pd.read_parquet(sample_path)
    log.info("Loaded %d samples × %d cols", len(df), len(df.columns))

    df = add_spatial_blocks(df, block_size_px=1000)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    log.info("Features: %d", len(feature_cols))

    seed      = cfg.get("random_seed", 42)
    xgb_cfg   = cfg["models"]["xgboost"]
    d_split   = cfg.get("data_split", {})
    train_frac = d_split.get("train", 0.70)
    val_frac   = d_split.get("validation", 0.15)

    train, val, test = spatial_train_test_split(
        df, train_frac=train_frac, val_frac=val_frac, seed=seed)

    X_train = train[feature_cols].values; y_train = train["flood_label"].values
    X_val   = val[feature_cols].values;   y_val   = val["flood_label"].values
    X_test  = test[feature_cols].values;  y_test  = test["flood_label"].values

    impute_medians = {
        col: float(train[col].median())
        for col in feature_cols
        if any(col.startswith(p) for p in IMPUTE_PREFIXES)
    }

    # ---- Train -----------------------------------------------------------
    from xgboost import XGBClassifier

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    log.info("scale_pos_weight = %.2f", scale_pos_weight)

    with Timer() as t:
        model = XGBClassifier(
            n_estimators          = xgb_cfg.get("n_estimators", 500),
            learning_rate         = xgb_cfg.get("learning_rate", 0.03),
            max_depth             = xgb_cfg.get("max_depth", 6),
            subsample             = xgb_cfg.get("subsample", 0.8),
            colsample_bytree      = xgb_cfg.get("colsample_bytree", 0.8),
            scale_pos_weight      = scale_pos_weight,
            random_state          = seed,
            eval_metric           = "logloss",
            early_stopping_rounds = 20,
            n_jobs                = -1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    log.info("XGBoost trained in %.1f s  (best iteration: %d)",
             t.elapsed, model.best_iteration)

    # ---- Evaluate --------------------------------------------------------
    metrics = [
        evaluate_split(model, X_train, y_train, "train"),
        evaluate_split(model, X_val,   y_val,   "validation"),
        evaluate_split(model, X_test,  y_test,  "test"),
    ]
    for m in metrics:
        log.info("  %-12s  ROC-AUC=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f",
                 m["split"], m["roc_auc"], m["f1"], m["precision"], m["recall"])

    # ---- Save ------------------------------------------------------------
    model_bundle = {
        "model":           model,
        "feature_cols":    feature_cols,
        "impute_medians":  impute_medians,
        "impute_prefixes": IMPUTE_PREFIXES,
    }
    model_path   = get_output_path("models/xgb_flood_model.joblib")
    metrics_path = get_output_path("tables/xgb_metrics.csv")
    fi_path      = get_output_path("figures/xgb_feature_importance.png")
    roc_path     = get_output_path("figures/xgb_roc_curve.png")

    joblib.dump(model_bundle, model_path, compress=3)
    log.info("Model bundle saved: %s", model_path)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    plot_feature_importance(model.feature_importances_, feature_cols, fi_path)
    plot_roc_curve(model, X_test, y_test, roc_path)

    log_run(
        script=__file__,
        input_paths=[str(sample_path)],
        output_paths=[str(model_path), str(metrics_path)],
        parameters={**xgb_cfg,
                    "scale_pos_weight":  round(scale_pos_weight, 2),
                    "best_iteration":    model.best_iteration,
                    "roc_auc_test":      metrics[2]["roc_auc"],
                    "f1_test":           metrics[2]["f1"]},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost flood model")
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
