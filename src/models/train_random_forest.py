from __future__ import annotations
"""
Phase 7 – Task 7.2: Random Forest Flood Susceptibility Model

Trains a Random Forest classifier on the assembled feature stack with
spatial-block train/validation/test splitting to prevent data leakage.

Spatial blocking: pixels are assigned to 10 km × 10 km grid cells (1 000 × 1 000
pixels at 10 m resolution). Blocks are randomly assigned to splits so that
nearby pixels always land in the same partition.

Outputs
-------
outputs/models/rf_flood_model.joblib     — model bundle (model + meta)
outputs/tables/rf_metrics.csv            — precision/recall/F1/AUC per split
outputs/figures/rf_feature_importance.png
outputs/figures/rf_roc_curve.png
"""

import argparse
import csv
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.utils.paths import get_project_root, get_output_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Columns that are metadata / labels, not features
NON_FEATURE_COLS = {
    "pixel_row", "pixel_col",
    "flood_label", "flood_class", "confidence",
    "s2_pre_valid", "sar_data_valid",
    "block_id",
}

# Columns imputed with median during training (cloud / orbit gaps, not bad pixels)
IMPUTE_PREFIXES = ("sar_", "ndvi_pre", "mndwi_pre", "ndwi_pre")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_spatial_blocks(df: pd.DataFrame, block_size_px: int = 1000) -> pd.DataFrame:
    """Assign a spatial block ID from pixel coordinates (10 km grid at 10 m)."""
    df = df.copy()
    df["block_id"] = (
        (df["pixel_row"] // block_size_px).astype(str) + "_" +
        (df["pixel_col"] // block_size_px).astype(str)
    )
    return df


def spatial_train_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by spatial block to prevent leakage between nearby pixels."""
    rng = np.random.default_rng(seed)
    blocks = df["block_id"].unique()
    rng.shuffle(blocks)
    n = len(blocks)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train = df[df["block_id"].isin(blocks[:n_train])]
    val   = df[df["block_id"].isin(blocks[n_train:n_train + n_val])]
    test  = df[df["block_id"].isin(blocks[n_train + n_val:])]
    log.info("Spatial blocks: total=%d  train=%d  val=%d  test=%d",
             n, len(blocks[:n_train]),
             len(blocks[n_train:n_train + n_val]),
             len(blocks[n_train + n_val:]))
    log.info("Sample split:   train=%d  val=%d  test=%d",
             len(train), len(val), len(test))
    return train, val, test


def evaluate_split(model, X: np.ndarray, y: np.ndarray, name: str) -> dict:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, balanced_accuracy_score,
    )
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "split":             name,
        "n":                 len(y),
        "flood_pct":         round(100 * float(y.mean()), 2),
        "precision":         round(precision_score(y, preds, zero_division=0), 4),
        "recall":            round(recall_score(y, preds, zero_division=0), 4),
        "f1":                round(f1_score(y, preds, zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y, proba), 4),
        "pr_auc":            round(average_precision_score(y, proba), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, preds), 4),
    }


def plot_feature_importance(importances: np.ndarray, names: list[str],
                            path: Path) -> None:
    import matplotlib.pyplot as plt
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.32)))
    ax.barh([names[i] for i in idx], importances[idx], color="#1565C0")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Random Forest — Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Feature importance: %s", path)


def plot_roc_curve(model, X_test: np.ndarray, y_test: np.ndarray,
                   path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#C62828", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Random Forest — ROC Curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ROC curve: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    proj_root     = get_project_root()
    model_cfg_p   = config_path or str(proj_root / "config" / "model_config.yaml")
    with open(model_cfg_p) as f:
        cfg = yaml.safe_load(f)

    for d in ["models", "tables", "figures"]:
        ensure_dir(get_output_path(d))

    sample_path = proj_root / "data/processed/features/model_training_samples.parquet"
    if not sample_path.exists():
        log.error("Training samples not found: %s", sample_path); return

    # ---- Load & prepare data ---------------------------------------------
    df = pd.read_parquet(sample_path)
    log.info("Loaded %d samples × %d cols", len(df), len(df.columns))

    df = add_spatial_blocks(df, block_size_px=1000)   # 10 km grid
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    seed    = cfg.get("random_seed", 42)
    rf_cfg  = cfg["models"]["random_forest"]
    d_split = cfg.get("data_split", {})
    train_frac = d_split.get("train", 0.70)
    val_frac   = d_split.get("validation", 0.15)

    train, val, test = spatial_train_test_split(
        df, train_frac=train_frac, val_frac=val_frac, seed=seed)

    X_train = train[feature_cols].values; y_train = train["flood_label"].values
    X_val   = val[feature_cols].values;   y_val   = val["flood_label"].values
    X_test  = test[feature_cols].values;  y_test  = test["flood_label"].values

    # Compute imputation medians from train split only
    impute_medians = {}
    for col in feature_cols:
        if any(col.startswith(p) for p in IMPUTE_PREFIXES):
            impute_medians[col] = float(train[col].median())

    # ---- Train -----------------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier
    with Timer() as t:
        model = RandomForestClassifier(
            n_estimators      = rf_cfg.get("n_estimators", 500),
            max_depth         = rf_cfg.get("max_depth") or None,
            min_samples_leaf  = rf_cfg.get("min_samples_leaf", 5),
            class_weight      = rf_cfg.get("class_weight", "balanced"),
            n_jobs            = rf_cfg.get("n_jobs", -1),
            random_state      = seed,
        )
        model.fit(X_train, y_train)
    log.info("RF trained in %.1f s", t.elapsed)

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
    model_path   = get_output_path("models/rf_flood_model.joblib")
    metrics_path = get_output_path("tables/rf_metrics.csv")
    fi_path      = get_output_path("figures/rf_feature_importance.png")
    roc_path     = get_output_path("figures/rf_roc_curve.png")

    joblib.dump(model_bundle, model_path, compress=3)
    log.info("Model bundle saved: %s", model_path)

    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    log.info("Metrics: %s", metrics_path)

    plot_feature_importance(model.feature_importances_, feature_cols, fi_path)
    plot_roc_curve(model, X_test, y_test, roc_path)

    log_run(
        script=__file__,
        input_paths=[str(sample_path)],
        output_paths=[str(model_path), str(metrics_path)],
        parameters={**rf_cfg,
                    "roc_auc_test": metrics[2]["roc_auc"],
                    "f1_test":      metrics[2]["f1"]},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest flood model")
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
