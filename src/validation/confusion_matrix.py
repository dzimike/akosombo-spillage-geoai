from __future__ import annotations
"""
Phase 9 – Task 9.2: Model Accuracy Assessment

Uses the held-out spatial test blocks (never seen during training) as the
reference set to assess the XGBoost flood susceptibility model.

Reference truth  = flood_label from fused flood confidence map
Model prediction = XGBoost predicted probability at optimal threshold
                   (threshold maximising F1 on the VALIDATION blocks, swept
                   from 0.05 to 0.95 in 0.05 steps; the test blocks are used
                   only to report performance at that pre-selected threshold)

Metrics reported
----------------
Overall Accuracy (OA), Cohen's Kappa, Precision, Recall, F1, IoU,
ROC-AUC, PR-AUC, Balanced Accuracy — at both the optimal threshold and
a standard 0.5 threshold.

Outputs
-------
outputs/tables/accuracy_assessment.csv
outputs/figures/confusion_matrix.png
outputs/figures/roc_pr_curves.png
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.paths import load_config, get_project_root, get_output_path, \
    get_processed_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BLOCK_SIZE_PX = 1000
RANDOM_SEED   = 42


def reproduce_test_split(df: pd.DataFrame):
    """Return test-split rows using the same seed/fractions as training."""
    df = df.copy()
    df["block_id"] = (df.pixel_row // BLOCK_SIZE_PX).astype(str) + "_" + \
                     (df.pixel_col // BLOCK_SIZE_PX).astype(str)
    rng    = np.random.default_rng(RANDOM_SEED)
    blocks = df.block_id.unique()
    rng.shuffle(blocks)
    n      = len(blocks)
    return df[df.block_id.isin(blocks[int(n * 0.85):])]


def reproduce_val_split(df: pd.DataFrame):
    """Return validation-split rows (same seed/fractions as training)."""
    df = df.copy()
    df["block_id"] = (df.pixel_row // BLOCK_SIZE_PX).astype(str) + "_" + \
                     (df.pixel_col // BLOCK_SIZE_PX).astype(str)
    rng    = np.random.default_rng(RANDOM_SEED)
    blocks = df.block_id.unique()
    rng.shuffle(blocks)
    n      = len(blocks)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    return df[df.block_id.isin(blocks[n_train:n_train + n_val])]


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return the threshold (0.05 grid) that maximises F1 score."""
    from sklearn.metrics import f1_score
    thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)
    best_thr, best_f1 = 0.5, 0.0
    for thr in thresholds:
        f1 = f1_score(y_true, (y_proba >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_proba: np.ndarray, label: str) -> dict:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
        balanced_accuracy_score, cohen_kappa_score,
        accuracy_score,
    )
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "split":              label,
        "n":                  len(y_true),
        "flood_pct":          round(100 * float(y_true.mean()), 2),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "overall_accuracy":   round(accuracy_score(y_true, y_pred), 4),
        "kappa":              round(cohen_kappa_score(y_true, y_pred), 4),
        "precision":          round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":             round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":                 round(f1_score(y_true, y_pred, zero_division=0), 4),
        "iou":                round(iou, 4),
        "roc_auc":            round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc":             round(average_precision_score(y_true, y_proba), 4),
        "balanced_accuracy":  round(balanced_accuracy_score(y_true, y_pred), 4),
    }


def plot_confusion_matrix(tp, fp, fn, tn, threshold: float,
                          output_path: Path) -> None:
    import matplotlib.pyplot as plt

    cm     = np.array([[tn, fp], [fn, tp]])
    labels = ["Non-flood", "Flood"]
    norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Reference")
    ax.set_title(f"Confusion Matrix  (threshold = {threshold:.2f})")

    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{cm[i,j]:,}\n({100*norm[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if norm[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="Row-normalised fraction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix: %s", output_path)


def plot_roc_pr(y_true: np.ndarray, y_proba: np.ndarray,
                opt_thr: float, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    fpr, tpr, _  = roc_curve(y_true, y_proba)
    roc_auc      = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    axes[0].plot(fpr, tpr, color="#C62828", lw=2,
                 label=f"ROC  AUC={roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    # Mark operating point
    from sklearn.metrics import roc_curve as roc_
    fprs, tprs, thrs = roc_(y_true, y_proba)
    idx = np.argmin(np.abs(thrs - opt_thr))
    axes[0].scatter(fprs[idx], tprs[idx], s=120, color="#C62828", zorder=5,
                    label=f"thr={opt_thr:.2f}")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve — XGBoost (test blocks)")
    axes[0].legend(loc="lower right")

    # Precision-Recall
    baseline = float(y_true.mean())
    pr_auc   = auc(rec, prec)
    axes[1].plot(rec, prec, color="#1565C0", lw=2,
                 label=f"PR  AUC={pr_auc:.4f}")
    axes[1].axhline(baseline, color="grey", lw=1, linestyle="--",
                    label=f"Baseline ({baseline:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — XGBoost (test blocks)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ROC / PR curves: %s", output_path)


def main(config_path: str | None = None) -> None:
    proj_root = get_project_root()
    config    = load_config(config_path)
    for d in ["tables", "figures"]:
        ensure_dir(get_output_path(d))

    sample_path = get_processed_path("features/model_training_samples.parquet", config)
    model_path  = proj_root / "outputs" / "models" / "xgb_flood_model.joblib"

    if not sample_path.exists():
        log.error("Training parquet not found: %s", sample_path); return
    if not model_path.exists():
        log.error("XGB model not found: %s", model_path); return

    with Timer() as t:
        log.info("Loading training samples…")
        df   = pd.read_parquet(sample_path)
        test = reproduce_test_split(df)
        log.info("Test split: %d samples  (%.1f%% flood)",
                 len(test), 100 * float(test.flood_label.mean()))

        log.info("Loading XGB model bundle…")
        bundle       = joblib.load(model_path)
        model        = bundle["model"]
        feature_cols = bundle["feature_cols"]

        X_test = test[feature_cols].values
        y_test = test["flood_label"].values

        log.info("Predicting on test set…")
        y_proba = model.predict_proba(X_test)[:, 1]

        # Optimal threshold: maximises F1 on the VALIDATION blocks, not the test blocks
        val = reproduce_val_split(df)
        val_proba = model.predict_proba(val[feature_cols].values)[:, 1]
        opt_thr = find_optimal_threshold(val["flood_label"].values, val_proba)
        log.info("Optimal threshold (max F1 on validation, n=%d): %.2f", len(val), opt_thr)

        metrics_rows = []
        for thr, label in [(0.50, "threshold_0.50"),
                           (opt_thr, f"threshold_opt_{opt_thr:.2f}")]:
            y_pred = (y_proba >= thr).astype(int)
            m = compute_all_metrics(y_test, y_pred, y_proba, label)
            metrics_rows.append(m)
            log.info("  [%s]  OA=%.4f  Kappa=%.4f  F1=%.4f  "
                     "Prec=%.4f  Rec=%.4f  ROC-AUC=%.4f  PR-AUC=%.4f",
                     label, m["overall_accuracy"], m["kappa"], m["f1"],
                     m["precision"], m["recall"], m["roc_auc"], m["pr_auc"])

        # ---- Save metrics ------------------------------------------------
        out_csv = get_output_path("tables/accuracy_assessment.csv")
        pd.DataFrame(metrics_rows).to_csv(out_csv, index=False)
        log.info("Metrics table: %s", out_csv)

        # ---- Plots -------------------------------------------------------
        opt_m = metrics_rows[1]
        plot_confusion_matrix(
            opt_m["TP"], opt_m["FP"], opt_m["FN"], opt_m["TN"], opt_thr,
            get_output_path("figures/confusion_matrix.png"))

        plot_roc_pr(y_test, y_proba, opt_thr,
                    get_output_path("figures/roc_pr_curves.png"))

    log.info("Accuracy assessment complete in %.1f s", t.elapsed)
    log_run(
        script=__file__,
        input_paths=[str(sample_path), str(model_path)],
        output_paths=[str(out_csv)],
        parameters={"optimal_threshold": opt_thr,
                    "roc_auc":  metrics_rows[1]["roc_auc"],
                    "f1_opt":   metrics_rows[1]["f1"],
                    "kappa":    metrics_rows[1]["kappa"]},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flood mapping accuracy assessment")
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
