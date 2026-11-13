from __future__ import annotations
"""
Phase 9 – Tasks 9.2–9.4: Sensitivity & Uncertainty Analysis

9.2 Probability threshold sensitivity
   Sweeps the susceptibility probability threshold from 0.05 to 0.95,
   recording precision / recall / F1 / predicted flood area at each step.
   Identifies the precision-recall break-even and maximum-F1 points.

9.3 Recovery index weight sensitivity (Monte Carlo)
   Samples 2 000 random weight vectors from a symmetric Dirichlet(α=2)
   distribution and recomputes the community recovery index for each draw.
   Reports rank-stability: how often each community holds its ranking.

9.4 Spatial block cross-validation
   Evaluates XGBoost performance within each of the 42 spatial blocks
   (treating each block as a mini hold-out set while using the rest for
   prediction). Reports per-block ROC-AUC, F1, and flood prevalence.

Outputs
-------
outputs/tables/threshold_sensitivity.csv
outputs/figures/threshold_sensitivity_plot.png
outputs/tables/recovery_weight_sensitivity.csv
outputs/figures/recovery_rank_stability.png
outputs/tables/spatial_cv_report.csv
outputs/figures/spatial_cv_auc_map.png
outputs/tables/uncertainty_summary.csv
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
N_MC          = 2_000   # Monte Carlo draws for weight sensitivity
PIXEL_AREA_HA = 0.01    # 10 m × 10 m = 0.01 ha


# ---------------------------------------------------------------------------
# 9.2 Threshold sensitivity
# ---------------------------------------------------------------------------

def threshold_sensitivity(y_true: np.ndarray, y_proba: np.ndarray,
                           thresholds: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import precision_score, recall_score, f1_score

    rows = []
    for thr in thresholds:
        y_pred        = (y_proba >= thr).astype(int)
        pred_flood_px = int(y_pred.sum())
        rows.append({
            "threshold":      round(float(thr), 2),
            "precision":      round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":         round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":             round(f1_score(y_true, y_pred, zero_division=0), 4),
            "predicted_flood_pct": round(100 * float(y_pred.mean()), 3),
        })
    return pd.DataFrame(rows)


def plot_threshold_sensitivity(df: pd.DataFrame, output_path: Path,
                               opt_thr: float | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Precision / Recall / F1 vs threshold
    ax = axes[0]
    ax.plot(df.threshold, df.precision, label="Precision", color="#1565C0", lw=2)
    ax.plot(df.threshold, df.recall,    label="Recall",    color="#C62828", lw=2)
    ax.plot(df.threshold, df.f1,        label="F1",        color="#2E7D32", lw=2, linestyle="--")
    best_row = df.loc[df.f1.idxmax()] if opt_thr is None else df.loc[(df.threshold - opt_thr).abs().idxmin()]
    ax.axvline(best_row.threshold, color="grey", lw=1, linestyle=":")
    ax.annotate(f"F1={best_row.f1:.3f}\n@thr={best_row.threshold:.2f} (selected on validation)",
                xy=(best_row.threshold, best_row.f1),
                xytext=(best_row.threshold + 0.05, best_row.f1 - 0.08),
                arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9)
    ax.set_xlabel("Probability Threshold"); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.legend(); ax.set_ylim(0, 1.05)

    # Predicted flood % vs threshold
    ax2 = axes[1]
    ax2.plot(df.threshold, df.predicted_flood_pct,
             color="#6A1B9A", lw=2)
    ax2.set_xlabel("Probability Threshold")
    ax2.set_ylabel("Predicted Flood  (%  of valid pixels)")
    ax2.set_title("Predicted Flood Coverage vs Threshold")
    ax2.axvline(best_row.threshold, color="grey", lw=1, linestyle=":")

    fig.suptitle("XGBoost Susceptibility Model — Threshold Sensitivity", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt_close; plt_close.close(fig)
    log.info("Threshold sensitivity plot: %s", output_path)


# ---------------------------------------------------------------------------
# 9.3 Recovery weight Monte Carlo
# ---------------------------------------------------------------------------

def recovery_weight_mc(df_recovery: pd.DataFrame,
                       component_cols: dict[str, str],
                       n_draws: int = N_MC,
                       seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Sample Dirichlet(α=2) weight vectors, recompute recovery index each time,
    record community rank per draw. Returns rank frequency table.
    """
    rng    = np.random.default_rng(seed)
    comps  = [c for c in component_cols.values() if c in df_recovery.columns]
    n_comp = len(comps)
    if n_comp == 0:
        log.warning("No valid component columns for MC sensitivity"); return pd.DataFrame()

    communities = df_recovery["community_name"].values
    n_comm      = len(communities)

    # Normalise each component to [0,1] once
    X = np.zeros((n_comm, n_comp), dtype=float)
    for j, col in enumerate(comps):
        v  = df_recovery[col].fillna(df_recovery[col].median()).values.astype(float)
        lo, hi = v.min(), v.max()
        X[:, j] = (v - lo) / (hi - lo) if hi > lo else 0.5

    rank_counts = np.zeros((n_comm, n_comm), dtype=int)  # [community, rank-1]

    for _ in range(n_draws):
        w   = rng.dirichlet(np.full(n_comp, 2.0))
        idx = np.argsort(X @ w)[::-1]   # descending index (best first)
        for rank, comm_idx in enumerate(idx):
            rank_counts[comm_idx, rank] += 1

    rows = []
    for i, comm in enumerate(communities):
        row = {"community_name": comm}
        for r in range(n_comm):
            row[f"rank_{r+1}_pct"] = round(100 * rank_counts[i, r] / n_draws, 1)
        row["modal_rank"]  = int(np.argmax(rank_counts[i]) + 1)
        row["rank_std"]    = round(float(np.std(
            np.repeat(np.arange(1, n_comm+1), rank_counts[i]))), 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("modal_rank")


def plot_rank_stability(df_mc: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rank_cols = [c for c in df_mc.columns if c.startswith("rank_") and c.endswith("_pct")]
    communities = df_mc["community_name"].values
    n_comm = len(communities)
    data   = df_mc[rank_cols].values   # (n_comm, n_ranks)

    cmap   = plt.cm.RdYlGn_r
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_comm)
    bottom = np.zeros(n_comm)
    for r in range(n_comm):
        ax.bar(x, data[:, r], bottom=bottom,
               color=cmap(r / max(n_comm - 1, 1)),
               label=f"Rank {r+1}", edgecolor="white", linewidth=0.4)
        bottom += data[:, r]

    ax.set_xticks(x)
    ax.set_xticklabels(communities, rotation=30, ha="right")
    ax.set_ylabel("% of MC draws")
    ax.set_title(f"Recovery Rank Stability  (Dirichlet MC, n={N_MC:,})")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt_close; plt_close.close(fig)
    log.info("Rank stability plot: %s", output_path)


# ---------------------------------------------------------------------------
# 9.4 Spatial block cross-validation
# ---------------------------------------------------------------------------

def spatial_block_cv(df: pd.DataFrame, model, feature_cols: list[str],
                     split_of_block: dict | None = None) -> pd.DataFrame:
    """True leave-one-block-out CV.

    For each 10 km block, a fresh XGBoost model (same hyperparameters as the
    deployed model, with the tree count fixed at its early-stopping optimum and
    scale_pos_weight recomputed on the training fold) is trained on all OTHER
    blocks and evaluated on the held-out block at threshold 0.50. The deployed
    model's own per-block scores are kept alongside for reference: they are
    in-sample for blocks in its training partition and are NOT hold-out
    estimates.
    """
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from xgboost import XGBClassifier

    params = model.get_params()
    params.update(n_estimators=int(model.best_iteration) + 1, early_stopping_rounds=None, n_jobs=-1)

    df = df.copy()
    df["block_id"] = (df.pixel_row // BLOCK_SIZE_PX).astype(str) + "_" + \
                     (df.pixel_col // BLOCK_SIZE_PX).astype(str)

    rows = []
    for block in sorted(df.block_id.unique()):
        block_df = df[df.block_id == block]
        if len(block_df) < 10:
            continue
        X = block_df[feature_cols].values
        y = block_df["flood_label"].values
        if y.sum() == 0 or y.sum() == len(y):
            log.debug("Block %s skipped (single class)", block)
            continue
        rest = df[df.block_id != block]
        y_rest = rest["flood_label"].values
        fold = XGBClassifier(**{**params, "scale_pos_weight":
                                float((y_rest == 0).sum() / max((y_rest == 1).sum(), 1))})
        fold.fit(rest[feature_cols].values, y_rest, verbose=False)
        proba  = fold.predict_proba(X)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        proba_single = model.predict_proba(X)[:, 1]
        rows.append({
            "block_id":     block,
            "main_split":   (split_of_block or {}).get(block, ""),
            "n_samples":    len(y),
            "flood_pct":    round(100 * float(y.mean()), 2),
            "roc_auc":      round(roc_auc_score(y, proba), 4),
            "f1":           round(f1_score(y, y_pred, zero_division=0), 4),
            "precision":    round(precision_score(y, y_pred, zero_division=0), 4),
            "recall":       round(recall_score(y, y_pred, zero_division=0), 4),
            "single_model_roc_auc": round(roc_auc_score(y, proba_single), 4),
            "single_model_f1": round(f1_score(y, (proba_single >= 0.5).astype(int), zero_division=0), 4),
        })

    return pd.DataFrame(rows)


def plot_spatial_cv(df_cv: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    # Sort by ROC-AUC so bars read top-to-bottom best→worst
    df_sorted = df_cv.sort_values("roc_auc", ascending=True).reset_index(drop=True)
    # Label: "block_id  (n=NNN, flood=X.X%)"
    labels = [
        f"Block {row.block_id}  (n={row.n_samples:,}, flood={row.flood_pct:.1f}%)"
        for _, row in df_sorted.iterrows()
    ]
    y_pos = np.arange(len(labels))
    bar_h = 0.55
    fig_h = max(6, len(labels) * 0.45 + 1.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, fig_h))

    for ax, col, color, xlabel, title in [
        (axes[0], "roc_auc", "#1565C0", "ROC-AUC",  "Per-Block ROC-AUC"),
        (axes[1], "f1",      "#2E7D32", "F1 Score",  "Per-Block F1 Score"),
    ]:
        values = df_sorted[col].values
        bars   = ax.barh(y_pos, values, height=bar_h, color=color, alpha=0.85)

        # Value label inside / outside each bar
        for bar, val in zip(bars, values):
            x_txt = min(val - 0.01, val)
            ax.text(max(val - 0.002, 0.01), bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="right",
                    fontsize=8, color="white", fontweight="bold")

        mean_val = float(values.mean())
        ax.axvline(mean_val, color="#C62828", lw=1.5, linestyle="--",
                   label=f"Mean = {mean_val:.4f}")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=12, pad=8)
        xlim_lo = max(0, float(values.min()) - 0.05)
        ax.set_xlim(xlim_lo, 1.01)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.legend(fontsize=9)

    fig.suptitle("Spatial Block Cross-Validation — XGBoost", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Spatial CV plot: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    proj_root = get_project_root()
    config    = load_config(config_path)
    for d in ["tables", "figures"]:
        ensure_dir(get_output_path(d))

    sample_path    = get_processed_path("features/model_training_samples.parquet", config)
    model_path     = proj_root / "outputs" / "models" / "xgb_flood_model.joblib"
    recovery_path  = proj_root / "outputs" / "tables" / "community_recovery_index.csv"

    if not sample_path.exists():
        log.error("Training parquet not found"); return
    if not model_path.exists():
        log.error("XGB model not found"); return

    with Timer() as t:
        # ---- Load data --------------------------------------------------
        log.info("Loading data…")
        df     = pd.read_parquet(sample_path)
        bundle = joblib.load(model_path)
        model  = bundle["model"]
        feat   = bundle["feature_cols"]

        # Reproduce test split for threshold + CV analyses
        df["block_id"] = (df.pixel_row // BLOCK_SIZE_PX).astype(str) + "_" + \
                         (df.pixel_col // BLOCK_SIZE_PX).astype(str)
        rng    = np.random.default_rng(RANDOM_SEED)
        blocks = df.block_id.unique(); rng.shuffle(blocks)
        n      = len(blocks)
        test   = df[df.block_id.isin(blocks[int(n * 0.85):])]
        n_tr, n_va = int(n * 0.70), int(n * 0.15)
        val    = df[df.block_id.isin(blocks[n_tr:n_tr + n_va])]
        split_of_block = {**{b: "train" for b in blocks[:n_tr]},
                          **{b: "validation" for b in blocks[n_tr:n_tr + n_va]},
                          **{b: "test" for b in blocks[n_tr + n_va:]}}
        X_test = test[feat].values
        y_test = test["flood_label"].values
        y_prob = model.predict_proba(X_test)[:, 1]
        log.info("Test set: %d samples", len(test))

        # ---- 9.2 Threshold sensitivity ----------------------------------
        log.info("9.2 Threshold sensitivity sweep…")
        thresholds = np.arange(0.05, 0.96, 0.05)
        df_thr = threshold_sensitivity(y_test, y_prob, thresholds)
        thr_path = get_output_path("tables/threshold_sensitivity.csv")
        df_thr.to_csv(thr_path, index=False)
        # Optimal threshold is selected on the validation blocks and then
        # reported on the test blocks (avoids selecting on test data).
        from src.validation.confusion_matrix import find_optimal_threshold
        val_prob = model.predict_proba(val[feat].values)[:, 1]
        opt_thr  = find_optimal_threshold(val["flood_label"].values, val_prob)
        best = df_thr.loc[(df_thr.threshold - opt_thr).abs().idxmin()]
        log.info("  Validation-selected threshold=%.2f -> test F1=%.4f "
                 "(precision=%.4f  recall=%.4f); test-set max F1=%.4f at %.2f",
                 opt_thr, best.f1, best.precision, best.recall,
                 df_thr.f1.max(), df_thr.loc[df_thr.f1.idxmax(), "threshold"])
        plot_threshold_sensitivity(
            df_thr, get_output_path("figures/threshold_sensitivity_plot.png"), opt_thr)

        # ---- 9.3 Recovery weight sensitivity ----------------------------
        summary_rows = []
        if recovery_path.exists():
            log.info("9.3 Recovery weight Monte Carlo (%d draws)…", N_MC)
            df_rec = pd.read_csv(recovery_path)
            component_cols = {
                "vegetation_recovery":      "mean_recovery_ratio",
                "water_recession":          "water_recession_index",
                "road_accessibility":       "road_accessibility_index",
                "settlement_activity_proxy":"settlement_proxy",
            }
            df_mc = recovery_weight_mc(df_rec, component_cols)
            if not df_mc.empty:
                mc_path = get_output_path("tables/recovery_weight_sensitivity.csv")
                df_mc.to_csv(mc_path, index=False)
                log.info("Recovery rank stability (modal rank / rank_std):\n%s",
                         df_mc[["community_name","modal_rank","rank_std"]].to_string(index=False))
                plot_rank_stability(
                    df_mc, get_output_path("figures/recovery_rank_stability.png"))
                summary_rows.append({
                    "analysis": "recovery_weight_mc",
                    "metric": "mean_rank_std",
                    "value": round(float(df_mc["rank_std"].mean()), 3),
                    "note": f"{N_MC} Dirichlet(alpha=2) draws",
                })
        else:
            log.warning("Recovery index table not found; skipping Task 9.3")

        # ---- 9.4 Spatial block CV ---------------------------------------
        log.info("9.4 Spatial block cross-validation…")
        df_cv = spatial_block_cv(df, model, feat, split_of_block)
        cv_path = get_output_path("tables/spatial_cv_report.csv")
        df_cv.to_csv(cv_path, index=False)
        log.info("  Per-block ROC-AUC: mean=%.4f  min=%.4f  max=%.4f",
                 df_cv.roc_auc.mean(), df_cv.roc_auc.min(), df_cv.roc_auc.max())
        log.info("  Per-block F1:      mean=%.4f  min=%.4f  max=%.4f",
                 df_cv.f1.mean(), df_cv.f1.min(), df_cv.f1.max())
        plot_spatial_cv(df_cv, get_output_path("figures/spatial_cv_plot.png"))

        summary_rows += [
            {"analysis": "threshold_sensitivity",
             "metric": "max_f1",
             "value": round(float(best.f1), 4),
             "note": f"at threshold={best.threshold:.2f}"},
            {"analysis": "threshold_sensitivity",
             "metric": "optimal_threshold",
             "value": round(float(best.threshold), 2),
             "note": "maximises F1 on validation blocks"},
            {"analysis": "spatial_block_cv",
             "metric": "mean_roc_auc",
             "value": round(float(df_cv.roc_auc.mean()), 4),
             "note": f"{len(df_cv)} blocks"},
            {"analysis": "spatial_block_cv",
             "metric": "min_roc_auc",
             "value": round(float(df_cv.roc_auc.min()), 4),
             "note": "worst block"},
            {"analysis": "spatial_block_cv",
             "metric": "mean_f1",
             "value": round(float(df_cv.f1.mean()), 4),
             "note": f"{len(df_cv)} blocks"},
        ]

        # ---- Summary table ----------------------------------------------
        df_summary = pd.DataFrame(summary_rows)
        summary_path = get_output_path("tables/uncertainty_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        log.info("Uncertainty summary:\n%s", df_summary.to_string(index=False))

    log.info("Uncertainty analysis complete in %.1f s", t.elapsed)
    log_run(
        script=__file__,
        input_paths=[str(sample_path), str(model_path)],
        output_paths=[str(thr_path), str(cv_path), str(summary_path)],
        parameters={"n_mc": N_MC, "n_blocks": len(df_cv)},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uncertainty analysis")
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
