"""
Google Earth validation scorer
================================
Reads the interpreter-labelled CSV produced after Google Earth review and
computes a full accuracy assessment of the fused flood confidence map.

Expected input
--------------
outputs/validation/ge_validation_points_labelled.csv

  Required columns:
    point_id          – matches the KML placemark names
    map_class         – 0-4 as assigned by the confidence raster
    map_class_name    – text label
    interpreter_label – 0 (not flooded) or 1 (flooded), entered by interpreter

  Optional columns:
    notes             – free-text interpreter notes (ignored in scoring)

Map class to flood label mapping used for scoring
--------------------------------------------------
  Classes 1, 2, 3  → predicted flood (1)
  Classes 0, 4     → predicted not-flooded (0)
  (Class 4 permanent water is treated as not-flooded here because it is a
   pre-existing water body, not event inundation)

Outputs
-------
  outputs/validation/ge_validation_report.txt   – full text report
  outputs/validation/ge_validation_confusion.csv – confusion matrix
  outputs/tables/table_ge_validation.md          – manuscript-ready markdown table

Usage
-----
    python -m src.validation.score_ge_validation
    python -m src.validation.score_ge_validation --labelled path/to/labelled.csv
"""

from __future__ import annotations
import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from src.utils.paths import get_project_root, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "Not flooded",
    1: "SAR only",
    2: "Optical only",
    3: "SAR + optical",
    4: "Permanent water",
}

# Map confidence classes to binary prediction
def to_binary(cls: int) -> int:
    return 1 if cls in (1, 2, 3) else 0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n  = tp + tn + fp + fn

    accuracy  = safe_div(tp + tn, n)
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1        = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)

    p_o = accuracy
    p_e = (safe_div(tp + fp, n) * safe_div(tp + fn, n) +
           safe_div(tn + fn, n) * safe_div(tn + fp, n))
    kappa = safe_div(p_o - p_e, 1 - p_e)

    return dict(tp=tp, tn=tn, fp=fp, fn=fn, n=n,
                accuracy=accuracy, precision=precision,
                recall=recall, f1=f1, specificity=specificity, kappa=kappa)


# ---------------------------------------------------------------------------
# Per-class breakdown
# ---------------------------------------------------------------------------

def per_class_accuracy(rows: list[dict]) -> list[dict]:
    results = []
    for cls_id, cls_name in CLASS_NAMES.items():
        subset = [r for r in rows if r["map_class"] == cls_id]
        if not subset:
            continue
        n_total  = len(subset)
        pred_bin = to_binary(cls_id)
        correct  = sum(1 for r in subset if r["interpreter_label"] == pred_bin)
        results.append({
            "cls_id":    cls_id,
            "cls_name":  cls_name,
            "n":         n_total,
            "correct":   correct,
            "accuracy":  safe_div(correct, n_total),
        })
    return results


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_report(metrics: dict, per_cls: list[dict],
                 rows: list[dict], path: Path) -> None:
    lines = [
        "=" * 60,
        "GOOGLE EARTH INDEPENDENT VALIDATION REPORT",
        "Akosombo 2023 Fused Flood Confidence Map",
        "=" * 60,
        "",
        f"Total validation points:  {metrics['n']}",
        f"  Interpreter-flooded:    {sum(r['interpreter_label'] for r in rows)}",
        f"  Interpreter-not-flooded:{metrics['n'] - sum(r['interpreter_label'] for r in rows)}",
        "",
        "BINARY FLOOD / NOT-FLOODED ACCURACY",
        "-" * 40,
        f"  Overall accuracy : {metrics['accuracy']:.4f}",
        f"  Precision        : {metrics['precision']:.4f}",
        f"  Recall           : {metrics['recall']:.4f}",
        f"  F1 score         : {metrics['f1']:.4f}",
        f"  Specificity      : {metrics['specificity']:.4f}",
        f"  Cohen kappa      : {metrics['kappa']:.4f}",
        "",
        "CONFUSION MATRIX (rows = interpreter, cols = map prediction)",
        "-" * 40,
        f"                     Pred NOT-FLOODED   Pred FLOODED",
        f"  Interp NOT-FLOODED      {metrics['tn']:>6}            {metrics['fp']:>6}",
        f"  Interp FLOODED          {metrics['fn']:>6}            {metrics['tp']:>6}",
        "",
        "PER MAP-CLASS ACCURACY",
        "-" * 40,
    ]
    for pc in per_cls:
        acc_str = f"{pc['accuracy']:.3f}" if not np.isnan(pc["accuracy"]) else "n/a"
        lines.append(
            f"  Class {pc['cls_id']} – {pc['cls_name']:<20} "
            f"n={pc['n']:>3}   correct={pc['correct']:>3}   accuracy={acc_str}"
        )
    lines += [
        "",
        "INTERPRETATION NOTES",
        "-" * 40,
        "  Precision = fraction of map-flooded pixels that are truly flooded",
        "  Recall    = fraction of truly flooded pixels captured by the map",
        "  F1        = harmonic mean of precision and recall",
        "  Kappa     = agreement above chance (> 0.80 = strong; > 0.60 = moderate)",
        "=" * 60,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_confusion_csv(metrics: dict, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "Predicted NOT-FLOODED", "Predicted FLOODED", "Row total"])
        w.writerow(["Interpreter NOT-FLOODED",
                    metrics["tn"], metrics["fp"], metrics["tn"] + metrics["fp"]])
        w.writerow(["Interpreter FLOODED",
                    metrics["fn"], metrics["tp"], metrics["fn"] + metrics["tp"]])
        w.writerow(["Column total",
                    metrics["tn"] + metrics["fn"],
                    metrics["fp"] + metrics["tp"],
                    metrics["n"]])


def write_markdown_table(metrics: dict, per_cls: list[dict],
                         n_total: int, path: Path) -> None:
    lines = [
        "**Table S1.** Independent validation of the fused flood confidence map "
        f"against manually interpreted Google Earth historical imagery "
        f"(September to November 2023). "
        f"n = {n_total} stratified random points drawn from the five confidence classes. "
        "Binary flood prediction assigns Classes 1, 2, and 3 as flooded and "
        "Classes 0 and 4 as not-flooded. "
        "Kappa: Cohen's kappa coefficient.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Overall accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 score | {metrics['f1']:.4f} |",
        f"| Cohen kappa | {metrics['kappa']:.4f} |",
        f"| True positives | {metrics['tp']} |",
        f"| True negatives | {metrics['tn']} |",
        f"| False positives | {metrics['fp']} |",
        f"| False negatives | {metrics['fn']} |",
        "",
        "**Per-class accuracy**",
        "",
        "| Map class | Class name | n | Correct | Class accuracy |",
        "|---|---|---|---|---|",
    ]
    for pc in per_cls:
        acc_str = f"{pc['accuracy']:.3f}" if not np.isnan(pc["accuracy"]) else "n/a"
        lines.append(
            f"| {pc['cls_id']} | {pc['cls_name']} "
            f"| {pc['n']} | {pc['correct']} | {acc_str} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(labelled_csv: str | None = None) -> None:
    proj_root = get_project_root()
    val_dir   = proj_root / "outputs/validation"
    ensure_dir(val_dir)
    ensure_dir(proj_root / "outputs/tables")

    csv_path = Path(labelled_csv) if labelled_csv else val_dir / "ge_validation_points_labelled.csv"
    if not csv_path.exists():
        log.error("Labelled CSV not found: %s", csv_path)
        log.error("Run generate_ge_validation_points.py first, label in Google Earth,")
        log.error("then save with an 'interpreter_label' column (0 or 1).")
        return

    rows: list[dict] = []
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            label = rec.get("interpreter_label", "").strip()
            if label not in ("0", "1"):
                skipped += 1
                continue
            rows.append({
                "point_id":         rec["point_id"],
                "map_class":        int(rec["map_class"]),
                "interpreter_label": int(label),
            })

    if skipped:
        log.warning("Skipped %d rows with missing or invalid interpreter_label", skipped)

    if not rows:
        log.error("No valid labelled rows found in %s", csv_path)
        return

    log.info("Loaded %d labelled points", len(rows))

    y_true = [r["interpreter_label"] for r in rows]
    y_pred = [to_binary(r["map_class"]) for r in rows]

    metrics  = compute_metrics(y_true, y_pred)
    per_cls  = per_class_accuracy(rows)

    report_path  = val_dir / "ge_validation_report.txt"
    conf_path    = val_dir / "ge_validation_confusion.csv"
    md_path      = proj_root / "outputs/tables/table_ge_validation.md"

    write_report(metrics, per_cls, rows, report_path)
    write_confusion_csv(metrics, conf_path)
    write_markdown_table(metrics, per_cls, len(rows), md_path)

    log.info("Report:           %s", report_path)
    log.info("Confusion matrix: %s", conf_path)
    log.info("Manuscript table: %s", md_path)
    log.info("")
    log.info("Overall accuracy %.4f   F1 %.4f   Kappa %.4f",
             metrics["accuracy"], metrics["f1"], metrics["kappa"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score Google Earth validation labels against flood confidence map"
    )
    parser.add_argument("--labelled", default=None,
                        help="Path to labelled CSV (default: outputs/validation/ge_validation_points_labelled.csv)")
    args = parser.parse_args()
    main(args.labelled)
