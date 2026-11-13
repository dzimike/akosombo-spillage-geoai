from __future__ import annotations
"""
Pairwise feature-group ablation for the XGBoost flood classifier.

The 28 predictors are grouped into six groups:
  Terrain         elevation, slope, aspect (sin, cos), TWI, elevation-rank index, relative elevation, height proxy   (8)
  Proximity       distance to waterway, to the dam, to the Volta                                                      (3)
  Optical event   event NDVI, NDWI, MNDWI, NDBI                                                                       (4)
  Optical pre     pre-event NDVI, MNDWI, NDWI                                                                         (3)
  SAR             pre- and event VV, VV change, pre- and event VH                                                     (5)
  Settlement      rule-based land-cover class, building density, road density, distance to road, population density   (5)

Variants: full model; each group removed (6); each pair removed (15); each group alone (6). Same split, hyperparameters and
early stopping as ablation_label_bands.py. Stage 1 scores every variant on the test blocks at threshold 0.50; stage 2 adds
true leave-one-block-out (LOBO) mean ROC-AUC and F1 (full, single and pair removals), writing the table after each variant.

Output: outputs/tables/xgb_ablation_pairwise.csv
"""
import itertools
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.models.ablation_label_bands import fit
from src.models.train_random_forest import NON_FEATURE_COLS, add_spatial_blocks, spatial_train_test_split
from src.utils.paths import get_project_root, get_output_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GROUPS = {
    "Terrain": ["elevation", "slope", "aspect_sin", "aspect_cos", "twi", "flow_accumulation_log", "relative_elevation", "hand_proxy"],
    "Proximity": ["dist_to_waterway", "dist_to_dam", "dist_to_volta"],
    "Optical event": ["ndvi_event", "ndwi_event", "mndwi_event", "ndbi_event"],
    "Optical pre": ["ndvi_pre", "mndwi_pre", "ndwi_pre"],
    "SAR": ["sar_vv_pre", "sar_vv_event", "sar_vv_change", "sar_vh_pre", "sar_vh_event"],
    "Settlement": ["landcover_class", "building_density_500m", "road_density_1km", "dist_to_road", "population_density"],
}


def lobo(df, feats, n_trees, cfg, seed):
    aucs, f1s = [], []
    for b in sorted(df.block_id.unique()):
        blk = df[df.block_id == b]; yb = blk.flood_label.values
        if len(blk) < 10 or yb.sum() in (0, len(yb)):
            continue
        rest = df[df.block_id != b]; yr = rest.flood_label.values
        f = XGBClassifier(n_estimators=n_trees, learning_rate=cfg.get("learning_rate", 0.03), max_depth=cfg.get("max_depth", 6),
                          subsample=cfg.get("subsample", 0.8), colsample_bytree=cfg.get("colsample_bytree", 0.8),
                          scale_pos_weight=float((yr == 0).sum() / max((yr == 1).sum(), 1)), random_state=seed, n_jobs=-1)
        f.fit(rest[feats].values, yr)
        pb = f.predict_proba(blk[feats].values)[:, 1]
        aucs.append(roc_auc_score(yb, pb)); f1s.append(f1_score(yb, (pb >= 0.5).astype(int), zero_division=0))
    return float(np.mean(aucs)), float(np.mean(f1s)), len(aucs)


def main() -> None:
    root = get_project_root()
    cfg_all = yaml.safe_load(open(root / "config/model_config.yaml"))
    cfg, seed = cfg_all["models"]["xgboost"], cfg_all.get("random_seed", 42)
    df = add_spatial_blocks(pd.read_parquet(root / "data/processed/features/model_training_samples.parquet"), 1000)
    all_feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    assert sorted(sum(GROUPS.values(), [])) == sorted(all_feats), set(all_feats) ^ set(sum(GROUPS.values(), []))
    tr, va, te = spatial_train_test_split(df, 0.70, 0.15, seed)
    names = list(GROUPS)
    variants = [("full", "full", ())]
    variants += [(f"drop:{g}", "drop1", (g,)) for g in names]
    variants += [(f"drop:{a}+{b}", "drop2", (a, b)) for a, b in itertools.combinations(names, 2)]
    variants += [(f"only:{g}", "only", tuple(x for x in names if x != g)) for g in names]
    rows, models = [], {}
    for name, kind, dropped in variants:
        drop = sum((GROUPS[g] for g in dropped), [])
        feats = [c for c in all_feats if c not in drop]
        m = fit(tr[feats].values, tr.flood_label.values, va[feats].values, va.flood_label.values, cfg, seed)
        p = m.predict_proba(te[feats].values)[:, 1]; y = te.flood_label.values; yh = (p >= 0.5).astype(int)
        rows.append({"variant": name, "kind": kind, "dropped_groups": ";".join(dropped), "n_features": len(feats),
                     "n_trees": int(m.best_iteration) + 1, "test_roc_auc": roc_auc_score(y, p), "test_f1": f1_score(y, yh),
                     "test_precision": precision_score(y, yh), "test_recall": recall_score(y, yh),
                     "lobo_auc_mean": np.nan, "lobo_f1_mean": np.nan, "lobo_blocks": 0})
        models[name] = (feats, int(m.best_iteration) + 1)
        log.info("stage 1 %-34s n=%2d test AUC %.4f F1 %.4f", name, len(feats), rows[-1]["test_roc_auc"], rows[-1]["test_f1"])
    out = pd.DataFrame(rows)
    out.to_csv(get_output_path("tables/xgb_ablation_pairwise.csv"), index=False)
    for i, r in out.iterrows():
        if r.kind == "only":
            continue
        feats, n_trees = models[r.variant]
        a, f, nb = lobo(df, feats, n_trees, cfg, seed)
        out.loc[i, ["lobo_auc_mean", "lobo_f1_mean", "lobo_blocks"]] = [a, f, nb]
        out.to_csv(get_output_path("tables/xgb_ablation_pairwise.csv"), index=False)
        log.info("stage 2 %-34s LOBO AUC %.4f F1 %.4f (%d blocks)", r.variant, a, f, nb)


if __name__ == "__main__":
    main()
