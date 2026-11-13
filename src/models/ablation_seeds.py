from __future__ import annotations
"""
Seed sensitivity of the predictor-group ablations (XGBoost).

The pairwise ablation (ablation_pairwise.py) uses one spatial split and one seed, so differences of a few hundredths in test F1 cannot be
told from noise. This script repeats the full model, the six single-group removals, the 15 pair removals, the six group-alone models and the three band ablations of ablation_label_bands.py under ten seeds. A seed
changes both the assignment of 10 km blocks to train, validation and test and the XGBoost random state; everything else is as in the main run.

Outputs: outputs/tables/xgb_ablation_seeds.csv (one row per seed and variant), outputs/tables/xgb_ablation_seeds_summary.csv
"""
import itertools
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, roc_auc_score

from src.models.ablation_label_bands import VARIANTS as LABEL_VARIANTS, fit
from src.models.ablation_pairwise import GROUPS
from src.models.train_random_forest import NON_FEATURE_COLS, add_spatial_blocks, spatial_train_test_split
from src.utils.paths import get_output_path, get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def main() -> None:
    root = get_project_root()
    cfg = yaml.safe_load(open(root / "config/model_config.yaml"))["models"]["xgboost"]
    df = add_spatial_blocks(pd.read_parquet(root / "data/processed/features/model_training_samples.parquet"), 1000)
    all_feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    names = list(GROUPS)
    variants = [("full", (), [])] + [(f"drop:{g}", (g,), []) for g in names]
    variants += [(f"drop:{a}+{b}", (a, b), []) for a, b in itertools.combinations(names, 2)]
    variants += [(f"only:{g}", tuple(x for x in names if x != g), []) for g in names]
    variants += [(f"bands:{k}", (), v) for k, v in LABEL_VARIANTS.items() if v]
    rows = []
    for seed in SEEDS:
        tr, va, te = spatial_train_test_split(df, 0.70, 0.15, seed)
        y = te.flood_label.values
        for name, dropped, extra in variants:
            feats = [c for c in all_feats if c not in sum((GROUPS[g] for g in dropped), []) and c not in extra]
            m = fit(tr[feats].values, tr.flood_label.values, va[feats].values, va.flood_label.values, cfg, seed)
            p = m.predict_proba(te[feats].values)[:, 1]
            rows.append({"seed": seed, "variant": name, "n_test_blocks": te.block_id.nunique(), "test_flood_share": float(y.mean()),
                         "test_f1": f1_score(y, (p >= 0.5).astype(int)), "test_roc_auc": roc_auc_score(y, p)})
            log.info("seed %2d %-22s F1 %.4f AUC %.4f", seed, name, rows[-1]["test_f1"], rows[-1]["test_roc_auc"])
        pd.DataFrame(rows).to_csv(get_output_path("tables/xgb_ablation_seeds.csv"), index=False)
    d = pd.DataFrame(rows)
    full = d[d.variant == "full"].set_index("seed").test_f1
    d["delta_vs_full"] = d.test_f1 - d.seed.map(full)
    s = d.groupby("variant").agg(f1_mean=("test_f1", "mean"), f1_sd=("test_f1", "std"), f1_min=("test_f1", "min"), f1_max=("test_f1", "max"),
                                 delta_mean=("delta_vs_full", "mean"), delta_sd=("delta_vs_full", "std"),
                                 seeds_lower_than_full=("delta_vs_full", lambda x: int((x < -0.005).sum()))).reset_index()
    s.to_csv(get_output_path("tables/xgb_ablation_seeds_summary.csv"), index=False)
    log.info("\n%s", s.round(3).to_string())


if __name__ == "__main__":
    main()
