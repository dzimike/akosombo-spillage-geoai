from __future__ import annotations
"""
Supplementary U-Net statistics for the deep-learning paper.

* patch counts per partition (flood-positive 256 px patches, >= 64 flood pixels)
* pooled flood prevalence of the test patches (valid pixels = not permanent water)
  and the F1 of a trivial all-flood classifier on them
* full-scene flood extent of every model at threshold 0.5 outside permanent water
  (compared with the 72.2 km2 of the fused map)

Output: outputs/tables/unet_extra_stats.csv
"""
import logging

import numpy as np
import pandas as pd
import rasterio

from src.models.train_unet import FloodPatchDataset, get_spatial_splits
from src.utils.paths import get_project_root, get_output_path
from src.validation.compare_models_on_test_blocks import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    root = get_project_root()
    feat = root / "data/processed/features/flood_model_feature_stack.tif"
    label = root / "data/processed/flood_extent/fused_flood_confidence.tif"
    tr, va, te = get_spatial_splits(root / "data/processed/features/model_training_samples.parquet")
    rows = []
    for nm, bs in (("train", tr), ("validation", va), ("test", te)):
        ds = FloodPatchDataset(feat, label, bs, list(range(28)), None, augment=False)
        pos = val = 0
        c1 = c2 = c3 = 0
        with rasterio.open(label) as ls:
            for r0, c0 in ds.patches:
                w = rasterio.windows.Window(c0, r0, 256, 256)
                l = ls.read(1, window=w)
                v = l != 4
                pos += int(((l >= 1) & (l <= 3) & v).sum()); val += int(v.sum())
                c1 += int((l == 1).sum()); c2 += int((l == 2).sum()); c3 += int((l == 3).sum())
        p = pos / val
        rows.append({"item": f"patches_{nm}", "n_patches": len(ds), "flood_px": pos, "valid_px": val, "flood_prevalence": p,
                     "all_flood_f1": 2 * p / (1 + p), "class1_px": c1, "class2_px": c2, "class3_px": c3})
        log.info("%s", rows[-1])
    with rasterio.open(label) as ls:
        lbl = ls.read(1)
    perm = lbl == 4
    xgb_valid = None
    for name, rel in MODELS.items():
        path = root / rel
        if not path.exists():
            continue
        with rasterio.open(path) as ps:
            prob = ps.read(1)
        if name.startswith("XGBoost"):
            xgb_valid = np.isfinite(prob)
        m = (np.nan_to_num(prob, nan=0.0) >= 0.5) & ~perm
        rows.append({"item": f"scene_extent_{name}", "flood_km2_thr0.5": m.sum() / 1e4,
                     "flood_km2_in_xgb_scored_domain": (m & xgb_valid).sum() / 1e4,
                     "reference_km2": np.isin(lbl, [1, 2, 3]).sum() / 1e4})
        log.info("%s", rows[-1])
    pd.DataFrame(rows).to_csv(get_output_path("tables/unet_extra_stats.csv"), index=False)


if __name__ == "__main__":
    main()
