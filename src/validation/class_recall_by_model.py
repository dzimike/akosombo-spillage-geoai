from __future__ import annotations
"""
Class-wise detection of the fused flood classes by each model on the test blocks.

For every model (XGBoost and the four U-Net experiments) the share of test-block
pixels of each flood class that the model labels as flood at threshold 0.5:

  class 1  SAR-only flood      (backscatter water, optical index below threshold)
  class 2  optical-only flood  (NDWI/MNDWI water, backscatter above threshold)
  class 3  SAR + optical agreement
and the false-positive rate on class 0 (not flooded, permanent water excluded).

This is the direct test of what each input stream contributes: an optical-only model
should find classes 2 and 3 and miss class 1, a SAR-only model the reverse. Pixels are
those of the test blocks where the XGBoost model has a score and the fused label is not
permanent water or no-data.

Output: outputs/tables/model_class_recall_test_blocks.csv
"""
import logging

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.models.train_unet import get_spatial_splits, BLOCK_PX
from src.utils.paths import get_project_root, get_output_path
from src.validation.compare_models_on_test_blocks import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(thr: float = 0.5) -> None:
    root = get_project_root()
    _, _, test_blocks = get_spatial_splits(root / "data/processed/features/model_training_samples.parquet")
    paths = {n: root / r for n, r in MODELS.items() if (root / r).exists()}
    srcs = {n: rasterio.open(p) for n, p in paths.items()}
    cnt = {n: {c: [0, 0] for c in (0, 1, 2, 3)} for n in paths}   # [predicted flood, total]
    with rasterio.open(root / "data/processed/flood_extent/fused_flood_confidence.tif") as ls:
        for b in sorted(test_blocks):
            r, c = map(int, b.split("_"))
            win = Window(c * BLOCK_PX, r * BLOCK_PX, min(BLOCK_PX, ls.width - c * BLOCK_PX), min(BLOCK_PX, ls.height - r * BLOCK_PX))
            lbl = ls.read(1, window=win)
            probs = {n: s.read(1, window=win).astype(np.float32) for n, s in srcs.items()}
            valid = (lbl != 9) & (lbl != 4) & np.isfinite(probs["XGBoost (raw score)"])
            for n, p in probs.items():
                pred = np.nan_to_num(p, nan=0.0) >= thr
                for k in (0, 1, 2, 3):
                    m = valid & (lbl == k)
                    cnt[n][k][0] += int((pred & m).sum()); cnt[n][k][1] += int(m.sum())
    rows = []
    for n, d in cnt.items():
        rows.append({"model": n, "recall_class1_sar_only": d[1][0] / max(d[1][1], 1),
                     "recall_class2_optical_only": d[2][0] / max(d[2][1], 1),
                     "recall_class3_agreement": d[3][0] / max(d[3][1], 1),
                     "false_positive_rate_class0": d[0][0] / max(d[0][1], 1),
                     "n_class1_px": d[1][1], "n_class2_px": d[2][1], "n_class3_px": d[3][1]})
        log.info("%s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    pd.DataFrame(rows).to_csv(get_output_path("tables/model_class_recall_test_blocks.csv"), index=False)


if __name__ == "__main__":
    main()
