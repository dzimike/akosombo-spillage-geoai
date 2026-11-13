from __future__ import annotations
"""
Prior-shift correction of the XGBoost flood probability.

The classifier is trained on a case-control sample: all flood samples up to 50,000 plus 100,000 non-flood
samples (29.5 % flood in the training partition), and the positive class is up-weighted by scale_pos_weight =
negatives/positives. The effective training prior is therefore w*pi_t / (w*pi_t + 1 - pi_t) (= 0.5 when w is the
negative:positive ratio), whereas flood is ~1.1 % of the pixels the model is applied to (fused flood /
(flood + not flooded), pre-existing water excluded). The raw output overstates the probability of flood by a
constant logit offset. With p the raw output, pi_s the effective training prior and pi the scene prevalence,

    p' = sigmoid( logit(p) + ln(pi / (1 - pi)) - ln(pi_s / (1 - pi_s)) )                 (Saerens et al., 2002)

p' is written to outputs/maps/flood_susceptibility_xgb_prior_corrected.tif and used for threshold sweeps on a
scale where 0.5 is the Bayes-optimal cut for equal misclassification costs at the scene prevalence.
"""
import logging

import numpy as np
import pandas as pd
import rasterio

from src.utils.paths import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    root = get_project_root()
    with rasterio.open(root / "data/processed/flood_extent/fused_flood_confidence.tif") as s:
        fu = s.read(1)
    nf = int(np.isin(fu, [1, 2, 3]).sum()); n0 = int((fu == 0).sum())
    pi = nf / (nf + n0)
    import joblib
    bundle = joblib.load(root / "outputs/models/xgb_flood_model.joblib")
    w = float(bundle["model"].get_params()["scale_pos_weight"])
    xm = pd.read_csv(root / "outputs/tables/xgb_metrics.csv").set_index("split")
    pi_t = float(xm.loc["train", "flood_pct"]) / 100
    pi_s = w * pi_t / (w * pi_t + 1 - pi_t)
    shift = np.log(pi / (1 - pi)) - np.log(pi_s / (1 - pi_s))
    log.info("scene prevalence %.4f, training prevalence %.4f x weight %.2f = effective prior %.3f, logit shift %.3f (raw p=%.4f maps to p'=0.5)", pi, pi_t, w, pi_s, shift,
             1 / (1 + np.exp(shift)))
    src_p = root / "outputs/maps/flood_susceptibility_xgb.tif"
    with rasterio.open(src_p) as s:
        prof = s.profile.copy(); p = s.read(1).astype(np.float32)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    q = (1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + shift)))).astype(np.float32)
    prof.update(dtype="float32", compress="lzw")
    out = root / "outputs/maps/flood_susceptibility_xgb_prior_corrected.tif"
    with rasterio.open(out, "w", **prof) as d:
        d.write(q, 1)
    log.info("written %s", out)


if __name__ == "__main__":
    main()
