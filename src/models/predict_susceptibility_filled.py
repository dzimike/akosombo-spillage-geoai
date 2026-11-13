from __future__ import annotations
"""
SUPERSEDED (kept for provenance): see build_gapfree_dem.py and the retrained
model; use predict_susceptibility.py.

Full-domain XGBoost probability raster using void-filled terrain predictors.

Identical to predict_susceptibility.py except the eight terrain bands are read
from data/processed/features_filled/terrain (see
src/preprocessing/build_filled_terrain.py). All other bands, the trained model
bundle, and the imputation rules are unchanged; no retraining.

Output: outputs/maps/flood_susceptibility_xgb_filled.tif
"""
import logging

import joblib

from src.models.predict_susceptibility import predict_tiled
from src.utils.logging_utils import Timer, log_run
from src.utils.paths import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TERRAIN = ["elevation", "slope", "aspect_sin", "aspect_cos", "twi",
           "flow_accumulation_log", "relative_elevation", "hand_proxy"]


def main() -> None:
    root = get_project_root()
    model_path = root / "outputs/models/xgb_flood_model.joblib"
    stack = root / "data/processed/features/flood_model_feature_stack.tif"
    out = root / "outputs/maps/flood_susceptibility_xgb_filled.tif"
    overrides = {n: root / f"data/processed/features_filled/terrain/{n}.tif" for n in TERRAIN}
    with Timer() as t:
        predict_tiled(joblib.load(model_path), stack, out, band_overrides=overrides)
    log_run(script=__file__, input_paths=[str(model_path), str(stack)] + [str(p) for p in overrides.values()],
            output_paths=[str(out)], parameters={"band_overrides": TERRAIN},
            status="success", runtime_seconds=t.elapsed)


if __name__ == "__main__":
    main()
