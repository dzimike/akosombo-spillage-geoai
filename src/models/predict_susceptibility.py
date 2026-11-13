from __future__ import annotations
"""
Phase 7 – Task 7.4: Full-raster Flood Susceptibility Prediction

Applies a trained model bundle to the 28-band feature stack in 500-row tiles
(~140 MB per tile) to stay well within available RAM.

Band→feature alignment is done by matching band tag names to the
feature_cols list stored in the model bundle — order is guaranteed.

For imputed features (pre-event S2, SAR) NaN pixels are filled with
training-set medians stored in the bundle. For core features (terrain,
event S2, settlement) NaN pixels receive NaN in the output (no-data).

Usage
-----
    python -m src.models.predict_susceptibility
    python -m src.models.predict_susceptibility --model outputs/models/rf_flood_model.joblib

Outputs
-------
outputs/maps/flood_susceptibility_<model_tag>.tif  — float32 probability [0–1]
outputs/maps/flood_susceptibility_<model_tag>.png  — overview PNG map
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from src.utils.paths import get_project_root, get_output_path, get_processed_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TILE_ROWS = 500   # rows per processing tile (~140 MB for 28 bands × 6978 cols × float32)


def predict_tiled(model_bundle: dict, stack_path: Path, output_path: Path,
                  band_overrides: dict[str, Path] | None = None) -> None:
    """Predict flood probability across the full feature stack using tiled I/O.

    band_overrides maps feature name -> single-band raster on the stack grid that
    replaces that band of the stack (used for the void-filled terrain run).
    """
    import joblib

    model          = model_bundle["model"]
    feature_cols   = model_bundle["feature_cols"]
    impute_medians = model_bundle.get("impute_medians", {})
    impute_pfx     = model_bundle.get("impute_prefixes", ())

    # Read band name → 1-based band index from TIF tags
    with rasterio.open(stack_path) as src:
        profile   = src.profile.copy()
        H, W      = src.height, src.width
        band_tags = {src.tags(i).get("name"): i for i in range(1, src.count + 1)}

    # Build ordered list of 1-based band indices matching feature_cols
    try:
        band_indices = [band_tags[col] for col in feature_cols]
    except KeyError as e:
        raise RuntimeError(
            f"Feature {e} not found in feature stack bands. "
            f"Available: {list(band_tags.keys())}"
        )

    n_feat = len(feature_cols)
    log.info("Predicting on %d × %d raster with %d features…", H, W, n_feat)
    log.info("Tile size: %d rows × %d cols", TILE_ROWS, W)

    profile.update(count=1, dtype="float32", nodata=np.nan,
                   compress="lzw", bigtiff="IF_SAFER")
    ensure_dir(output_path.parent)

    with rasterio.open(output_path, "w", **profile) as dst:
        n_tiles = (H + TILE_ROWS - 1) // TILE_ROWS
        for t_idx, row_off in enumerate(range(0, H, TILE_ROWS)):
            row_count = min(TILE_ROWS, H - row_off)
            window    = Window(0, row_off, W, row_count)

            with rasterio.open(stack_path) as src:
                # Read only the required bands for this tile: (n_feat, row_count, W)
                tile = np.stack(
                    [src.read(b, window=window).astype(np.float32)
                     for b in band_indices], axis=0
                )
            if band_overrides:
                for j, col in enumerate(feature_cols):
                    if col in band_overrides:
                        with rasterio.open(band_overrides[col]) as osrc:
                            tile[j] = osrc.read(1, window=window).astype(np.float32)

            N = row_count * W
            X = tile.reshape(n_feat, N).T   # (N, n_feat)

            # Impute pre-event S2 and SAR NaN with training medians
            for j, col in enumerate(feature_cols):
                if any(col.startswith(p) for p in impute_pfx):
                    nan_mask = ~np.isfinite(X[:, j])
                    if nan_mask.any():
                        X[nan_mask, j] = impute_medians.get(col, 0.0)

            # Core feature validity — predict only where all core features are finite
            core_valid = np.ones(N, dtype=bool)
            for j, col in enumerate(feature_cols):
                if not any(col.startswith(p) for p in impute_pfx):
                    core_valid &= np.isfinite(X[:, j])

            proba_flat = np.full(N, np.nan, dtype=np.float32)
            if core_valid.any():
                proba_flat[core_valid] = model.predict_proba(
                    X[core_valid])[:, 1].astype(np.float32)

            dst.write(proba_flat.reshape(row_count, W), 1, window=window)

            if (t_idx + 1) % 4 == 0 or t_idx == n_tiles - 1:
                log.info("  Tile %d/%d (row %d–%d)  valid=%.1f%%",
                         t_idx + 1, n_tiles, row_off, row_off + row_count - 1,
                         100 * float(core_valid.mean()))

    log.info("Susceptibility raster written: %s", output_path)


def save_overview_png(tif_path: Path, png_path: Path) -> None:
    """Quick overview PNG of the susceptibility raster."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    with rasterio.open(tif_path) as src:
        arr = src.read(1)

    cmap = plt.cm.RdYlGn_r
    cmap.set_bad("lightgrey", alpha=0.4)

    fig, ax = plt.subplots(figsize=(10, 12))
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Flood Susceptibility Probability")
    ax.set_title("Akosombo Spillage — Flood Susceptibility Map", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Overview PNG: %s", png_path)


def main(config_path: str | None = None,
         model_path:  str | None = None) -> None:
    import joblib

    proj_root = get_project_root()

    # Default to the XGBoost model; fall back to RF
    if model_path is None:
        for candidate in ["xgb_flood_model.joblib", "rf_flood_model.joblib"]:
            p = proj_root / "outputs" / "models" / candidate
            if p.exists():
                model_path = str(p)
                break

    if model_path is None:
        log.error("No trained model found in outputs/models/. Train a model first.")
        return

    model_tag  = Path(model_path).stem.replace("_flood_model", "")
    stack_path = get_processed_path("features/flood_model_feature_stack.tif")
    output_tif = get_output_path(f"maps/flood_susceptibility_{model_tag}.tif")
    output_png = get_output_path(f"maps/flood_susceptibility_{model_tag}.png")

    if not stack_path.exists():
        log.error("Feature stack not found: %s", stack_path); return

    log.info("Model:  %s", model_path)
    log.info("Stack:  %s", stack_path)
    log.info("Output: %s", output_tif)

    model_bundle = joblib.load(model_path)

    with Timer() as t:
        predict_tiled(model_bundle, stack_path, output_tif)
        save_overview_png(output_tif, output_png)

    log.info("Prediction complete in %.1f s", t.elapsed)
    log_run(
        script=__file__,
        input_paths=[model_path, str(stack_path)],
        output_paths=[str(output_tif)],
        parameters={"model_tag": model_tag, "tile_rows": TILE_ROWS},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict flood susceptibility")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None,
                        help="Path to .joblib model bundle (default: best available)")
    args = parser.parse_args()
    main(args.config, args.model)
