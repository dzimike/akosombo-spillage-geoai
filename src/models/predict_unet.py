from __future__ import annotations
"""
Deep-learning paper - U-Net full-scene flood probability prediction

Applies a trained U-Net bundle to the complete 28-band feature stack using
an overlapping-tile strategy with cosine blending at tile boundaries.  This
eliminates the hard-edge artefacts that arise when tiles are stitched without
weight blending.

Tile geometry
-------------
  tile_size  : 256 px (must match the training PATCH_SIZE)
  stride     : 128 px  (50 % overlap on every side)
  blend      : 2-D outer product of a cosine window, giving maximum weight at
               tile centres and zero weight at tile edges

For each experiment bundle the script writes:
  outputs/maps/unet_<name>_flood_prob.tif   float32 probability [0, 1]
  outputs/maps/unet_<name>_flood_mask.tif   uint8 binary mask (thr=0.5)
  outputs/maps/unet_<name>_flood_prob.png   overview PNG for quick QC

Usage
-----
    python -m src.models.predict_unet
    python -m src.models.predict_unet --bundle outputs/models/unet_full.pt
    python -m src.models.predict_unet --threshold 0.4 --bundle outputs/models/unet_full.pt
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds

from src.models.unet_model import UNet
from src.utils.paths import (get_project_root, get_output_path,
                              get_processed_path, ensure_dir)
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TILE_SIZE    = 256
STRIDE       = 128          # 50 % overlap
DEFAULT_THR  = 0.5
READ_BANDS   = 500          # rows of source raster read into RAM at once


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        log.info("Device: Apple MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        log.info("Device: CUDA")
        return torch.device("cuda")
    log.info("Device: CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Blending window
# ---------------------------------------------------------------------------

def _cosine_window(size: int) -> np.ndarray:
    """Return a (size, size) cosine-tapered weight window, peak = 1.0."""
    axis = np.hanning(size).astype(np.float32)
    axis /= axis.max()
    return np.outer(axis, axis)


# ---------------------------------------------------------------------------
# Core prediction routine
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_scene(bundle_path: Path,
                  feat_path: Path,
                  out_prob_path: Path,
                  out_mask_path: Path,
                  threshold: float = DEFAULT_THR,
                  device: torch.device | None = None) -> None:
    """
    Tile-predict the full scene and write probability + binary-mask GeoTIFFs.

    Parameters
    ----------
    bundle_path   : Path to .pt model bundle saved by train_unet.py
    feat_path     : Path to the 28-band feature-stack GeoTIFF
    out_prob_path : Destination float32 probability raster
    out_mask_path : Destination uint8 binary flood mask raster
    threshold     : Decision threshold applied to probabilities
    device        : Torch device (auto-selected if None)
    """
    if device is None:
        device = get_device()

    # ---- Load bundle -------------------------------------------------------
    log.info("Loading bundle: %s", bundle_path)
    bundle      = torch.load(bundle_path, map_location="cpu", weights_only=False)
    band_idx    = bundle["band_indices"]             # list[int], 0-based
    stats_mean  = bundle["stats_mean"]               # (28,) float32
    stats_std   = bundle["stats_std"]                # (28,) float32
    patch_size  = bundle.get("patch_size", TILE_SIZE)
    n_bands     = len(band_idx)
    exp_name    = bundle.get("experiment", bundle_path.stem)

    base_filters = bundle.get("base_filters", 16)
    model = UNet(in_channels=n_bands, base_filters=base_filters)
    model.load_state_dict(bundle["model_state"])
    model.eval().to(device)
    log.info("  Experiment: %s   bands: %d   patch_size: %d   base_filters: %d",
             exp_name, n_bands, patch_size, base_filters)
    log.info("  Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    stride      = patch_size // 2
    blend_win   = _cosine_window(patch_size)          # (P, P)

    # Per-band normalisation arrays for selected bands
    mean_sel = stats_mean[band_idx].reshape(n_bands, 1, 1).astype(np.float32)
    std_sel  = stats_std[band_idx].reshape(n_bands, 1, 1).astype(np.float32)

    # ---- Open source raster ------------------------------------------------
    with rasterio.open(feat_path) as src:
        H, W      = src.height, src.width
        profile   = src.profile.copy()
        nodata_in = src.nodata

    log.info("  Scene size: %d x %d px", H, W)

    # Accumulation buffers (float64 to avoid precision loss during summation)
    prob_acc    = np.zeros((H, W), dtype=np.float64)
    weight_acc  = np.zeros((H, W), dtype=np.float64)

    # Row/col start positions for overlapping tiles
    row_starts = list(range(0, H - patch_size + 1, stride))
    col_starts = list(range(0, W - patch_size + 1, stride))

    # Ensure last column and row are fully covered
    if row_starts[-1] + patch_size < H:
        row_starts.append(H - patch_size)
    if col_starts[-1] + patch_size < W:
        col_starts.append(W - patch_size)

    n_tiles     = len(row_starts) * len(col_starts)
    log.info("  Total tiles: %d  (stride=%d)", n_tiles, stride)

    processed   = 0
    with rasterio.open(feat_path) as src:
        for r0 in row_starts:
            for c0 in col_starts:
                win  = Window(c0, r0, patch_size, patch_size)
                feat = src.read(window=win).astype(np.float32)   # (28, P, P)
                feat = feat[band_idx]                             # (n, P, P)

                # Handle nodata
                if nodata_in is not None:
                    feat = np.where(feat == nodata_in, np.nan, feat)

                # Normalise
                feat = (feat - mean_sel) / (std_sel + 1e-8)
                feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

                # Inference
                t_in  = torch.from_numpy(feat[None]).to(device)  # (1,n,P,P)
                prob  = torch.sigmoid(model(t_in))[0, 0].cpu().numpy()  # (P,P)

                # Weighted accumulation
                prob_acc[r0:r0 + patch_size, c0:c0 + patch_size]   += prob * blend_win
                weight_acc[r0:r0 + patch_size, c0:c0 + patch_size] += blend_win

                processed += 1
                if processed % 500 == 0:
                    log.info("    %d / %d tiles", processed, n_tiles)

    log.info("  All tiles processed; compositing…")

    # Avoid division by zero for pixels not covered by any tile
    valid        = weight_acc > 0
    prob_map     = np.where(valid, prob_acc / np.maximum(weight_acc, 1e-12), 0.0)
    prob_map     = np.clip(prob_map, 0.0, 1.0).astype(np.float32)

    binary_mask  = (prob_map >= threshold).astype(np.uint8)

    # ---- Write probability raster ------------------------------------------
    prob_profile = profile.copy()
    prob_profile.update(
        dtype="float32", count=1, nodata=None,
        compress="lzw", predictor=3,
    )
    out_prob_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_prob_path, "w", **prob_profile) as dst:
        dst.write(prob_map[None])
    log.info("  Probability map: %s", out_prob_path)

    # ---- Write binary mask raster ------------------------------------------
    mask_profile = profile.copy()
    mask_profile.update(
        dtype="uint8", count=1, nodata=255,
        compress="lzw", predictor=2,
    )
    with rasterio.open(out_mask_path, "w", **mask_profile) as dst:
        dst.write(binary_mask[None])
    log.info("  Binary mask:     %s", out_mask_path)

    # ---- Quick-look PNG ----------------------------------------------------
    _save_png(prob_map, out_prob_path.with_suffix(".png"))

    # Summary stats
    flooded_px  = int(binary_mask.sum())
    res_m       = abs(profile.get("transform", [10])[0])   # pixel size in metres
    flood_km2   = flooded_px * (res_m ** 2) / 1e6
    log.info("  Flood extent: %.1f km²  (%.1f %% of scene)",
             flood_km2, 100.0 * flooded_px / (H * W))


def _save_png(prob_map: np.ndarray, png_path: Path) -> None:
    """Write a coloured quick-look PNG of the probability map."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(8, 8 * prob_map.shape[0] / prob_map.shape[1]))
        im = ax.imshow(prob_map, cmap="RdYlBu_r", vmin=0, vmax=1,
                       interpolation="none")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Flood probability")
        ax.set_title("U-Net flood probability", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("  PNG overview:    %s", png_path)
    except Exception as exc:
        log.warning("  PNG skipped: %s", exc)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main(bundle_arg: str | None = None,
         threshold: float = DEFAULT_THR,
         config_path: str | None = None) -> None:
    proj_root = get_project_root()
    maps_dir  = get_output_path("maps", None)
    ensure_dir(maps_dir)

    feat_path  = proj_root / "data/processed/features/flood_model_feature_stack.tif"
    if not feat_path.exists():
        log.error("Feature stack not found: %s", feat_path)
        return

    device = get_device()

    if bundle_arg:
        # Single bundle specified on command line
        bundles = [Path(bundle_arg)]
    else:
        # Auto-discover all unet_*.pt bundles
        models_dir = get_output_path("models", None)
        bundles    = sorted(models_dir.glob("unet_*.pt"))
        if not bundles:
            log.error("No unet_*.pt bundles found in %s", models_dir)
            return

    out_paths = []
    with Timer() as t:
        for bp in bundles:
            exp_name      = bp.stem.replace("unet_", "")
            out_prob      = maps_dir / f"unet_{exp_name}_flood_prob.tif"
            out_mask      = maps_dir / f"unet_{exp_name}_flood_mask.tif"
            log.info("=" * 60)
            log.info("Predicting: %s", exp_name)
            log.info("=" * 60)
            predict_scene(bp, feat_path, out_prob, out_mask,
                          threshold=threshold, device=device)
            out_paths += [str(out_prob), str(out_mask)]

    log_run(
        script=__file__,
        input_paths=[str(feat_path)],
        output_paths=out_paths,
        parameters={"threshold": threshold, "n_bundles": len(bundles)},
        status="success",
        runtime_seconds=t.elapsed,
    )
    log.info("Prediction complete in %.1f s", t.elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full-scene U-Net flood probability prediction")
    parser.add_argument("--bundle",    default=None,
                        help="Path to a specific .pt bundle "
                             "(default: all unet_*.pt in outputs/models/)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THR,
                        help="Binary classification threshold (default 0.5)")
    parser.add_argument("--config",    default=None)
    args = parser.parse_args()
    main(args.bundle, args.threshold, args.config)
