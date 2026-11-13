from __future__ import annotations
"""
Phase 6 – Task 6.5: Feature Stack Assembly

Collects all upstream feature rasters, aligns them to the common 10 m
S2 grid, and writes:
  1. Multi-band GeoTIFF feature stack (one band per predictor)
  2. Stratified-sample Parquet training table (pixel-level)
  3. Feature metadata CSV

Training label is derived from the fused flood confidence raster:
  1 = flooded (confidence classes 1-3)
  0 = not flooded (class 0, excluding permanent water class 4)
Permanent water pixels (class 4) and no-data pixels are excluded.

Sampling strategy
-----------------
Flood pixels    → up to MAX_FLOOD_SAMPLES (stratified by confidence class)
Non-flood pixels → up to MAX_NOFLOOD_SAMPLES (random from valid pixels)

Outputs
-------
data/processed/features/flood_model_feature_stack.tif
data/processed/features/model_training_samples.parquet
data/processed/features/feature_metadata.csv
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
import pandas as pd

from src.utils.paths import load_config, get_processed_path, ensure_dir
from src.utils.logging_utils import Timer, log_run
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_FLOOD_SAMPLES   = 50_000
MAX_NOFLOOD_SAMPLES = 100_000
RANDOM_SEED         = 42


def collect_feature_paths(config) -> dict[str, Path]:
    """Return ordered dict of feature_name → file path."""
    feat = get_processed_path("features", config)
    sat  = get_processed_path("satellite", config)

    candidates = {
        # Terrain (Task 6.1)
        "elevation":              feat / "terrain/elevation.tif",
        "slope":                  feat / "terrain/slope.tif",
        "aspect_sin":             feat / "terrain/aspect_sin.tif",
        "aspect_cos":             feat / "terrain/aspect_cos.tif",
        "twi":                    feat / "terrain/twi.tif",
        "flow_accumulation_log":  feat / "terrain/flow_accumulation_log.tif",
        "relative_elevation":     feat / "terrain/relative_elevation.tif",
        "hand_proxy":             feat / "terrain/hand_proxy.tif",
        # Hydrological (Task 6.2)
        "dist_to_waterway":       feat / "hydro/dist_to_waterway.tif",
        "dist_to_dam":            feat / "hydro/dist_to_dam.tif",
        "dist_to_volta":          feat / "hydro/dist_to_volta.tif",
        # Landcover — event-period indices (96% valid, primary spectral features)
        "ndvi_event":             sat / "s2_ndvi_event.tif",
        "ndwi_event":             sat / "s2_ndwi_event.tif",
        "mndwi_event":            sat / "s2_mndwi_event.tif",
        "ndbi_event":             feat / "landcover/ndbi_event.tif",
        # Landcover — pre-event indices (47% valid; NaN-imputed, s2_pre_valid flag added)
        "ndvi_pre":               feat / "landcover/ndvi_pre.tif",
        "mndwi_pre":              feat / "landcover/mndwi_pre.tif",
        "ndwi_pre":               feat / "landcover/ndwi_pre.tif",
        # Land-cover class (100% valid — derived from available pixels)
        "landcover_class":        feat / "landcover/landcover_class.tif",
        # Settlement (Task 6.3)
        "building_density_500m":  feat / "settlement/building_density_500m.tif",
        "road_density_1km":       feat / "settlement/road_density_1km.tif",
        "dist_to_road":           feat / "settlement/dist_to_road.tif",
        "population_density":     feat / "settlement/population_density.tif",
        # SAR (preprocessed) — NaN-imputed, sar_data_valid flag added
        "sar_vv_pre":             sat / "s1_pre_event_vv.tif",
        "sar_vv_event":           sat / "s1_event_vv.tif",
        "sar_vv_change":          sat / "s1_vv_change.tif",
        "sar_vh_pre":             sat / "s1_pre_event_vh.tif",
        "sar_vh_event":           sat / "s1_event_vh.tif",
    }
    # Only include files that exist
    available = {k: v for k, v in candidates.items() if v.exists()}
    missing   = [k for k, v in candidates.items() if not v.exists()]
    if missing:
        log.warning("Features missing (will be skipped): %s", missing)
    return available


# Features whose NaN is due to CLOUD COVER or ORBIT GAPS (not invalid pixels).
# These are imputed with column median rather than causing row exclusion.
IMPUTE_COLS_PREFIXES = ("sar_", "ndvi_pre", "mndwi_pre", "ndwi_pre")


def main(config_path: str | None = None) -> None:
    config   = load_config(config_path)
    feat_dir = get_processed_path("features", config)
    ensure_dir(feat_dir)

    template_path = get_processed_path("satellite/s2_ndwi_event.tif", config)
    flood_path    = get_processed_path("flood_extent/fused_flood_confidence.tif", config)

    for p in [template_path, flood_path]:
        if not p.exists():
            log.error("Required file missing: %s", p); return

    with Timer() as t:
        # ---- Collect available features ----------------------------------
        feature_paths = collect_feature_paths(config)
        n_feat = len(feature_paths)
        log.info("Features available: %d", n_feat)
        for name, path in feature_paths.items():
            log.info("  %-30s  %s", name, path.name)

        # ---- Load template grid ------------------------------------------
        with rasterio.open(template_path) as src:
            profile   = src.profile.copy()
            height    = src.height
            width     = src.width
            transform = src.transform
            crs       = src.crs

        # ---- Load flood label -------------------------------------------
        log.info("Loading flood label…")
        with rasterio.open(flood_path) as src:
            flood_class = src.read(1).astype(np.uint8)   # fused class
            confidence  = src.read(2).astype(np.uint8)   # confidence

        flood_binary = np.isin(flood_class, [1, 2, 3]).astype(np.uint8)  # any flood
        valid_mask   = (flood_class != 9)                                  # excludes nodata
        perm_water   = (flood_class == 4)                                  # permanent water

        # Training label: flood=1, not_flood=0 (exclude perm water + nodata)
        train_valid = valid_mask & ~perm_water
        log.info("  Flood pixels:        %8d  (%.2f%%)",
                 int(flood_binary[train_valid].sum()),
                 100 * float(flood_binary[train_valid].mean()))
        log.info("  Non-flood pixels:    %8d",
                 int((~flood_binary.astype(bool) & train_valid).sum()))
        log.info("  Valid training px:   %8d", int(train_valid.sum()))

        # ---- Load feature arrays ----------------------------------------
        log.info("Loading and aligning %d feature rasters…", n_feat)
        arrays: dict[str, np.ndarray] = {}
        for name, path in feature_paths.items():
            log.info("  %s…", name)
            with rasterio.open(path) as s:
                same_grid = (s.height == height and s.width == width and
                             s.transform == transform)
            if same_grid:
                with rasterio.open(path) as s:
                    arr = s.read(1).astype(np.float32)
            else:
                arr = read_aligned(str(path), str(template_path), Resampling.bilinear)
            arrays[name] = arr

        # ---- Write multi-band feature stack TIF -------------------------
        stack_path = feat_dir / "flood_model_feature_stack.tif"
        log.info("Writing feature stack (%d bands)…", n_feat)
        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=n_feat, nodata=np.nan,
                           compress="lzw", bigtiff="IF_SAFER")
        with rasterio.open(stack_path, "w", **out_profile) as dst:
            for i, (name, arr) in enumerate(arrays.items(), 1):
                dst.write(arr, i)
                dst.update_tags(i, name=name)
        log.info("  Saved: %s  (%.1f MB)", stack_path,
                 stack_path.stat().st_size / 1e6)

        # ---- Feature metadata CSV ----------------------------------------
        meta_path = feat_dir / "feature_metadata.csv"
        with open(meta_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["band", "feature_name", "source_file",
                        "valid_pct", "mean", "std", "min", "max"])
            for i, (name, arr) in enumerate(arrays.items(), 1):
                valid = arr[np.isfinite(arr)]
                w.writerow([
                    i, name, feature_paths[name].name,
                    round(100 * len(valid) / arr.size, 2),
                    round(float(valid.mean()), 5) if valid.size else "NaN",
                    round(float(valid.std()),  5) if valid.size else "NaN",
                    round(float(valid.min()),  5) if valid.size else "NaN",
                    round(float(valid.max()),  5) if valid.size else "NaN",
                ])
        log.info("  Saved metadata: %s", meta_path)

        # ---- Stratified pixel sampling for training table ---------------
        log.info("Sampling training pixels (max flood=%d, non-flood=%d)…",
                 MAX_FLOOD_SAMPLES, MAX_NOFLOOD_SAMPLES)
        rng = np.random.default_rng(RANDOM_SEED)

        # Require valid event-period S2 (96% coverage) as minimum sample filter
        ndwi_valid = np.isfinite(arrays.get("ndwi_event", np.ones((height, width))))
        sample_valid = train_valid & ndwi_valid

        flood_idx    = np.where(flood_binary.ravel() & sample_valid.ravel())[0]
        nonflood_idx = np.where((~flood_binary.astype(bool)).ravel() & sample_valid.ravel())[0]

        n_flood    = min(len(flood_idx),    MAX_FLOOD_SAMPLES)
        n_nonflood = min(len(nonflood_idx), MAX_NOFLOOD_SAMPLES)

        sampled_flood    = rng.choice(flood_idx,    n_flood,    replace=False)
        sampled_nonflood = rng.choice(nonflood_idx, n_nonflood, replace=False)
        sampled_all      = np.concatenate([sampled_flood, sampled_nonflood])
        rng.shuffle(sampled_all)

        log.info("  Flood samples:    %d", n_flood)
        log.info("  Non-flood samples: %d", n_nonflood)
        log.info("  Total samples:    %d", len(sampled_all))

        # Build DataFrame
        rows, cols = np.unravel_index(sampled_all, (height, width))
        sample_dict = {
            "pixel_row":       rows.astype(np.int32),
            "pixel_col":       cols.astype(np.int32),
            "flood_label":     flood_binary.ravel()[sampled_all].astype(np.uint8),
            "flood_class":     flood_class.ravel()[sampled_all].astype(np.uint8),
            "confidence":      confidence.ravel()[sampled_all].astype(np.uint8),
        }
        for name, arr in arrays.items():
            sample_dict[name] = arr.ravel()[sampled_all]

        df = pd.DataFrame(sample_dict)
        feat_cols = list(arrays.keys())

        # --- Imputed columns (pre-event S2 + SAR): cloud/orbit gaps, not invalid pixels ---
        impute_cols = [c for c in feat_cols
                       if any(c.startswith(p) for p in IMPUTE_COLS_PREFIXES)]
        # Core columns that MUST be valid: terrain / hydro / settlement / event-S2
        core_feat_cols = [c for c in feat_cols if c not in impute_cols]

        # Pre-event S2: add validity flag, then impute with column median
        s2_pre_cols = [c for c in impute_cols if not c.startswith("sar_")]
        if s2_pre_cols:
            s2_pre_any_valid = df[s2_pre_cols].notna().any(axis=1)
            df["s2_pre_valid"] = s2_pre_any_valid.astype(np.uint8)
            df[s2_pre_cols] = df[s2_pre_cols].fillna(df[s2_pre_cols].median())
            log.info("  Pre-event S2 valid pixels in sample: %d / %d (%.1f%%)",
                     int(s2_pre_any_valid.sum()), len(df),
                     100 * float(s2_pre_any_valid.mean()))

        # SAR: add validity flag, then impute with column median
        sar_cols = [c for c in impute_cols if c.startswith("sar_")]
        if sar_cols:
            sar_any_valid = df[sar_cols].notna().any(axis=1)
            df["sar_data_valid"] = sar_any_valid.astype(np.uint8)
            df[sar_cols] = df[sar_cols].fillna(df[sar_cols].median())
            log.info("  SAR valid pixels in sample: %d / %d (%.1f%%)",
                     int(sar_any_valid.sum()), len(df),
                     100 * float(sar_any_valid.mean()))

        # Core features: only drop rows where terrain/hydro/settlement/event-S2 is NaN
        # (these NaNs = permanent water body or out-of-AOI — correctly excluded)
        n_before = len(df)
        df = df.dropna(subset=core_feat_cols)
        log.info("  Dropped %d rows with NaN in core features; final: %d rows",
                 n_before - len(df), len(df))

        parquet_path = feat_dir / "model_training_samples.parquet"
        df.to_parquet(str(parquet_path), index=False)
        log.info("  Saved training table: %s  (%d rows × %d cols)",
                 parquet_path, len(df), len(df.columns))

        # Quick class balance check
        log.info("  Label balance: flood=%.1f%%  non-flood=%.1f%%",
                 100 * df["flood_label"].mean(),
                 100 * (1 - df["flood_label"].mean()))

    log.info("Feature stack assembly complete in %.1f s", t.elapsed)
    log_run(
        script=__file__,
        input_paths=[str(flood_path)] + [str(p) for p in feature_paths.values()],
        output_paths=[str(stack_path), str(parquet_path), str(meta_path)],
        parameters={"n_bands": n_feat, "n_samples": len(df),
                    "flood_pct": round(100 * df["flood_label"].mean(), 1)},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble feature stack")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
