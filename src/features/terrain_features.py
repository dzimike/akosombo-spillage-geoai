from __future__ import annotations
"""
Phase 6 – Task 6.1: Terrain Feature Engineering

Aligns DEM derivatives to the 10 m S2 reference grid and produces
terrain predictor rasters for flood susceptibility modelling.

Features produced
-----------------
elevation, slope, aspect_sin, aspect_cos, twi,
flow_accumulation_log, relative_elevation, hand_proxy

Notes
-----
- All outputs aligned to: data/processed/satellite/s2_ndwi_event.tif
- aspect converted to sin/cos components to avoid circular discontinuity
- flow_accumulation log-transformed (heavy right skew)
- HAND proxy = relative_elevation (height above valley floor)

Outputs
-------
data/processed/features/terrain/  (one GeoTIFF per feature)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from src.utils.paths import load_config, get_processed_path, ensure_dir
from src.utils.logging_utils import Timer, log_run
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEMPLATE_RASTER = "satellite/s2_ndwi_event.tif"


def write_feature(arr: np.ndarray, profile: dict, path: Path, description: str) -> None:
    """Write a float32 single-band feature raster."""
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.update_tags(description=description)


def main(config_path: str | None = None) -> None:
    config      = load_config(config_path)
    dem_dir     = get_processed_path("dem", config)
    feat_dir    = get_processed_path("features/terrain", config)
    ensure_dir(feat_dir)

    template    = get_processed_path(TEMPLATE_RASTER, config)

    with rasterio.open(template) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff")

    dem_files = {
        "elevation":         dem_dir / "elevation.tif",
        "slope":             dem_dir / "slope.tif",
        "aspect":            dem_dir / "aspect.tif",
        "twi":               dem_dir / "twi.tif",
        "flow_accumulation": dem_dir / "flow_accumulation.tif",
        "relative_elevation":dem_dir / "relative_elevation.tif",
    }
    for name, p in dem_files.items():
        if not p.exists():
            log.error("Missing: %s", p); return

    with Timer() as t:
        log.info("Aligning terrain features to 10 m grid…")

        features: dict[str, np.ndarray] = {}
        for name, path in dem_files.items():
            log.info("  Loading %s…", name)
            arr = read_aligned(str(path), str(template), Resampling.bilinear)
            # Replace nodata with NaN
            arr = np.where(arr < -9000, np.nan, arr).astype(np.float32)
            features[name] = arr

        # ---- Derived features -------------------------------------------
        # Aspect → sin/cos components
        log.info("  Computing aspect sin/cos…")
        rad = np.radians(features["aspect"])
        features["aspect_sin"] = np.sin(rad).astype(np.float32)
        features["aspect_cos"] = np.cos(rad).astype(np.float32)
        del features["aspect"]      # replaced by components

        # Flow accumulation → log1p (heavy right skew)
        log.info("  Log-transforming flow accumulation…")
        fa = features["flow_accumulation"]
        features["flow_accumulation_log"] = np.where(
            np.isfinite(fa), np.log1p(np.clip(fa, 0, None)), np.nan
        ).astype(np.float32)
        del features["flow_accumulation"]   # raw replaced by log

        # HAND proxy = relative_elevation (already computed in Phase 3)
        features["hand_proxy"] = features["relative_elevation"].copy()

        # ---- Write rasters -----------------------------------------------
        descriptions = {
            "elevation":           "Elevation above sea level (m), SRTM 30m → 10m",
            "slope":               "Slope (degrees), derived from SRTM",
            "aspect_sin":          "sin(aspect) — circular component",
            "aspect_cos":          "cos(aspect) — circular component",
            "twi":                 "Topographic Wetness Index ln(a/tanβ)",
            "flow_accumulation_log": "log1p(flow accumulation) — upslope contributing area",
            "relative_elevation":  "Height above local minimum (m), 5 km window",
            "hand_proxy":          "HAND proxy = relative_elevation (m)",
        }

        for name, arr in features.items():
            out = feat_dir / f"{name}.tif"
            write_feature(arr, profile, out, descriptions.get(name, name))
            valid_pct = 100 * np.isfinite(arr).sum() / arr.size
            log.info("  %-30s  valid=%.1f%%  range=[%.2f, %.2f]",
                     name, valid_pct,
                     float(np.nanmin(arr)), float(np.nanmax(arr)))

        log.info("Terrain features written to %s", feat_dir)

    log.info("Terrain feature engineering complete in %.1f s", t.elapsed)
    log_run(
        script=__file__,
        input_paths=[str(p) for p in dem_files.values()],
        output_paths=[str(feat_dir / f"{n}.tif") for n in features],
        parameters={"n_features": len(features), "resolution_m": 10},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terrain feature engineering")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
