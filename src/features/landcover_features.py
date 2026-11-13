from __future__ import annotations
"""
Phase 6 – Task 6.4: Land-Cover Feature Engineering

Derives a rule-based land-cover classification and binary class masks
from pre-event Sentinel-2 spectral indices.

Classification rules (pre-event NDVI / MNDWI / NDBI)
-----------------------------------------------------
0 = no data
1 = vegetation  (NDVI > 0.30)
2 = water       (MNDWI > 0.00, takes precedence)
3 = built-up    (NDBI > 0.10)
4 = bare soil   (NDVI < 0.10 AND MNDWI < 0 AND NDBI < 0)
5 = other

Features produced
-----------------
landcover_class     — 0-5 integer class
vegetation_binary   — 0/1
water_binary        — 0/1
builtup_binary      — 0/1
ndvi_pre            — pass-through index
mndwi_pre           — pass-through index
ndbi_event          — event NDBI (built-up change indicator)

Outputs
-------
data/processed/features/landcover/  (one GeoTIFF per feature)
"""

import argparse
import logging

import numpy as np
import rasterio
from rasterio.enums import Resampling

from src.utils.paths import load_config, get_processed_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEMPLATE_RASTER = "satellite/s2_ndwi_event.tif"


def classify_landcover(ndvi: np.ndarray, mndwi: np.ndarray,
                       ndbi: np.ndarray) -> np.ndarray:
    """Rule-based land-cover classification (uint8)."""
    lc = np.full_like(ndvi, 5, dtype=np.uint8)   # default: other
    lc = np.where(np.isfinite(ndvi), lc, 0)       # no-data
    lc = np.where(np.isfinite(ndvi) & (ndvi > 0.30), 1, lc)   # vegetation
    lc = np.where(np.isfinite(mndwi) & (mndwi > 0.00), 2, lc) # water (override)
    lc = np.where(np.isfinite(ndbi) & (ndbi > 0.10) & (lc != 2), 3, lc)  # built-up
    bare = (np.isfinite(ndvi)  & (ndvi  < 0.10) &
            np.isfinite(mndwi) & (mndwi < 0.00) &
            np.isfinite(ndbi)  & (ndbi  < 0.00))
    lc = np.where(bare & (lc == 5), 4, lc)
    return lc


def write_float(arr, profile, path, description):
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.update_tags(description=description)


def write_uint8(arr, profile, path, description):
    p = profile.copy()
    p.update(dtype="uint8", count=1, nodata=255, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.uint8), 1)
        dst.update_tags(description=description)


def main(config_path: str | None = None) -> None:
    config   = load_config(config_path)
    sat_dir  = get_processed_path("satellite", config)
    feat_dir = get_processed_path("features/landcover", config)
    ensure_dir(feat_dir)

    template     = get_processed_path(TEMPLATE_RASTER, config)
    ndvi_pre_p   = sat_dir / "s2_ndvi_pre.tif"
    ndwi_pre_p   = sat_dir / "s2_ndwi_pre.tif"
    mndwi_pre_p  = sat_dir / "s2_mndwi_pre.tif"
    ndbi_evt_p   = sat_dir / "s2_ndbi_event.tif"

    for p in [ndvi_pre_p, mndwi_pre_p, ndbi_evt_p]:
        if not p.exists():
            log.error("Missing: %s", p); return

    with rasterio.open(template) as src:
        profile = src.profile.copy()

    with Timer() as t:
        log.info("Loading pre-event S2 indices…")
        # These are already on the 10m template grid (same GEE export)
        def load(path):
            with rasterio.open(path) as s:
                return s.read(1).astype(np.float32)

        ndvi  = load(ndvi_pre_p)
        mndwi = load(mndwi_pre_p)
        ndbi  = load(ndbi_evt_p)     # event NDBI — better signal for built-up
        ndwi  = load(ndwi_pre_p) if ndwi_pre_p.exists() else np.full_like(ndvi, np.nan)

        log.info("  Classifying land cover…")
        lc = classify_landcover(ndvi, mndwi, ndbi)

        class_labels = {1:"vegetation", 2:"water", 3:"built-up", 4:"bare_soil", 5:"other"}
        for code, label in class_labels.items():
            n = int((lc == code).sum())
            log.info("    %2d %-12s  %8d px  (%.1f%%)", code, label,
                     n, 100 * n / lc.size)

        # ---- Write outputs -----------------------------------------------
        out_lc = feat_dir / "landcover_class.tif"
        write_uint8(lc, profile, out_lc,
                    "Rule-based LC: 0=nodata,1=veg,2=water,3=builtup,4=baresoil,5=other")
        log.info("  Saved: %s", out_lc)

        binary_maps = {
            "vegetation_binary": (lc == 1).astype(np.uint8),
            "water_binary":      (lc == 2).astype(np.uint8),
            "builtup_binary":    (lc == 3).astype(np.uint8),
        }
        for name, arr in binary_maps.items():
            out = feat_dir / f"{name}.tif"
            write_uint8(arr, profile, out, f"{name} from rule-based LC")
            log.info("  Saved: %s", out)

        index_maps = {
            "ndvi_pre":  (ndvi,  "Pre-event NDVI (Sentinel-2)"),
            "mndwi_pre": (mndwi, "Pre-event MNDWI (Sentinel-2)"),
            "ndwi_pre":  (ndwi,  "Pre-event NDWI (Sentinel-2)"),
            "ndbi_event":(ndbi,  "Event-period NDBI (Sentinel-2)"),
        }
        for name, (arr, desc) in index_maps.items():
            out = feat_dir / f"{name}.tif"
            write_float(arr, profile, out, desc)
            log.info("  Saved: %s", out)

    log.info("Land-cover features complete in %.1f s", t.elapsed)
    log_run(
        script=__file__,
        input_paths=[str(ndvi_pre_p), str(mndwi_pre_p)],
        output_paths=[str(feat_dir / f) for f in
                      ["landcover_class.tif","ndvi_pre.tif","mndwi_pre.tif"]],
        parameters={"classification": "rule-based NDVI/MNDWI/NDBI"},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Land-cover feature engineering")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
