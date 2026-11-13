from __future__ import annotations
"""
Phase 9 – Task 9.1: Validation Sample Generation

Creates a stratified random sample of validation points across all flood
confidence classes (SAR-only, optical-only, agreement, non-flood,
permanent water) spatially separated from the ML training blocks.

Samples are drawn from the fused flood confidence raster using the
same 10 km spatial block grid as the model split, drawing ONLY from
test-split blocks (blocks 70%+ away from training data) to ensure
independence.

Target: ~500 points stratified across classes.

Outputs
-------
data/processed/validation/validation_sample_points.gpkg
data/processed/validation/validation_sample_points.csv
"""

import argparse
import logging

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.transform
from shapely.geometry import Point

from src.utils.crs import PROJ_CRS
from src.utils.paths import load_config, get_processed_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Fused confidence class codes
CLASS_MAP = {
    0: "not_flooded",
    1: "sar_only",
    2: "optical_only",
    3: "sar_optical_agreement",
    4: "permanent_water",
    9: "nodata",
}
# Samples per class (non-flood gets more to balance)
SAMPLES_PER_CLASS = {
    0: 150,
    1: 50,
    2: 75,
    3: 75,
    4: 50,
}
BLOCK_SIZE_PX = 1000   # 10 km at 10 m resolution
RANDOM_SEED   = 99     # different from training seed (42)


def get_test_block_mask(height: int, width: int,
                        test_block_ids: set[str]) -> np.ndarray:
    """Return bool mask of pixels belonging to test spatial blocks."""
    rows_grid = np.arange(height) // BLOCK_SIZE_PX
    cols_grid = np.arange(width)  // BLOCK_SIZE_PX
    rr, cc    = np.meshgrid(rows_grid, cols_grid, indexing="ij")
    block_ids = np.char.add(np.char.add(rr.astype(str), "_"), cc.astype(str))
    mask = np.zeros((height, width), dtype=bool)
    for bid in test_block_ids:
        mask |= (block_ids == bid)
    return mask


def stratified_sample(flood_conf_path: str,
                      test_block_ids: set[str],
                      seed: int = RANDOM_SEED) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)

    with rasterio.open(flood_conf_path) as src:
        data      = src.read(1).astype(np.int16)
        transform = src.transform
        crs_str   = src.crs.to_string()
        H, W      = src.height, src.width

    test_mask = get_test_block_mask(H, W, test_block_ids)
    log.info("Test-block pixels available for sampling: %d (%.1f%%)",
             int(test_mask.sum()), 100 * float(test_mask.mean()))

    rows = []
    point_id = 0
    for cls_val, n_target in SAMPLES_PER_CLASS.items():
        label    = CLASS_MAP.get(cls_val, str(cls_val))
        eligible = np.where((data == cls_val) & test_mask)
        n_avail  = len(eligible[0])
        n_draw   = min(n_target, n_avail)
        if n_draw == 0:
            log.warning("Class %d (%s): 0 eligible pixels in test blocks", cls_val, label)
            continue
        chosen = rng.choice(n_avail, n_draw, replace=False)
        for idx in chosen:
            r, c = eligible[0][idx], eligible[1][idx]
            x, y = rasterio.transform.xy(transform, r, c)
            rows.append({
                "point_id":    point_id,
                "class_value": int(cls_val),
                "class_label": label,
                "block_id":    f"{r // BLOCK_SIZE_PX}_{c // BLOCK_SIZE_PX}",
                "pixel_row":   int(r),
                "pixel_col":   int(c),
                "notes":       "auto-generated — requires human review for operational use",
                "geometry":    Point(x, y),
            })
            point_id += 1
        log.info("  Class %d %-25s  sampled %d / %d available",
                 cls_val, label, n_draw, n_avail)

    gdf = gpd.GeoDataFrame(rows, crs=crs_str)
    return gdf


def main(config_path: str | None = None) -> None:
    config  = load_config(config_path)
    val_dir = get_processed_path("validation", config)
    ensure_dir(val_dir)

    flood_conf_p  = get_processed_path("flood_extent/fused_flood_confidence.tif", config)
    sample_path_p = get_processed_path("features/model_training_samples.parquet", config)

    if not flood_conf_p.exists():
        log.error("Missing flood confidence raster: %s", flood_conf_p); return

    # Reproduce test-block IDs from training split (seed 42, 70/15/15)
    if sample_path_p.exists():
        df = pd.read_parquet(sample_path_p)
        df["block_id"] = (df.pixel_row // BLOCK_SIZE_PX).astype(str) + "_" + \
                         (df.pixel_col // BLOCK_SIZE_PX).astype(str)
        rng_split = np.random.default_rng(42)
        blocks    = df.block_id.unique()
        rng_split.shuffle(blocks)
        n         = len(blocks)
        test_blocks = set(blocks[int(n * 0.85):])
        log.info("Test blocks (%d): %s", len(test_blocks), sorted(test_blocks))
    else:
        log.warning("Training parquet not found — sampling from entire raster")
        test_blocks = None

    with Timer() as t:
        gdf = stratified_sample(str(flood_conf_p), test_blocks or set())

    out_gpkg = val_dir / "validation_sample_points.gpkg"
    out_csv  = val_dir / "validation_sample_points.csv"
    gdf.to_file(str(out_gpkg), driver="GPKG")
    gdf.drop(columns="geometry").to_csv(str(out_csv), index=False)

    log.info("Validation points: %d total → %s", len(gdf), out_gpkg)
    log.warning("Points are auto-generated. Review against imagery before use as ground truth.")

    log_run(
        script=__file__,
        input_paths=[str(flood_conf_p)],
        output_paths=[str(out_gpkg), str(out_csv)],
        parameters={"n_points": len(gdf), "test_blocks": sorted(test_blocks or [])},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate validation sample points")
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
