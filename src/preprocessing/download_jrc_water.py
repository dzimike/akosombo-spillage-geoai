from __future__ import annotations
"""
JRC Global Surface Water occurrence on the 10 m analysis grid.

The flood-mapping scripts (sar_change_detection, water_index_mapping,
flood_extent_fusion) were written to use data/processed/satellite/
jrc_permanent_water.tif (occurrence in %, permanent water = >= 50) but the
raster was never downloaded, so they silently fell back to pre-event optical
water, which is only ~47 % valid. This script fetches it.

Output: data/processed/satellite/jrc_permanent_water.tif (float32, occurrence 0-100,
0 where JRC has no observation)
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.windows import Window

from src.preprocessing.build_s1_composites import fetch_tile, GEE_PROJECT, TILE
from src.utils.paths import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    import ee
    ee.Initialize(project=GEE_PROJECT)
    root = get_project_root()
    with rasterio.open(root / "archive/v2_sparse_s1/s1_pre_event_vv.tif") as s:
        tr, crs, H, W = s.transform, s.crs.to_string(), s.height, s.width
    img = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).toFloat()
    out = root / "data/processed/satellite/jrc_permanent_water.tif"
    prof = {"driver": "GTiff", "dtype": "float32", "count": 1, "height": H, "width": W, "crs": crs,
            "transform": tr, "nodata": np.nan, "compress": "lzw", "tiled": True, "BIGTIFF": "YES"}
    with rasterio.open(out, "w", **prof):
        pass
    jobs = [(r, c) for r in range(0, H, TILE) for c in range(0, W, TILE)]
    done = 0
    with ThreadPoolExecutor(4) as ex, rasterio.open(out, "r+") as dst:
        futs = {ex.submit(fetch_tile, img, crs, tr, r, c, min(TILE, H - r), min(TILE, W - c)): (r, c) for r, c in jobs}
        for f in as_completed(futs):
            r, c = futs[f]
            d = f.result()[0]
            dst.write(d, 1, window=Window(c, r, d.shape[1], d.shape[0]))
            done += 1
            if done % 20 == 0:
                log.info("  %d/%d tiles", done, len(jobs))
    with rasterio.open(out) as s:
        a = s.read(1)[::4, ::4]
    log.info("done: occurrence>=50 covers %.1f%% of the grid", 100 * (a >= 50).mean())


if __name__ == "__main__":
    main()
