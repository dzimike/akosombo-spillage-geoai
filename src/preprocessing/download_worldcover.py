from __future__ import annotations
"""
ESA WorldCover 2021 (v200, 10 m land-cover map; Zanaga et al., 2022) on the project 10 m grid.

Classes: 10 tree cover, 20 shrubland, 30 grassland, 40 cropland, 50 built-up, 60 bare / sparse vegetation,
70 snow and ice, 80 permanent water bodies, 90 herbaceous wetland, 95 mangroves, 100 moss and lichen.
Used as an independent land-use layer for the diagnosis of unconfirmed flood patches (validation/patch_landuse_check.py).

Output: data/processed/satellite/worldcover_2021.tif (uint8)
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
    ref = root / "data/processed/satellite/s1_pre_event_vv.tif"
    with rasterio.open(ref) as s:
        tr, crs, H, W = s.transform, s.crs.to_string(), s.height, s.width
    img = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").unmask(0)
    out = root / "data/processed/satellite/worldcover_2021.tif"
    prof = {"driver": "GTiff", "dtype": "uint8", "count": 1, "height": H, "width": W, "crs": crs, "transform": tr,
            "nodata": 0, "compress": "lzw", "tiled": True, "BIGTIFF": "YES"}
    with rasterio.open(out, "w", **prof):
        pass
    jobs = [(r, c) for r in range(0, H, TILE) for c in range(0, W, TILE)]
    done = 0
    with ThreadPoolExecutor(4) as ex, rasterio.open(out, "r+") as dst:
        futs = {ex.submit(fetch_tile, img, crs, tr, r, c, min(TILE, H - r), min(TILE, W - c)): (r, c) for r, c in jobs}
        for f in as_completed(futs):
            r, c = futs[f]; d = f.result()
            dst.write(d[0].astype(np.uint8), 1, window=Window(c, r, d.shape[2], d.shape[1]))
            done += 1
            if done % 20 == 0:
                log.info("  %d/%d tiles", done, len(jobs))
    log.info("done")


if __name__ == "__main__":
    main()
