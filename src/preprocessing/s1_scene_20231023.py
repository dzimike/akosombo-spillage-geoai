from __future__ import annotations
"""
Download the Sentinel-1 scene of 23 October 2023 (relative orbit 74, ascending) on the 10 m grid, so our
water classification can be compared with the UNOSAT FL20231018GHA product on the SAME date.

7x7 focal mean on the dB values (as in the composites). Output: data/processed/satellite/s1_scene_20231023_{vv,vh}.tif
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
FILL = -9999.0


def main() -> None:
    import ee
    ee.Initialize(project=GEE_PROJECT)
    root = get_project_root()
    ref = root / "data/processed/satellite/s1_pre_event_vv.tif"
    with rasterio.open(ref) as s:
        tr, crs, H, W = s.transform, s.crs.to_string(), s.height, s.width
    region = ee.Geometry.Rectangle([tr.c, tr.f + H * tr.e, tr.c + W * tr.a, tr.f], proj=crs, geodesic=False)
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate("2023-10-23", "2023-10-24")
           .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.eq("relativeOrbitNumber_start", 74)))
    n = col.size().getInfo()
    log.info("scenes found: %d", n)
    sm = col.mosaic().select(["VV", "VH"]).focal_mean(radius=3.5, kernelType="square", units="pixels")
    img = ee.Image.constant([FILL, FILL]).rename(["VV", "VH"]).blend(sm.unmask(FILL))
    prof = {"driver": "GTiff", "dtype": "float32", "count": 1, "height": H, "width": W, "crs": crs, "transform": tr,
            "nodata": np.nan, "compress": "lzw", "tiled": True, "BIGTIFF": "YES"}
    out = {k: root / f"data/processed/satellite/s1_scene_20231023_{k}.tif" for k in ("vv", "vh")}
    for p in out.values():
        with rasterio.open(p, "w", **prof):
            pass
    jobs = [(r, c) for r in range(0, H, TILE) for c in range(0, W, TILE)]
    with ThreadPoolExecutor(4) as ex, rasterio.open(out["vv"], "r+") as dv, rasterio.open(out["vh"], "r+") as dh:
        futs = {ex.submit(fetch_tile, img, crs, tr, r, c, min(TILE, H - r), min(TILE, W - c)): (r, c) for r, c in jobs}
        done = 0
        for f in as_completed(futs):
            r, c = futs[f]; d = f.result(); d[d <= FILL + 1] = np.nan
            w = Window(c, r, d.shape[2], d.shape[1])
            dv.write(d[0].astype(np.float32), 1, window=w); dh.write(d[1].astype(np.float32), 1, window=w)
            done += 1
            if done % 20 == 0:
                log.info("  %d/%d tiles", done, len(jobs))
    log.info("done")


if __name__ == "__main__":
    main()
