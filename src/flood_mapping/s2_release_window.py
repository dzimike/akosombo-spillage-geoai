from __future__ import annotations
"""
Sentinel-2 NDWI/MNDWI composites for the release period only (15 Sep-30 Oct 2023), to test the optical window.

The event composite used by the flood map spans 15 Sep-15 Nov 2023, two weeks past the end of the release. This script builds the same median
composite (same collection, scene cloud filter and SCL mask as src/preprocessing/preprocess_sentinel2.py) for the release period and, for a
like-for-like comparison, rebuilds the 15 Sep-15 Nov composite the same way. Both are pulled on the analysis grid (EPSG:32630, 10 m, the grid of
fused_flood_confidence.tif) with ee.data.computePixels in tiles.

Output: data/processed/satellite/s2_water_indices_windows.tif (int16, x10000; bands: NDWI event, MNDWI event, NDWI release, MNDWI release;
-32768 = no valid observation), and the scene counts in s2_water_indices_windows.json.
"""
import io
import json
import logging

import numpy as np
import rasterio

from src.preprocessing.preprocess_sentinel2 import add_indices, init_gee, mask_s2_clouds
from src.utils.paths import get_project_root, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
WINDOWS = {"event": ("2023-09-15", "2023-11-15"), "release": ("2023-09-15", "2023-10-31")}  # filterDate end is exclusive
TILE = 1536
NODATA = -32768


def composite(ee, geom, start, end, max_cloud):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud)).map(mask_s2_clouds).map(add_indices))
    return col.select(["NDWI", "MNDWI"]).median(), col.size().getInfo()


def main() -> None:
    root = get_project_root()
    gee_cfg = load_config(root / "config/gee_config.yaml")
    ee = init_gee(gee_cfg)
    max_cloud = gee_cfg.get("sentinel2", {}).get("cloud_cover_max", 20)
    ref = rasterio.open(root / "data/processed/flood_extent/fused_flood_confidence.tif")
    H, W = ref.shape
    tr = ref.transform
    x0, y1 = tr.c, tr.f
    geom = ee.Geometry.Rectangle([x0, y1 - H * 10, x0 + W * 10, y1], proj="EPSG:32630", geodesic=False)
    imgs, counts = [], {}
    for k, (a, b) in WINDOWS.items():
        im, n = composite(ee, geom, a, b, max_cloud)
        counts[k] = {"start": a, "end_exclusive": b, "scenes": n}
        log.info("%s %s-%s: %d scenes", k, a, b, n)
        imgs.append(im.rename([f"NDWI_{k}", f"MNDWI_{k}"]))
    stack = ee.Image.cat(imgs).multiply(10000).round().toInt16().unmask(NODATA)
    out = np.full((4, H, W), NODATA, np.int16)
    for r in range(0, H, TILE):
        for c in range(0, W, TILE):
            h, w = min(TILE, H - r), min(TILE, W - c)
            req = {"expression": stack, "fileFormat": "NPY",
                   "grid": {"dimensions": {"width": w, "height": h},
                            "affineTransform": {"scaleX": 10, "shearX": 0, "translateX": x0 + c * 10, "shearY": 0, "scaleY": -10, "translateY": y1 - r * 10},
                            "crsCode": "EPSG:32630"}}
            for attempt in range(4):
                try:
                    arr = np.load(io.BytesIO(ee.data.computePixels(req)))
                    break
                except Exception as e:  # transient quota/timeouts
                    log.warning("tile %d,%d attempt %d: %s", r, c, attempt, e)
            else:
                raise RuntimeError(f"tile {r},{c} failed")
            for i, name in enumerate(arr.dtype.names):
                out[i, r:r + h, c:c + w] = arr[name]
            log.info("tile %d,%d done", r, c)
    prof = ref.profile | {"count": 4, "dtype": "int16", "nodata": NODATA, "compress": "deflate"}
    dst = root / "data/processed/satellite/s2_water_indices_windows.tif"
    with rasterio.open(dst, "w", **prof) as d:
        d.write(out)
        d.descriptions = ("NDWI_event", "MNDWI_event", "NDWI_release", "MNDWI_release")
    dst.with_suffix(".json").write_text(json.dumps(counts, indent=1))
    log.info("saved %s", dst)


if __name__ == "__main__":
    main()
