from __future__ import annotations
"""
Per-acquisition-date Sentinel-1 change flags, 1 July 2023 - 31 January 2024, for the per-date change time series.

For each relative orbit (74, 147) the reference is the same orbit-matched median of the 7x7 focal-mean dB scenes of the
pre-event window (1 Aug - 10 Sep 2023) used by s1_change_stack.py. Every acquisition date of that orbit in the
period (frames mosaicked per date) is differenced against it and encoded in one uint8 band:

    bit 0 (1)  valid (both polarisations observed)
    bit 1 (2)  dVV < -2 dB and dVH < -2 dB
    bit 2 (4)  dVV < -3 dB and dVH < -3 dB      (primary threshold)
    bit 3 (8)  dVV < -4 dB and dVH < -4 dB

Output: data/processed/satellite/s1_change_by_date.tif (uint8, one band per orbit-date, band description 'orbit_date')
        data/processed/satellite/s1_change_by_date_bands.csv
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.preprocessing.build_s1_composites import fetch_tile, GEE_PROJECT, TILE
from src.utils.paths import get_project_root, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
START, END = "2023-07-01", "2024-02-01"


def main() -> None:
    import ee
    ee.Initialize(project=GEE_PROJECT)
    root = get_project_root()
    cfg = load_config()["event_windows"]
    ref = root / "data/processed/satellite/s1_pre_event_vv.tif"
    with rasterio.open(ref) as s:
        tr, crs, H, W = s.transform, s.crs.to_string(), s.height, s.width
    region = ee.Geometry.Rectangle([tr.c, tr.f + H * tr.e, tr.c + W * tr.a, tr.f], proj=crs, geodesic=False)

    def coll(a, b, orbit):
        return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate(a, b)
                .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select(["VV", "VH"]))
    fm = lambda im: im.focal_mean(radius=3.5, kernelType="square", units="pixels")
    bands, names = [], []
    for orbit in (74, 147):
        pre = coll(cfg["pre_event"]["start"], cfg["pre_event"]["end"], orbit).map(fm).median()
        col = coll(START, END, orbit)
        dates = sorted(set(col.aggregate_array("system:time_start").map(lambda t: ee.Date(t).format("YYYY-MM-dd")).getInfo()))
        log.info("orbit %d dates (%d): %s", orbit, len(dates), dates)
        for d in dates:
            d0 = ee.Date(d)
            scene = fm(col.filterDate(d0, d0.advance(1, "day")).mosaic())
            dd = scene.subtract(pre)
            dvv, dvh = dd.select("VV"), dd.select("VH")
            valid = dvv.mask().And(dvh.mask())
            code = (valid.unmask(0)
                    .add(dvv.lt(-2).And(dvh.lt(-2)).And(valid).unmask(0).multiply(2))
                    .add(dvv.lt(-3).And(dvh.lt(-3)).And(valid).unmask(0).multiply(4))
                    .add(dvv.lt(-4).And(dvh.lt(-4)).And(valid).unmask(0).multiply(8)))
            bands.append(code.toUint8().rename(f"o{orbit}_{d}")); names.append((orbit, d))
    img = ee.Image.cat(bands)
    out = root / "data/processed/satellite/s1_change_by_date.tif"
    prof = {"driver": "GTiff", "dtype": "uint8", "count": len(names), "height": H, "width": W, "crs": crs, "transform": tr,
            "nodata": 0, "compress": "lzw", "tiled": True, "BIGTIFF": "YES"}
    with rasterio.open(out, "w", **prof) as dst:
        for i, (o, d) in enumerate(names, 1):
            dst.set_band_description(i, f"{o}_{d}")
    pd.DataFrame(names, columns=["orbit", "date"]).to_csv(root / "data/processed/satellite/s1_change_by_date_bands.csv", index=False)
    jobs = [(r, c) for r in range(0, H, TILE) for c in range(0, W, TILE)]
    done = 0
    with ThreadPoolExecutor(3) as ex, rasterio.open(out, "r+") as dst:
        futs = {ex.submit(fetch_tile, img, crs, tr, r, c, min(TILE, H - r), min(TILE, W - c)): (r, c) for r, c in jobs}
        for f in as_completed(futs):
            r, c = futs[f]; d = f.result()
            dst.write(d.astype(np.uint8), window=Window(c, r, d.shape[2], d.shape[1]), indexes=list(range(1, len(names) + 1)))
            done += 1
            if done % 10 == 0:
                log.info("  %d/%d tiles", done, len(jobs))
    log.info("done: %d bands", len(names))


if __name__ == "__main__":
    main()
