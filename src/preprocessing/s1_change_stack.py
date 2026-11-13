from __future__ import annotations
"""
Scene-wise Sentinel-1 change stack for the event window (15 Sep - 15 Nov 2023).

For each relative orbit (74, 147) a pre-event reference is built (median of the 7x7 focal-mean dB of the pre-event
window 1 Aug - 10 Sep). Every event-window acquisition date of that orbit (frames mosaicked per date) is then
differenced against the orbit-matched reference, so that incidence-angle differences between orbits do not enter.
Per pixel, summed over all dates and both orbits:

  cnt2, cnt3, cnt4  number of dates with dVV < -k dB AND dVH < -k dB   (k = 2, 3, 4)
  n_obs             number of event-window observations (dates x orbits) with data
  min_dvv, min_dvh  minimum change over the observations (dB)

Output: data/processed/satellite/s1_change_stack.tif (float32, 6 bands, NaN where n_obs = 0)
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.windows import Window

from src.preprocessing.build_s1_composites import fetch_tile, GEE_PROJECT, TILE
from src.utils.paths import get_project_root, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
FILL = -9999.0
BANDS = ["cnt2", "cnt3", "cnt4", "n_obs", "min_dvv", "min_dvh"]


def main() -> None:
    import ee
    ee.Initialize(project=GEE_PROJECT)
    root = get_project_root()
    cfg = load_config()["event_windows"]
    ref = root / "archive/v2_sparse_s1/s1_pre_event_vv.tif"
    with rasterio.open(ref) as s:
        tr, crs, H, W = s.transform, s.crs.to_string(), s.height, s.width
    region = ee.Geometry.Rectangle([tr.c, tr.f + H * tr.e, tr.c + W * tr.a, tr.f], proj=crs, geodesic=False)

    def coll(a, b, orbit):
        return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate(a, b)
                .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select(["VV", "VH"]))
    fm = lambda im: im.focal_mean(radius=3.5, kernelType="square", units="pixels")
    cnt = {k: ee.Image.constant(0).toFloat() for k in (2, 3, 4)}
    nobs = ee.Image.constant(0).toFloat()
    min_vv = ee.Image.constant(999).toFloat(); min_vh = ee.Image.constant(999).toFloat()
    n_dates = {}
    for orbit in (74, 147):
        pre = coll(cfg["pre_event"]["start"], cfg["pre_event"]["end"], orbit).map(fm).median()
        ev = coll(cfg["event"]["start"], cfg["event"]["end"], orbit)
        dates = sorted(set(ev.aggregate_array("system:time_start").map(lambda t: ee.Date(t).format("YYYY-MM-dd")).getInfo()))
        n_dates[orbit] = dates
        log.info("orbit %d event dates: %s", orbit, dates)
        for d in dates:
            d0 = ee.Date(d)
            scene = fm(ev.filterDate(d0, d0.advance(1, "day")).mosaic())
            dd = scene.subtract(pre)
            valid = dd.select("VV").mask().And(dd.select("VH").mask())
            dvv, dvh = dd.select("VV"), dd.select("VH")
            for k in (2, 3, 4):
                flag = dvv.lt(-k).And(dvh.lt(-k)).And(valid)
                cnt[k] = cnt[k].add(flag.unmask(0))
            nobs = nobs.add(valid.unmask(0))
            min_vv = min_vv.min(dvv.unmask(999)); min_vh = min_vh.min(dvh.unmask(999))
    stack = (cnt[2].rename("cnt2").addBands(cnt[3].rename("cnt3")).addBands(cnt[4].rename("cnt4")).addBands(nobs.rename("n_obs"))
             .addBands(min_vv.rename("min_dvv")).addBands(min_vh.rename("min_dvh")))
    out = root / "data/processed/satellite/s1_change_stack.tif"
    prof = {"driver": "GTiff", "dtype": "float32", "count": 6, "height": H, "width": W, "crs": crs, "transform": tr,
            "nodata": np.nan, "compress": "lzw", "tiled": True, "BIGTIFF": "YES"}
    with rasterio.open(out, "w", **prof) as dst:
        for i, b in enumerate(BANDS, 1):
            dst.set_band_description(i, b)
    jobs = [(r, c) for r in range(0, H, TILE) for c in range(0, W, TILE)]
    done = 0
    with ThreadPoolExecutor(3) as ex, rasterio.open(out, "r+") as dst:
        futs = {ex.submit(fetch_tile, stack, crs, tr, r, c, min(TILE, H - r), min(TILE, W - c)): (r, c) for r, c in jobs}
        for f in as_completed(futs):
            r, c = futs[f]; d = f.result()
            d[:, d[3] <= 0] = np.nan                 # no observation
            d[4:, d[4] > 900] = np.nan
            dst.write(d.astype(np.float32), window=Window(c, r, d.shape[2], d.shape[1]), indexes=[1, 2, 3, 4, 5, 6])
            done += 1
            if done % 10 == 0:
                log.info("  %d/%d tiles", done, len(jobs))
    log.info("done; event dates %s", n_dates)


if __name__ == "__main__":
    main()
