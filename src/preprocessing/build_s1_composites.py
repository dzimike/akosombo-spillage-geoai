from __future__ import annotations
"""
Full-coverage Sentinel-1 composites (replaces the sparse s1_*.tif rasters).

Why this exists
---------------
preprocess_sentinel1.py applied 10*log10 to COPERNICUS/S1_GRD, whose VV/VH
bands are ALREADY in dB. log10 of a negative dB value is NaN, so the exported
rasters were valid only where backscatter was positive in dB (0.1-1.6 % of
pixels, median +0.5 dB) and were not composites of the scenes at all.

Processing chain here (identical to the original apart from the removed
double conversion): S1 GRD IW VV+VH, 7x7 focal mean on the dB values, per-window
median over ALL scenes of both relative orbits (74 and 147, ascending), windows
from config event_windows. Rasters are fetched from Earth Engine in tiles on the
exact 10 m UTM-30N grid of the existing rasters.

Outputs (data/processed/satellite/, float32, NaN = no scene coverage)
-------
s1_pre_event_vv/vh, s1_event_vv/vh, s1_post_event_vv, s1_vv_change,
s1_vh_change, s1_vv_ratio, plus s1_scene_counts.json
"""
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.windows import Window

from src.utils.paths import load_config, get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GEE_PROJECT = "instigis"
TILE = 750
BANDS = ["pre_vv", "pre_vh", "event_vv", "event_vh", "post_vv", "post_vh"]
OUT = {"pre_vv": "s1_pre_event_vv", "pre_vh": "s1_pre_event_vh", "event_vv": "s1_event_vv",
       "event_vh": "s1_event_vh", "post_vv": "s1_post_event_vv"}
FILL = -9999.0


def composite(ee, geom, start, end):
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .select(["VV", "VH"]))
    # bands are already dB: only the focal mean is applied, then the median
    img = col.map(lambda i: i.focal_mean(radius=3.5, kernelType="square", units="pixels")).median()
    return img, col


def fetch_tile(img, crs, tr, r0, c0, h, w):
    import ee
    x0 = tr.c + c0 * tr.a
    y_top = tr.f + r0 * tr.e
    region = ee.Geometry.Rectangle([x0, y_top + h * tr.e, x0 + w * tr.a, y_top], proj=crs, geodesic=False)
    url = img.getDownloadURL({"name": "s1", "crs": crs, "crs_transform": [tr.a, 0, x0, 0, tr.e, y_top],
                              "region": region, "format": "GEO_TIFF"})
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=600); r.raise_for_status()
            with rasterio.MemoryFile(r.content) as mf, mf.open() as s:
                data = s.read().astype(np.float32)
            return data[:, :h, :w]
        except Exception as exc:  # noqa: BLE001
            log.warning("tile %d,%d attempt %d failed: %s", r0, c0, attempt + 1, exc)
    raise RuntimeError(f"tile {r0},{c0} failed")


def main() -> None:
    import ee
    ee.Initialize(project=GEE_PROJECT)
    root = get_project_root()
    sat = root / "data/processed/satellite"
    cfg = load_config()["event_windows"]
    ref = sat / "s1_pre_event_vv.tif"
    arch = root / "archive/v2_sparse_s1"
    if not (arch / "s1_pre_event_vv.tif").exists():
        arch.mkdir(parents=True, exist_ok=True)
        for p in sat.glob("s1_*.tif"):
            shutil.copy2(p, arch / p.name)
        log.info("archived original sparse S1 rasters to %s", arch)
    with rasterio.open(arch / "s1_pre_event_vv.tif") as s:
        tr, crs, H, W = s.transform, s.crs.to_string(), s.height, s.width

    geom = ee.Geometry.Rectangle([tr.c, tr.f + H * tr.e, tr.c + W * tr.a, tr.f], proj=crs, geodesic=False)
    pre, c_pre = composite(ee, geom, cfg["pre_event"]["start"], cfg["pre_event"]["end"])
    ev, c_ev = composite(ee, geom, cfg["event"]["start"], cfg["event"]["end"])
    po, c_po = composite(ee, geom, cfg["post_event_early"]["start"], cfg["post_event_early"]["end"])
    counts = {}
    for n, c in (("pre", c_pre), ("event", c_ev), ("post", c_po)):
        counts[n] = {"scenes": c.size().getInfo(),
                     "orbit_pass": c.aggregate_histogram("orbitProperties_pass").getInfo(),
                     "relative_orbit": c.aggregate_histogram("relativeOrbitNumber_start").getInfo()}
    log.info("scene counts: %s", counts)
    (sat / "s1_scene_counts.json").write_text(json.dumps({"windows": cfg, "counts": counts}, indent=2, default=str))

    stack = (pre.select(["VV", "VH"], ["pre_vv", "pre_vh"]).addBands(ev.select(["VV", "VH"], ["event_vv", "event_vh"]))
             .addBands(po.select(["VV", "VH"], ["post_vv", "post_vh"])).unmask(FILL))

    prof = {"driver": "GTiff", "dtype": "float32", "count": 1, "height": H, "width": W, "crs": crs,
            "transform": tr, "nodata": np.nan, "compress": "lzw", "tiled": True, "BIGTIFF": "YES"}
    out_names = {**OUT, "vv_change": "s1_vv_change", "vh_change": "s1_vh_change", "vv_ratio": "s1_vv_ratio"}
    tmp = root / "data/raw/s1_rebuild"; tmp.mkdir(parents=True, exist_ok=True)
    dsts = {}
    for k, n in out_names.items():
        p = tmp / f"{n}.tif"
        with rasterio.open(p, "w", **prof):
            pass
        dsts[k] = p

    jobs = [(r, c) for r in range(0, H, TILE) for c in range(0, W, TILE)]
    log.info("fetching %d tiles", len(jobs))
    done = 0
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(fetch_tile, stack, crs, tr, r, c, min(TILE, H - r), min(TILE, W - c)): (r, c) for r, c in jobs}
        handles = {k: rasterio.open(p, "r+") for k, p in dsts.items()}
        for f in as_completed(futs):
            r, c = futs[f]; d = f.result()
            d[d <= FILL + 1] = np.nan
            win = Window(c, r, d.shape[2], d.shape[1])
            vals = dict(zip(BANDS, d))
            vals["vv_change"] = vals["event_vv"] - vals["pre_vv"]
            vals["vh_change"] = vals["event_vh"] - vals["pre_vh"]
            lin = lambda x: np.power(10.0, x / 10.0)
            vals["vv_ratio"] = lin(vals["event_vv"]) / lin(vals["pre_vv"])
            for k, h in handles.items():
                h.write(vals[k].astype(np.float32), 1, window=win)
            done += 1
            if done % 10 == 0:
                log.info("  %d/%d tiles", done, len(jobs))
        for h in handles.values():
            h.close()
    for k, p in dsts.items():
        shutil.move(str(p), sat / f"{out_names[k]}.tif")
    log.info("done; wrote %d rasters", len(dsts))


if __name__ == "__main__":
    main()
