from __future__ import annotations
"""
Per-scene Sentinel-1 open-water area through the 2023 season, on the analysis grid.

Every IW scene 1 Jul 2023 - 31 Jan 2024 is downloaded (7x7 focal-mean dB, as in the
composites) for three 15 km x 15 km reach windows and classified with the same
VH/VV water thresholds as the flood mask (sar_threshold_report.csv). Water is counted
only outside static water (JRC >= 50 %, open sea, pre-event optical water) and on
slopes <= 5 deg, i.e. under exactly the masks used for the flood labels. Pixels
without coverage in a scene are excluded from the covered area.

Windows: A = Mepe-Battor-Adidome reach, B = Sogakope-Sokpoe-Tadze reach,
C = Ada / Songor lagoon wetland. Relative orbit 74 covers all windows; orbit 147
covers only part of the raster (covered_km2 shows how much of each window).
Spillage: began 15 Sep 2023, stepped up 9 Oct, ended 30 Oct 2023.

Output: outputs/tables/s1_water_timeseries.csv
"""
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

from src.flood_mapping.water_masks import static_water
from src.preprocessing.build_s1_composites import GEE_PROJECT, fetch_tile
from src.utils.paths import get_project_root, get_output_path
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
WINDOWS = {"A_Mepe_Battor": (872000, 663000), "B_Sogakope_Sokpoe": (888000, 655000),
           "C_Ada_Songor": (878000, 640600)}
SIZE_M = 15000
FILL = -9999.0


def main() -> None:
    import ee
    ee.Initialize(project=GEE_PROJECT)
    root = get_project_root()
    sat, fl = root / "data/processed/satellite", root / "data/processed/flood_extent"
    rep = pd.read_csv(fl / "sar_threshold_report.csv").set_index("metric")["value"]
    t_vh, t_vv = float(rep["threshold_vh_dB_youden"]), float(rep["threshold_vv_dB_youden"])
    ref = sat / "s1_pre_event_vv.tif"
    with rasterio.open(ref) as s:
        tr, crs, H, W = s.transform, s.crs.to_string(), s.height, s.width
    static = static_water(ref, sat, fl)["union"]
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), str(ref), Resampling.bilinear)
    ok_land = ~static & ~(np.isfinite(slope) & (slope > 5))

    geoms = {}
    for name, (x0, y0) in WINDOWS.items():
        c0, r0 = int(round((x0 - tr.c) / tr.a)), int(round((tr.f - (y0 + SIZE_M)) / -tr.e))
        geoms[name] = (r0, c0, SIZE_M // 10, SIZE_M // 10)
    region = ee.Geometry.Rectangle([tr.c, tr.f - H * 10, tr.c + W * 10, tr.f], proj=crs, geodesic=False)
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate("2023-07-01", "2024-02-01")
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).sort("system:time_start"))
    n = col.size().getInfo()
    lst = col.toList(n)
    meta = []
    for i in range(n):
        img = ee.Image(lst.get(i))
        meta.append((i, img.date().format("YYYY-MM-dd").getInfo(), int(img.get("relativeOrbitNumber_start").getInfo())))

    def job(args):
        i, date, orbit, name = args
        sm = ee.Image(lst.get(i)).select(["VV", "VH"]).focal_mean(radius=3.5, kernelType="square", units="pixels")
        # blend onto an unbounded constant so a scene that covers only part of the
        # window still exports the full window (else Earth Engine crops to the footprint)
        img = ee.Image.constant([FILL, FILL]).rename(["VV", "VH"]).blend(sm.unmask(FILL))
        r0, c0, h, w = geoms[name]
        d = fetch_tile(img, crs, tr, r0, c0, h, w)
        vv, vh = d[0], d[1]
        cov = (vv > FILL + 1) & (vh > FILL + 1)
        land = ok_land[r0:r0 + h, c0:c0 + w]
        wat = cov & (vh < t_vh) & (vv < t_vv) & land
        return {"date": date, "relative_orbit": orbit, "window": name,
                "water_km2": round(float(wat.sum()) / 1e4, 3),
                "covered_land_km2": round(float((cov & land).sum()) / 1e4, 2)}

    jobs = [(i, d, o, nm) for (i, d, o) in meta for nm in WINDOWS]
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(job, jobs))
    df = pd.DataFrame(rows).sort_values(["window", "relative_orbit", "date"])
    df.to_csv(get_output_path("tables/s1_water_timeseries.csv"), index=False)
    log.info("wrote %d rows", len(df))


if __name__ == "__main__":
    main()
