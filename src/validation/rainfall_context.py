from __future__ import annotations
"""
Local rainfall during the reference and release periods, for the hazard-attribution discussion.

CHIRPS daily precipitation (UCSB-CHG/CHIRPS/DAILY, 0.05 deg) is averaged over the analysis domain and summed over the pre-event reference window
(1 Aug-10 Sep), the release period (15 Sep-30 Oct) and the two weeks after it (31 Oct-15 Nov), for 2023 and for each year 1991-2022, so the 2023
totals can be placed in the 1991-2022 distribution. This is rainfall over the lower basin only; it says nothing about the upstream inflow that
caused the release.

Output: outputs/tables/rainfall_context.csv
"""
import logging

import pandas as pd
import rasterio

from src.preprocessing.preprocess_sentinel2 import init_gee
from src.utils.paths import get_project_root, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
WINDOWS = {"reference (1 Aug-10 Sep)": ("08-01", "09-11"), "release (15 Sep-30 Oct)": ("09-15", "10-31"), "after release (31 Oct-15 Nov)": ("10-31", "11-16")}


def main() -> None:
    root = get_project_root()
    ee = init_gee(load_config(root / "config/gee_config.yaml"))
    with rasterio.open(root / "data/processed/flood_extent/fused_flood_confidence.tif") as s:
        b = s.bounds
    geom = ee.Geometry.Rectangle([b.left, b.bottom, b.right, b.top], proj="EPSG:32630", geodesic=False)
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    rows = []
    for name, (a, z) in WINDOWS.items():
        def total(y):
            y = ee.Number(y).format("%d")
            img = chirps.filterDate(ee.String(y).cat("-").cat(a), ee.String(y).cat("-").cat(z)).sum()
            return ee.Feature(None, {"year": y, "mm": img.reduceRegion(ee.Reducer.mean(), geom, 5000).get("precipitation")})
        fc = ee.FeatureCollection(ee.List.sequence(1991, 2023).map(total)).getInfo()
        d = pd.DataFrame([f["properties"] for f in fc["features"]]).astype({"year": int, "mm": float})
        v23 = float(d.loc[d.year == 2023, "mm"].iloc[0]); clim = d[d.year < 2023].mm
        rows.append({"window": name, "mm_2023": round(v23, 1), "mm_1991_2022_median": round(clim.median(), 1),
                     "mm_1991_2022_p10": round(clim.quantile(0.1), 1), "mm_1991_2022_p90": round(clim.quantile(0.9), 1),
                     "percentile_2023": round(100 * (clim < v23).mean(), 0)})
        log.info("%s: %s", name, rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(root / "outputs/tables/rainfall_context.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
