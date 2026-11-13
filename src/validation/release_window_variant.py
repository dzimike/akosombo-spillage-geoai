from __future__ import annotations
"""
Release-window persistence variant of the SAR change definition.

The per-date series (s1_change_timeseries.py) shows that seasonal change affects the late dates of the 15 Sep - 15 Nov
event window: patches flagged on 4 and 9 Nov (and throughout Dec-Jan, with the whole-domain flagged area rising from
~50 km2 to 600 km2) can satisfy 'changed on >= 2 dates' without any change during the release (15 Sep - 30 Oct).
This script re-derives the SAR flood requiring >= M flagged dates (dVV & dVH < -K dB) within the release period only and
scores it against the reference products. Outputs: data/processed/flood_extent/sar_flood_mask_releasewindow.tif and
outputs/tables/release_window_variant.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize

from src.flood_mapping.sar_change_detection import remove_small_patches
from src.utils.paths import get_project_root, get_output_path
from src.utils.raster import read_aligned
from src.validation.compare_unosat import UN, metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
WIN = ("2023-09-15", "2023-10-30")


def main() -> None:
    root = get_project_root()
    sat, fl = root / "data/processed/satellite", root / "data/processed/flood_extent"
    bands = pd.read_csv(sat / "s1_change_by_date_bands.csv")
    sel = [i + 1 for i, d in enumerate(bands.date) if WIN[0] <= d <= WIN[1]]
    log.info("release-window dates (%d): %s", len(sel), list(bands.date[[i - 1 for i in sel]]))
    with rasterio.open(sat / "s1_change_by_date.tif") as src:
        cnt = np.zeros(src.shape, np.uint8)
        for i in sel:
            cnt += ((src.read(i) & 4) > 0).astype(np.uint8)
    fu = rasterio.open(fl / "fused_flood_confidence.tif").read(1)
    with rasterio.open(fl / "fused_flood_confidence.tif") as s:
        tr, crs, shape = s.transform, s.crs, s.shape
    pre = fu == 4
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), str(fl / "fused_flood_confidence.tif"), Resampling.bilinear)
    flat = ~(np.isfinite(slope) & (slope > 5))
    opt = (rasterio.open(fl / "optical_water_mask.tif").read(1) == 1) & ~pre
    prim = np.isin(fu, [1, 2, 3])
    prim_sar = rasterio.open(fl / "sar_flood_mask_binary.tif").read(1) == 1

    def ras(name):
        g = gpd.read_file(root / UN.format(name)).to_crs(crs)
        return rasterize(((geom, 1) for geom in g.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    dom, uf = ras("AnalysisExtent"), ras("FloodExtent")
    tsx = rasterio.open(root / "data/processed/validation/charter_terrasarx_20231016.tif").read(1)
    ins = tsx != 255
    rows = []
    for m in (1, 2, 3):
        sar = remove_small_patches(((cnt >= m) & ~pre & flat).astype(np.uint8)) == 1
        fused = sar | opt
        for nm, a in (("SAR", sar), ("fused", fused)):
            r = metrics(a, uf, dom)
            tp = (a & ins & (tsx == 1)).sum()
            rows.append({"definition": f"release window 15 Sep-30 Oct, >= {m} dates", "mask": nm, "area_km2": a.sum() / 1e4,
                         "unosat_precision": r["precision_vs_unosat"], "unosat_recall": r["recall_vs_unosat"], "unosat_f1": r["f1"],
                         "tsx_in_frame_km2": (a & ins).sum() / 1e4,
                         "tsx_f1": 2 * tp / max((a & ins).sum() + (tsx == 1).sum(), 1)})
        if m == 2:
            np.save("/tmp/_unused.npy", np.zeros(1)) if False else None
            prof = rasterio.open(fl / "fused_flood_confidence.tif").profile.copy()
            prof.update(count=1, dtype="uint8", nodata=255, compress="lzw")
            with rasterio.open(fl / "sar_flood_mask_releasewindow.tif", "w", **prof) as d:
                d.write(sar.astype(np.uint8), 1)
            removed = prim_sar & ~sar; added = sar & ~prim_sar
            log.info("M=2: removed %.1f km2 of the primary SAR flood, added %.1f km2", removed.sum() / 1e4, added.sum() / 1e4)
            grp = rasterio.open(fl / "flood_patch_groups.tif").read(1)
            for g, nme in ((5, "Ningo-Prampram isolated"), (6, "Ada West adjacent"), (3, "isolated >= 2 ha"), (1, "adjacent >= 2 ha")):
                log.info("   group %s: %.2f km2 of primary SAR flood, %.0f%% retained", nme, (prim_sar & (grp == g)).sum() / 1e4,
                         100 * (sar & (grp == g)).sum() / max((prim_sar & (grp == g)).sum(), 1))
    for nm, a in (("primary SAR (15 Sep-15 Nov, >= 2 dates)", prim_sar), ("primary fused", prim)):
        r = metrics(a, uf, dom); tp = (a & ins & (tsx == 1)).sum()
        rows.append({"definition": nm, "mask": "", "area_km2": a.sum() / 1e4, "unosat_precision": r["precision_vs_unosat"],
                     "unosat_recall": r["recall_vs_unosat"], "unosat_f1": r["f1"], "tsx_in_frame_km2": (a & ins).sum() / 1e4,
                     "tsx_f1": 2 * tp / max((a & ins).sum() + (tsx == 1).sum(), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(get_output_path("tables/release_window_variant.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
