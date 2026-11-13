from __future__ import annotations
"""
Agreement of every flood definition with every independent reference (precision, recall, F1).

Rows: open-water rule, optical mask, single-scene change, event-window two-date extent (archived), SAR change mask,
and the release-period change grid K = 2, 3, 4 dB x M = 1, 2, 3 dates (each fused with optical water outside
pre-existing water).
Columns: UNOSAT FloodExtent 23 Oct (common domain), Charter TerraSAR-X 16 Oct (map frame, as published),
Charter TanDEM-X 20 Oct (analysis polygon, after the permanent-water-based translation), Mepe flooded buildings
(building-level: footprint centroid within 30 m of the extent vs the 179 flagged buildings).

Output: outputs/tables/definition_reference_matrix.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy import ndimage

from src.flood_mapping.change_counts import count_flags
from src.flood_mapping.sar_change_detection import remove_small_patches
from src.flood_mapping.water_masks import static_water
from src.utils.paths import get_project_root, get_output_path
from src.utils.raster import read_aligned
from src.validation.compare_unosat import UN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def prf(pred, ref, dom):
    p, r = pred & dom, ref & dom
    tp = (p & r).sum()
    prec = tp / max(p.sum(), 1); rec = tp / max(r.sum(), 1)
    return prec, rec, 2 * tp / max(p.sum() + r.sum(), 1)


def main() -> None:
    root = get_project_root()
    sat, fl = root / "data/processed/satellite", root / "data/processed/flood_extent"
    rd = lambda p: rasterio.open(p).read(1)
    with rasterio.open(fl / "fused_flood_confidence.tif") as s:
        fu, tr, crs, shape = s.read(1), s.transform, s.crs, s.shape
    ref_path = str(fl / "fused_flood_confidence.tif")
    pre = fu == 4
    static = static_water(sat / "s1_pre_event_vv.tif", sat, fl)["union"]
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), ref_path, Resampling.bilinear)
    flat = ~(np.isfinite(slope) & (slope > 5))
    opt = rd(fl / "optical_water_mask.tif") == 1
    opt_np = opt & ~pre
    ow = rd(fl / "sar_openwater_flood_mask.tif") == 1
    # references
    def ras(name):
        g = gpd.read_file(root / UN.format(name)).to_crs(crs)
        return rasterio.features.rasterize(((geom, 1) for geom in g.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    dom_u, uf = ras("AnalysisExtent"), ras("FloodExtent")
    tsx = rd(root / "data/processed/validation/charter_terrasarx_20231016.tif")
    tdx_raw = rd(root / "data/processed/validation/charter_tandemx_20231020.tif")
    roll = lambda a: np.roll(np.roll(a, -36, 0), 56, 1)
    tdx_dom = roll(tdx_raw != 255); tdx_fl = roll(tdx_raw == 1) & tdx_dom
    refs = {"UNOSAT 23 Oct": (uf, dom_u), "TerraSAR-X 16 Oct": (tsx == 1, tsx != 255), "TanDEM-X 20 Oct": (tdx_fl, tdx_dom)}
    # Mepe buildings
    mb = gpd.read_file(root / "data/processed/validation/charter_mepe_flooded_buildings.gpkg").to_crs(crs)
    c = mb.geometry.centroid
    rr, cc = rasterio.transform.rowcol(tr, c.x.values, c.y.values); rr, cc = np.array(rr), np.array(cc)
    flagged = mb.charter_affected.values
    r0, r1, c0, c1 = rr.min() - 60, rr.max() + 60, cc.min() - 60, cc.max() + 60
    def mepe(mask):
        near = ndimage.binary_dilation(mask[r0:r1, c0:c1], iterations=3)
        pred = near[rr - r0, cc - c0]
        tp = (pred & flagged).sum()
        prec = tp / max(pred.sum(), 1); rec = tp / flagged.sum()
        return prec, rec, 2 * tp / max(pred.sum() + flagged.sum(), 1)
    # definitions
    scene_ow = rd(fl / "sar_flood_scene_20231023.tif") == 1
    scene = rd(fl / "sar_flood_scene_change_20231023.tif") == 1
    old_fused = np.isin(rd(root / "archive/v6_event_window_persistence/flood_extent/fused_flood_confidence.tif"), [1, 2, 3])
    sar_primary = rd(fl / "sar_flood_mask_binary.tif") == 1
    defs = [("Open-water rule (event water outside static water)", "Open-water definition", (ow | (opt & ~static)), "openwater"),
            ("Optical water mask alone", "Open-water definition", opt_np, "openwater"),
            ("Open-water rule on the 23 Oct scene", "Open-water definition", scene_ow, "openwater"),
            ("Single scene 23 Oct, change rule (dVV, dVH < -3 dB)", "Single scene", scene, "scene"),
            ("Event window 15 Sep-15 Nov, K=3 dB, >= 2 dates (fused)", "Event window", old_fused, "event_window"),
            ("SAR change mask alone (K=3 dB, >= 2 release dates)", "Release period", sar_primary, "change")]
    for k in (2, 3, 4):
        cnt = count_flags(sat, k)[0]
        for m in (1, 2, 3):
            ch = remove_small_patches(((cnt >= m) & ~pre & flat).astype(np.uint8)) == 1
            tag = " [primary]" if (k, m) == (3, 2) else ""
            defs.append((f"Release period, K={k} dB, >= {m} dates (fused){tag}", "Release period", ch | opt_np, "change"))
    rows = []
    for name, group, mask, key in defs:
        row = {"definition": name, "group": group, "key": key, "area_km2": mask.sum() / 1e4}
        for rn, (rf, dm) in refs.items():
            p, r, f = prf(mask, rf, dm)
            row.update({f"{rn} precision": p, f"{rn} recall": r, f"{rn} F1": f})
        p, r, f = mepe(mask)
        row.update({"Mepe buildings precision": p, "Mepe buildings recall": r, "Mepe buildings F1": f})
        rows.append(row)
        log.info("%-62s %6.1f km2  UNOSAT %.2f  TSX %.2f  TDX %.3f  Mepe %.3f", name, row["area_km2"], row["UNOSAT 23 Oct F1"],
                 row["TerraSAR-X 16 Oct F1"], row["TanDEM-X 20 Oct F1"], row["Mepe buildings F1"])
    pd.DataFrame(rows).to_csv(get_output_path("tables/definition_reference_matrix.csv"), index=False)


if __name__ == "__main__":
    main()
