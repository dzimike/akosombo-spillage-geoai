from __future__ import annotations
"""
Why does our extent disagree with the UNOSAT FL20231018GHA flood extent?

1. Classify the single 23 Oct 2023 scene with OUR rule (VH < -26.7 dB and VV < -15.2 dB, outside permanent
   water, slope <= 5 deg, minimum mapping unit) -> sar_flood_scene_20231023.tif (same date as UNOSAT).
2. Backscatter, pre-event backscatter and optical indices on UNOSAT flood pixels versus on land not mapped
   as water by UNOSAT.
3. Single-band thresholds that best separate UNOSAT flood from dry land in the 23 Oct scene (Youden J), and
   the F1 against UNOSAT obtained by applying them, as a check on whether the disagreement is mainly a matter
   of threshold strictness.

Outputs: data/processed/flood_extent/sar_flood_scene_20231023.tif, outputs/tables/unosat_diagnostics.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize

from src.flood_mapping.sar_change_detection import remove_small_patches
from src.flood_mapping.water_masks import static_water
from src.utils.paths import get_project_root, get_output_path
from src.utils.raster import read_aligned
from src.validation.compare_unosat import UN, metrics, rd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def youden(flood_vals, land_vals, lo=-35.0, hi=0.0):
    ts = np.linspace(lo, hi, 351); f, l = np.sort(flood_vals), np.sort(land_vals)
    j = np.searchsorted(f, ts) / len(f) - np.searchsorted(l, ts) / len(l)
    i = int(np.argmax(j)); return float(ts[i]), float(j[i])


def main() -> None:
    root = get_project_root()
    sat, fl = root / "data/processed/satellite", root / "data/processed/flood_extent"
    ref = sat / "s1_pre_event_vv.tif"
    with rasterio.open(ref) as s:
        tr, crs, shape, prof = s.transform, s.crs, s.shape, s.profile.copy()
    vv, vh = rd(sat / "s1_scene_20231023_vv.tif"), rd(sat / "s1_scene_20231023_vh.tif")
    fused = rd(fl / "fused_flood_confidence.tif"); perm = fused == 4
    static = static_water(ref, sat, fl)["union"]
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), str(ref), Resampling.bilinear)
    flat = ~(np.isfinite(slope) & (slope > 5))
    rep = pd.read_csv(fl / "sar_threshold_report.csv").set_index("metric")["value"]
    t_vh, t_vv = float(rep["threshold_vh_dB_youden"]), float(rep["threshold_vv_dB_youden"])
    ours = (vh < t_vh) & (vv < t_vv) & ~static & flat
    ours = remove_small_patches(ours.astype(np.uint8)) == 1
    p = prof.copy(); p.update(dtype="uint8", count=1, nodata=255, compress="lzw")
    with rasterio.open(fl / "sar_flood_scene_20231023.tif", "w", **p) as d:
        d.write(ours.astype(np.uint8), 1)
    log.info("our 23 Oct scene flood: %.1f km2", ours.sum() / 1e4)

    def ras(name):
        g = gpd.read_file(root / UN.format(name)).to_crs(crs)
        return rasterize(((geom, 1) for geom in g.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    dom = ras("AnalysisExtent") & np.isfinite(vv); uf = ras("FloodExtent") & dom; uw = ras("WaterExtent") & dom
    sea = rd(fl / "sea_mask.tif") == 1
    dom &= ~sea
    rows = []
    r = metrics(ours, uf, dom); r.update(item="our 23 Oct scene (our thresholds) vs UNOSAT flood"); rows.append(r)

    pre_vv, pre_vh = rd(sat / "s1_pre_event_vv.tif"), rd(sat / "s1_pre_event_vh.tif")
    ev_vv, ev_vh = rd(sat / "s1_event_vv.tif"), rd(sat / "s1_event_vh.tif")
    ndwi, mndwi = rd(sat / "s2_ndwi_event.tif"), rd(sat / "s2_mndwi_event.tif")
    dry = dom & ~uw & ~perm & flat
    sets = {"UNOSAT flood": uf & ~perm, "UNOSAT flood in our fused flood": uf & np.isin(fused, [1, 2, 3]),
            "UNOSAT flood NOT in our flood/permanent water": uf & ~perm & ~np.isin(fused, [1, 2, 3]), "dry land (not UNOSAT water)": dry}
    stats = []
    for nm, m in sets.items():
        o = {"set": nm, "area_km2": m.sum() / 1e4}
        for lab, a in (("scene_vv", vv), ("scene_vh", vh), ("pre_vv", pre_vv), ("pre_vh", pre_vh), ("event_comp_vv", ev_vv), ("event_comp_vh", ev_vh)):
            o[f"{lab}_median"] = float(np.nanmedian(a[m])) if m.any() else np.nan
        o["frac_ndwi_ge0"] = float(np.nanmean(ndwi[m] >= 0)) if m.any() else np.nan
        o["frac_mndwi_ge0"] = float(np.nanmean(mndwi[m] >= 0)) if m.any() else np.nan
        stats.append(o)
    st = pd.DataFrame(stats)
    # implied single-band thresholds
    uf_np = uf & ~perm
    for lab, a in (("VV", vv), ("VH", vh)):
        t, j = youden(a[uf_np & np.isfinite(a)][::3], a[dry & np.isfinite(a)][::10])
        pred = (a < t) & ~static & flat & dom
        pred = remove_small_patches(pred.astype(np.uint8)) == 1
        m = metrics(pred, uf, dom); m.update(item=f"scene {lab} < {t:.1f} dB (Youden J {j:.2f} vs UNOSAT flood)"); rows.append(m)
    t_vv, _ = youden(vv[uf_np & np.isfinite(vv)][::3], vv[dry & np.isfinite(vv)][::10])
    t_vh, _ = youden(vh[uf_np & np.isfinite(vh)][::3], vh[dry & np.isfinite(vh)][::10])
    pred = remove_small_patches(((vv < t_vv) & (vh < t_vh) & ~static & flat & dom).astype(np.uint8)) == 1
    m = metrics(pred, uf, dom); m.update(item=f"scene VV < {t_vv:.1f} and VH < {t_vh:.1f}"); rows.append(m)
    res = pd.DataFrame(rows)
    res.to_csv(get_output_path("tables/unosat_comparison_same_date.csv"), index=False)
    st.to_csv(get_output_path("tables/unosat_diagnostics.csv"), index=False)
    log.info("\n%s", res.round(3).to_string(index=False))
    log.info("\n%s", st.round(2).T.to_string())


if __name__ == "__main__":
    main()
