from __future__ import annotations
"""
Assess SAR flood detection on the rebuilt full-coverage Sentinel-1 composites.

Does NOT overwrite the production flood labels. It
  1. runs the original change-detection recipe (Otsu on the VV dB difference),
  2. runs a backscatter water classification (Otsu on event VV / VH within the
     non-permanent-water, slope <= 5 deg domain),
  3. compares each to the current optical-derived flood mask (fused classes 2+3)
     and builds a candidate fused map (classes 1/2/3/4) written to
     data/processed/flood_extent/candidate_sar_rebuild/,
and writes outputs/tables/sar_rebuild_assessment.csv.
"""
import logging

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage
from skimage.filters import threshold_otsu

from src.utils.paths import get_project_root, get_output_path
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
MMU, SLOPE_MAX = 10, 5.0


def rd(p):
    with rasterio.open(p) as s:
        return s.read(1)


def mmu(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= MMU
    keep[0] = False
    return keep[lab]


def score(sar: np.ndarray, opt: np.ndarray, domain: np.ndarray) -> dict:
    s, o = sar & domain, opt & domain
    tp = (s & o).sum()
    return {"sar_km2": s.sum() / 1e4, "optical_km2": o.sum() / 1e4, "overlap_km2": tp / 1e4,
            "precision_vs_optical": tp / max(s.sum(), 1), "recall_vs_optical": tp / max(o.sum(), 1),
            "f1_vs_optical": 2 * tp / max(s.sum() + o.sum(), 1), "iou_vs_optical": tp / max((s | o).sum(), 1)}


def main() -> None:
    root = get_project_root()
    S, F = root / "data/processed/satellite", root / "data/processed/flood_extent"
    fused = rd(F / "fused_flood_confidence.tif")
    perm = fused == 4
    opt = np.isin(fused, [2, 3])
    ref = S / "s1_pre_event_vv.tif"
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), str(ref), Resampling.bilinear)
    flat = ~(np.isfinite(slope) & (slope > SLOPE_MAX))
    pre, ev = rd(S / "s1_pre_event_vv.tif"), rd(S / "s1_event_vv.tif")
    evh = rd(S / "s1_event_vh.tif")
    domain = ~perm & flat & np.isfinite(pre) & np.isfinite(ev)
    log.info("domain (non-permanent, slope<=%.0f): %.0f km2", SLOPE_MAX, domain.sum() / 1e4)

    d = ev - pre
    methods = {}
    t_chg = threshold_otsu(d[::5, ::5][np.isfinite(d[::5, ::5])])
    methods["change_otsu_vv"] = (d < t_chg, t_chg)
    t_vv = threshold_otsu(ev[domain][::5])
    methods["backscatter_otsu_event_vv"] = (ev < t_vv, t_vv)
    t_vh = threshold_otsu(evh[domain & np.isfinite(evh)][::5])
    methods["backscatter_otsu_event_vh"] = (evh < t_vh, t_vh)
    both = (ev < t_vv) & (evh < t_vh)
    methods["backscatter_vv_and_vh"] = (both, np.nan)
    chg_and_bs = (d < -1.0) & (ev < t_vv)
    methods["new_water_vv<otsu_and_change<-1dB"] = (chg_and_bs, -1.0)

    rows = []
    for name, (m, thr) in methods.items():
        m = mmu(m & domain)
        r = score(m, opt, domain); r.update(method=name, threshold_dB=float(thr)); rows.append(r)
        log.info("%-36s thr %7.2f dB | SAR %7.1f km2 | optical %6.1f | overlap %6.1f | P %.3f R %.3f F1 %.3f",
                 name, thr, r["sar_km2"], r["optical_km2"], r["overlap_km2"],
                 r["precision_vs_optical"], r["recall_vs_optical"], r["f1_vs_optical"])
        if name == "backscatter_vv_and_vh":
            sar_best = m
    # unconditional pre-event state of optical-flood pixels
    rows.append({"method": "context: mean VV (dB) event/pre on optical flood",
                 "sar_km2": float(np.nanmean(ev[opt])), "optical_km2": float(np.nanmean(pre[opt]))})

    # candidate fused map with the backscatter-Otsu SAR mask
    cand = np.zeros(fused.shape, np.uint8)
    cand[sar_best & ~opt] = 1
    cand[opt & ~sar_best] = 2
    cand[opt & sar_best] = 3
    cand[perm] = 4
    out = F / "candidate_sar_rebuild"; out.mkdir(exist_ok=True)
    with rasterio.open(F / "fused_flood_confidence.tif") as s:
        prof = s.profile
    with rasterio.open(out / "fused_flood_confidence_candidate.tif", "w", **prof) as dst:
        dst.write(cand, 1)
    area = lambda m: float(m.sum() / 1e4)
    cmp_rows = [{"map": "current_fused", **{f"class_{c}_km2": area(fused == c) for c in (1, 2, 3, 4)},
                 "flooded_1_2_3_km2": area(np.isin(fused, [1, 2, 3]))},
                {"map": "candidate_backscatter_vv_and_vh", **{f"class_{c}_km2": area(cand == c) for c in (1, 2, 3, 4)},
                 "flooded_1_2_3_km2": area(np.isin(cand, [1, 2, 3]))}]
    log.info("class areas:\n%s", pd.DataFrame(cmp_rows).round(1).to_string(index=False))
    pd.DataFrame(rows).to_csv(get_output_path("tables/sar_rebuild_assessment.csv"), index=False)
    pd.DataFrame(cmp_rows).to_csv(get_output_path("tables/sar_rebuild_candidate_fusion.csv"), index=False)


if __name__ == "__main__":
    main()
