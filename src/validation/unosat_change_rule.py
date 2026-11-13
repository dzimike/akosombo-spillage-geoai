from __future__ import annotations
"""
Is the UNOSAT extent consistent with a change-detection rule (event scene minus pre-event composite)?

Applies (a) the conventional untuned rules dVV < -3 dB, dVH < -3 dB, and both, and (b) Youden-optimal
single-band change thresholds against the UNOSAT flood (a diagnostic, not a validation: tuned on the
reference), to the 23 Oct 2023 scene minus our pre-event composite, outside permanent water, slope <= 5 deg,
with the 10-pixel minimum mapping unit. Scores against UNOSAT FloodExtent in the common domain.

Output: outputs/tables/unosat_change_rule.csv
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
from src.validation.unosat_diagnostics import youden

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    root = get_project_root()
    sat, fl = root / "data/processed/satellite", root / "data/processed/flood_extent"
    ref = sat / "s1_pre_event_vv.tif"
    with rasterio.open(ref) as s:
        tr, crs, shape = s.transform, s.crs, s.shape
    dvv = rd(sat / "s1_scene_20231023_vv.tif") - rd(sat / "s1_pre_event_vv.tif")
    dvh = rd(sat / "s1_scene_20231023_vh.tif") - rd(sat / "s1_pre_event_vh.tif")
    fused = rd(fl / "fused_flood_confidence.tif")
    static = static_water(ref, sat, fl)["union"]
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), str(ref), Resampling.bilinear)
    flat = ~(np.isfinite(slope) & (slope > 5))
    def ras(name):
        g = gpd.read_file(root / UN.format(name)).to_crs(crs)
        return rasterize(((geom, 1) for geom in g.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    sea = rd(fl / "sea_mask.tif") == 1
    dom = ras("AnalysisExtent") & np.isfinite(dvv) & np.isfinite(dvh) & ~sea
    uf = ras("FloodExtent") & dom; uw = ras("WaterExtent") & dom
    dry = dom & ~uw & ~static & flat
    ok = ~static & flat
    def score(mask, name):
        m = remove_small_patches((mask & ok & dom).astype(np.uint8)) == 1
        r = metrics(m, uf, dom); r["rule"] = name; return r
    rows = [score(dvv < -3, "dVV < -3 dB (conventional, untuned)"), score(dvh < -3, "dVH < -3 dB (conventional, untuned)"),
            score((dvv < -3) & (dvh < -3), "dVV < -3 and dVH < -3 (untuned)")]
    uf_np = uf & ~static
    for nm, a in (("dVV", dvv), ("dVH", dvh)):
        t, j = youden(a[uf_np][::3], a[dry][::10], lo=-15, hi=5)
        rows.append(score(a < t, f"{nm} < {t:.1f} dB (Youden J {j:.2f}, tuned on UNOSAT)"))
    df = pd.DataFrame(rows)
    df.to_csv(get_output_path("tables/unosat_change_rule.csv"), index=False)
    log.info("\n%s", df.round(3).to_string(index=False))
    log.info("median change on UNOSAT flood: dVV %.1f dVH %.1f | dry land: dVV %.1f dVH %.1f dB",
             np.nanmedian(dvv[uf_np]), np.nanmedian(dvh[uf_np]), np.nanmedian(dvv[dry]), np.nanmedian(dvh[dry]))


if __name__ == "__main__":
    main()
