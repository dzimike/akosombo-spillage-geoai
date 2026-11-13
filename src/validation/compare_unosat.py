from __future__ import annotations
"""
Comparison with the UNOSAT flood product FL20231018GHA (Sentinel-1, 23 October 2023).

UNOSAT (United Nations Satellite Centre) published an analyst-reviewed water extent over
Greater Accra and Volta Region (HDX, CC BY-SA; preliminary, not validated in the field):
  FloodExtent   53.4 km2  water outside the reference/permanent water
  WaterExtent  667.0 km2  all detected water
  AnalysisExtent 4,311 km2

Both products are Sentinel-1 based, so this is NOT a sensor-independent validation: it checks
our extent against an independent analyst-reviewed processing chain, on a different (single)
date. Comparison is on the 10 m grid, within the intersection of the UNOSAT analysis extent and
our raster domain. Our maps compared:
  fused       fused classes 1-3 (SAR or optical flood, primary definition)
  sar_flood   SAR flood mask (event-composite water outside permanent water)
  open_water  previous definition: event open water (VH/VV thresholds) or optical water, SAR pre-event water not removed
  scene_oct23 our SAR classification of the single 23 Oct 2023 scene, if built (s1_scene_20231023.py)
and, for total water, our event SAR water | permanent water | fused flood vs the UNOSAT WaterExtent.

Outputs: outputs/tables/unosat_comparison.csv, unosat_comparison_districts.csv,
         outputs/figures/fig_unosat_comparison.png (via narrow_paper_figures)
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from src.flood_mapping.water_masks import static_water
from src.utils.paths import get_project_root, get_output_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
UN = "data/raw/validation/unosat/shp/FL20231018GHA_SHP/S1_20231023_{}_GreaterAccra_VoltaRegion.shp"


def rd(p):
    with rasterio.open(p) as s:
        return s.read(1)


def metrics(pred, ref, dom):
    p, r = pred & dom, ref & dom
    tp, fp, fn = int((p & r).sum()), int((p & ~r).sum()), int((~p & r).sum())
    tn = int((dom & ~p & ~r).sum()); n = tp + fp + fn + tn
    po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return {"ours_km2": p.sum() / 1e4, "unosat_km2": r.sum() / 1e4, "overlap_km2": tp / 1e4,
            "precision_vs_unosat": tp / max(tp + fp, 1), "recall_vs_unosat": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "iou": tp / max(tp + fp + fn, 1), "kappa": (po - pe) / (1 - pe)}


def main() -> None:
    root = get_project_root()
    fl = root / "data/processed/flood_extent"
    with rasterio.open(fl / "fused_flood_confidence.tif") as s:
        fused, tr, crs, shape = s.read(1), s.transform, s.crs, s.shape
    def ras(name):
        g = gpd.read_file(root / UN.format(name)).to_crs(crs)
        return rasterize(((geom, 1) for geom in g.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    dom = ras("AnalysisExtent"); u_flood = ras("FloodExtent"); u_water = ras("WaterExtent")
    log.info("common domain %.0f km2; UNOSAT flood %.1f km2, water %.1f km2 inside it", dom.sum() / 1e4, (u_flood & dom).sum() / 1e4, (u_water & dom).sum() / 1e4)
    perm = fused == 4
    sea = rd(fl / "sea_mask.tif") == 1
    ours = {"fused (classes 1-3)": np.isin(fused, [1, 2, 3]), "SAR flood mask": rd(fl / "sar_flood_mask_binary.tif") == 1,
            "previous open-water rule (SAR open water | optical)": (rd(fl / "sar_openwater_flood_mask.tif") == 1)
            | ((rd(fl / "optical_water_mask.tif") == 1) & ~static_water(fl / "fused_flood_confidence.tif",
                                                                         root / "data/processed/satellite", fl)["union"])}
    scene = root / "data/processed/flood_extent/sar_flood_scene_20231023.tif"
    if scene.exists():
        ours["our 23 Oct scene (SAR flood)"] = rd(scene) == 1
    rows = []
    for k, m in ours.items():
        r = metrics(m, u_flood, dom); r.update(comparison=f"flood: {k} vs UNOSAT FloodExtent"); rows.append(r)
    r = {"comparison": "share of UNOSAT FloodExtent that lies in our permanent water (class 4)",
         "value": float((u_flood & dom & perm).sum() / max((u_flood & dom).sum(), 1))}; rows.append(r)
    r = {"comparison": "share of UNOSAT FloodExtent inside our fused flood (classes 1-3)",
         "value": float((u_flood & dom & np.isin(fused, [1, 2, 3])).sum() / max((u_flood & dom).sum(), 1))}; rows.append(r)
    r = {"comparison": "share of UNOSAT FloodExtent in our fused flood, SAR mask, or permanent water",
         "value": float((u_flood & dom & (np.isin(fused, [1, 2, 3]) | perm | ours['SAR flood mask'])).sum() / max((u_flood & dom).sum(), 1))}; rows.append(r)
    land_dom = dom & ~sea
    all_water = (rd(fl / "sar_event_water_mask.tif") == 1) | perm | np.isin(fused, [1, 2, 3])
    r = metrics(all_water, u_water, land_dom); r.update(comparison="total water (land): ours (event SAR water | permanent | flood) vs UNOSAT WaterExtent"); rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(get_output_path("tables/unosat_comparison.csv"), index=False)
    log.info("\n%s", df.round(3).to_string(index=False))

    # district comparison within the common domain
    gd = gpd.read_file(root / "data/raw/admin/gadm41_GHA.gpkg", layer="ADM_ADM_2").to_crs(crs)
    from shapely.geometry import box
    gd = gd[gd.intersects(box(tr.c, tr.f + shape[0] * tr.e, tr.c + shape[1] * tr.a, tr.f))].reset_index(drop=True)
    ids = rasterize(((geom, i + 1) for i, geom in enumerate(gd.geometry)), out_shape=shape, transform=tr, fill=0, dtype="int32")
    ufl = {"Ada East": 7.73, "Ada West": 1.42, "Ningo Prampram": 0.80, "Shai Osudoku": 2.65, "Anloga": 11.03, "Central Tongu": 3.95,
           "North Tongu": 8.74, "South Tongu": 15.47, "Keta Municipal": 0.51}
    drows = []
    for i, d in gd.iterrows():
        m = (ids == i + 1) & dom
        nm = d.NAME_2.replace("-", " ")
        drows.append({"district": d.NAME_2, "ours_fused_km2": (m & ours["fused (classes 1-3)"]).sum() / 1e4,
                      "unosat_geometry_km2": (m & u_flood).sum() / 1e4, "unosat_table_km2": ufl.get(nm, np.nan),
                      "both_km2": (m & ours["fused (classes 1-3)"] & u_flood).sum() / 1e4})
    dd = pd.DataFrame(drows); dd = dd[(dd.ours_fused_km2 > 0.3) | (dd.unosat_geometry_km2 > 0.3)].sort_values("unosat_geometry_km2", ascending=False)
    dd.to_csv(get_output_path("tables/unosat_comparison_districts.csv"), index=False)
    log.info("\n%s", dd.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
