from __future__ import annotations
"""
Diagnose the scatter of small flood patches in the change-based extent.

Every connected component (8-neighbour) of the fused flood (classes 1-3) is described by area, distance to
pre-existing water, pre-event NDVI, slope, elevation, UNOSAT overlap (where UNOSAT's analysis extent covers it),
bounding-box fill (a rectangle-like field gives ~1) and district. Components are grouped into
'river/lagoon-adjacent' (within 150 m of pre-existing water) and 'isolated', and by size.

Output: outputs/tables/flood_patch_diagnostics.csv (one row per component), flood_patch_summary.csv (by district and group)
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage

from src.utils.paths import get_project_root, get_output_path
from src.validation.compare_unosat import UN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    root = get_project_root()
    fl, sat = root / "data/processed/flood_extent", root / "data/processed/satellite"
    rd = lambda p: rasterio.open(p).read(1)
    with rasterio.open(fl / "fused_flood_confidence.tif") as s:
        fu, tr, crs, shape = s.read(1), s.transform, s.crs, s.shape
    flood = np.isin(fu, [1, 2, 3])
    lab, n = ndimage.label(flood, structure=np.ones((3, 3)))
    log.info("components: %d", n)
    idx = np.arange(1, n + 1)
    area = ndimage.sum(flood, lab, idx)
    sl = ndimage.find_objects(lab)
    bbox = np.array([(s_[0].stop - s_[0].start) * (s_[1].stop - s_[1].start) for s_ in sl], float)
    dist = ndimage.distance_transform_edt(fu != 4) * 10          # m to pre-existing water
    ndvi = rd(sat / "s2_ndvi_pre.tif")
    from rasterio.enums import Resampling
    from src.utils.raster import read_aligned
    ref = str(fl / "fused_flood_confidence.tif")
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), ref, Resampling.bilinear)
    elev = read_aligned(str(root / "data/processed/dem/elevation.tif"), ref, Resampling.bilinear)
    with rasterio.open(sat / "s1_change_stack.tif") as cs:
        cnt3, nobs = cs.read(2), cs.read(4)
    sar = rd(fl / "sar_flood_mask_binary.tif") == 1
    ow = rd(fl / "sar_openwater_flood_mask.tif") == 1
    def ras(name):
        g = gpd.read_file(root / UN.format(name)).to_crs(crs)
        return rasterize(((geom, 1) for geom in g.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    dom, uf = ras("AnalysisExtent"), ras("FloodExtent")
    tsx_p = root / "data/processed/validation/charter_terrasarx_20231016.tif"
    tsx = rd(tsx_p) if tsx_p.exists() else None
    # district per pixel (GADM level 2 via the raster grid)
    d = gpd.read_file(root / "data/raw/admin/ghana_districts.gpkg").to_crs(crs)
    d = d[d.intersects(gpd.GeoSeries.from_xy([tr.c + shape[1] * tr.a / 2], [tr.f + shape[0] * tr.e / 2], crs=crs).buffer(60000).iloc[0])]
    name_col = "NAME_2" if "NAME_2" in d.columns else [c for c in d.columns if "name" in c.lower()][0]
    dist_id = np.zeros(shape, np.int16)
    names = list(d[name_col])
    for i, g in enumerate(d.geometry, 1):
        dist_id[rasterize([(g, i)], out_shape=shape, transform=tr, fill=0, dtype="int16") == i] = i
    cm = lambda a: ndimage.mean(a, lab, idx)
    area_px = area
    rows = pd.DataFrame({
        "component": idx, "area_ha": area * 0.01,
        "bbox_fill": area / bbox,
        "dist_to_preexisting_water_m_min": ndimage.minimum(dist, lab, idx),
        "ndvi_pre_mean": cm(np.where(np.isfinite(ndvi), ndvi, np.nan)),
        "slope_mean": cm(np.where(np.isfinite(slope), slope, 0)),
        "elev_mean": cm(np.where(np.isfinite(elev), elev, 0)),
        "cnt3_mean": cm(cnt3), "n_obs_mean": cm(nobs), "frac_sar_change": cm(sar.astype(float)), "frac_openwater": cm(ow.astype(float)),
        "frac_in_unosat_domain": cm(dom.astype(float)),
        "frac_unosat_flood": cm((uf & dom).astype(float)),
        "district_id": ndimage.maximum(dist_id, lab, idx),
    })
    if tsx is not None:
        rows["frac_in_tsx_frame"] = cm((tsx != 255).astype(float)); rows["frac_tsx_flooded"] = cm((tsx == 1).astype(float))
    rows["district"] = [names[i - 1] if i > 0 else "outside" for i in rows.district_id]
    rows["group"] = np.where(rows.dist_to_preexisting_water_m_min <= 150, "adjacent to pre-existing water", "isolated")
    rows["size_class"] = pd.cut(rows.area_ha, [0, 0.1, 0.5, 2, 10, 1e9], labels=["<0.1 ha", "0.1-0.5", "0.5-2", "2-10", ">10 ha"])
    rows.to_csv(get_output_path("tables/flood_patch_diagnostics.csv"), index=False)
    # labelled raster of patch groups for the per-date time series:
    # 1 adjacent >= 2 ha, 2 adjacent < 2 ha, 3 isolated >= 2 ha, 4 isolated < 2 ha, 5 Ningo-Prampram isolated, 6 Ada West adjacent
    code = np.where(rows.group.str.startswith("adjacent"), np.where(rows.area_ha >= 2, 1, 2), np.where(rows.area_ha >= 2, 3, 4))
    code = np.where((rows.district == "Ningo-Prampram") & (rows.group == "isolated"), 5, code)
    code = np.where((rows.district == "Ada West") & (rows.group.str.startswith("adjacent")), 6, code).astype(np.uint8)
    lut = np.zeros(n + 1, np.uint8); lut[1:] = code
    with rasterio.open(fl / "fused_flood_confidence.tif") as s_:
        prof = s_.profile.copy()
    prof.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(root / "data/processed/flood_extent/flood_patch_groups.tif", "w", **prof) as dst:
        dst.write(lut[lab], 1)
    tot = rows.area_ha.sum()
    rows["date_share"] = rows.cnt3_mean / rows.n_obs_mean
    g = rows.groupby("group").agg(patches=("area_ha", "size"), area_ha=("area_ha", "sum"), median_ha=("area_ha", "median"),
                                  ndvi=("ndvi_pre_mean", "mean"), bbox_fill=("bbox_fill", "median"), date_share=("date_share", "mean"))
    g["share_of_area"] = g.area_ha / tot
    log.info("\n%s", g.round(3).to_string())
    s2 = rows.groupby("size_class", observed=True).apply(lambda x: pd.Series({
        "patches": len(x), "area_km2": x.area_ha.sum() / 100, "share_area": x.area_ha.sum() / tot,
        "unosat_overlap_in_domain": (x.frac_unosat_flood * x.area_ha * x.frac_in_unosat_domain).sum() / max((x.frac_in_unosat_domain * x.area_ha).sum(), 1e-9)}))
    log.info("\n%s", s2.round(3).to_string())
    s3 = rows.groupby(["district", "group"]).apply(lambda x: pd.Series({
        "patches": len(x), "area_km2": x.area_ha.sum() / 100, "median_ha": x.area_ha.median(),
        "unosat_overlap_in_domain": (x.frac_unosat_flood * x.area_ha).sum() / max((x.frac_in_unosat_domain * x.area_ha).sum(), 1e-9),
        "ndvi_pre": x.ndvi_pre_mean.mean(), "date_share": x.date_share.mean(), "frac_in_unosat_domain": (x.frac_in_unosat_domain * x.area_ha).sum() / x.area_ha.sum()})).reset_index()
    s3.to_csv(get_output_path("tables/flood_patch_summary.csv"), index=False)
    log.info("\n%s", s3[s3.area_km2 >= 0.5].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
