from __future__ import annotations
"""
Sensitivity of two exposure quantities to how they are measured.

Population. The main estimate spreads WorldPop 2020 1 km (UN-adjusted) counts evenly over each 1 km cell and sums them over flood pixels. Two
alternatives: (a) WorldPop 2020 100 m (unconstrained; Earth Engine WorldPop/GP/100m/pop), rescaled to the same domain total and spread evenly over each 100 m
cell; (b) the
same 100 m counts allocated to the OSM building footprints inside each 100 m cell in proportion to footprint area (cells without buildings keep
their count spread evenly), summed over buildings whose centroid is in the flood extent plus the unallocated count over flood pixels.

Cropland. The main estimate intersects the extent with pre-event NDVI >= 0.35, which marks vegetated land, not cropland. Alternatives: ESA WorldCover
2021 cropland (class 40), and its intersection with the NDVI mask.

Output: outputs/tables/exposure_sensitivity.csv
"""
import io
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize

from src.preprocessing.preprocess_sentinel2 import init_gee
from src.utils.paths import get_project_root, load_config
from src.utils.raster import read_aligned
from src.exposure.district_exposure import POP_SCALE
from src.validation.threshold_exposure_direct import reproject_population_conserving

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def worldpop_100m(ee, x0, y1, nx, ny):
    img = ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("country", "GHA")).filter(ee.Filter.eq("year", 2020)).first().unmask(0)
    req = {"expression": img.toFloat(), "fileFormat": "NPY",
           "grid": {"dimensions": {"width": nx, "height": ny},
                    "affineTransform": {"scaleX": 100, "shearX": 0, "translateX": x0, "shearY": 0, "scaleY": -100, "translateY": y1}, "crsCode": "EPSG:32630"}}
    a = np.load(io.BytesIO(ee.data.computePixels(req)))
    return a[a.dtype.names[0]].astype(np.float64) if a.dtype.names else a.astype(np.float64)


def main() -> None:
    root = get_project_root()
    ref_p = root / "data/processed/flood_extent/fused_flood_confidence.tif"
    with rasterio.open(ref_p) as s:
        fused = s.read(1); tr = s.transform; crs = s.crs
    H, W = fused.shape
    flood = np.isin(fused, [1, 2, 3])
    land = ~np.isin(fused, [4, 255])
    rows = []
    # ---------------------------------------------------------------- population
    pop1k = reproject_population_conserving(root / "data/processed/population/population_grid_clipped.tif", ref_p) * POP_SCALE
    in_dom = fused != 255
    total = float(pop1k[in_dom].sum())
    rows.append(("population", "WorldPop 1 km UN-adjusted, even spread (main)", float(pop1k[flood].sum()), total))
    ee = init_gee(load_config(root / "config/gee_config.yaml"))
    ny, nx = -(-H // 10), -(-W // 10)
    p100 = worldpop_100m(ee, tr.c, tr.f, nx, ny)
    Hp, Wp = ny * 10, nx * 10
    fpad = np.zeros((Hp, Wp), bool); fpad[:H, :W] = flood
    dom = np.zeros((Hp, Wp), bool); dom[:H, :W] = in_dom
    frac_f = fpad.reshape(ny, 10, nx, 10).mean((1, 3))
    frac_d = dom.reshape(ny, 10, nx, 10).mean((1, 3))
    raw100 = float((p100 * frac_d).sum())
    log.info("WorldPop 100 m domain total %.0f, rescaled to the 1 km UN-adjusted total %.0f", raw100, total)
    p100 = p100 * total / raw100   # same domain total, so only the allocation differs
    rows.append(("population", "WorldPop 100 m unconstrained, even spread", float((p100 * frac_f).sum()), float((p100 * frac_d).sum())))
    b = gpd.read_file(root / "data/raw/buildings/osm_buildings.gpkg").to_crs(crs)
    b = b[b.geometry.notna() & ~b.geometry.is_empty].copy()
    b["area"] = b.geometry.area
    c = b.geometry.centroid
    col = ((c.x - tr.c) // 100).astype(int); row = ((tr.f - c.y) // 100).astype(int)
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    b, col, row, c = b[ok], col[ok], row[ok], c[ok]
    cell = row * nx + col
    area_cell = np.bincount(cell, weights=b["area"], minlength=ny * nx).reshape(ny, nx)
    has_b = area_cell > 0
    b["pop"] = p100.ravel()[cell] * b["area"].values / area_cell.ravel()[cell]
    pc = ((tr.f - c.y) // 10).astype(int).clip(0, H - 1); pcc = ((c.x - tr.c) // 10).astype(int).clip(0, W - 1)
    b_fl = flood[pc.values, pcc.values]
    unalloc = np.where(has_b, 0, p100)
    dasy = float(b.loc[b_fl, "pop"].sum() + (unalloc * frac_f).sum())
    rows.append(("population", "WorldPop 100 m allocated to OSM building footprints", dasy, float(b["pop"].sum() + (unalloc * frac_d).sum())))
    log.info("buildings used %d, exposed %d", len(b), int(b_fl.sum()))
    # ---------------------------------------------------------------- cropland
    ndvi = read_aligned(str(root / "data/processed/satellite/s2_ndvi_pre.tif"), str(ref_p), Resampling.nearest)
    veg = np.isfinite(ndvi) & (ndvi >= 0.35)
    wc = read_aligned(str(root / "data/processed/satellite/worldcover_2021.tif"), str(ref_p), Resampling.nearest)
    crop = wc == 40
    for name, m in (("pre-event NDVI >= 0.35 (vegetated land; main)", veg), ("ESA WorldCover 2021 cropland", crop), ("WorldCover cropland and NDVI >= 0.35", crop & veg)):
        rows.append(("cropland_ha", name, float((m & flood).sum() / 100), float((m & land).sum() / 100)))
    df = pd.DataFrame(rows, columns=["quantity", "method", "exposed", "domain_total"])
    df["exposed_pct"] = (100 * df.exposed / df.domain_total).round(2)
    df[["exposed", "domain_total"]] = df[["exposed", "domain_total"]].round(0)
    out = root / "outputs/tables/exposure_sensitivity.csv"
    df.to_csv(out, index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
