from __future__ import annotations
"""
Land-use check of the unconfirmed lagoon-margin flood patches.

Question: are the small patches along the Songor lagoon (Ada East, Ada West, Ningo-Prampram) that overlap UNOSAT on < 3 %
associated with salt-production ponds?

Layers: OpenStreetMap salt ponds (landuse=salt_pond, 20 polygons, 36.5 km2; data/raw/landuse/osm_landuse_songor.gpkg) and
ESA WorldCover 2021 (10 m). Pixel groups: (1) 'unresolved' = pixels of connected components of the change-based flood
(classes 1-3) in Ada East, Ada West and Ningo-Prampram whose overlap with UNOSAT's flood is < 3 %; (2) the rest of the flood
extent; (3) background land (not flood, not pre-existing water, slope <= 5 deg) in the same three districts.
Statistics: share of each group inside an OSM salt pond (and within 30 m), WorldCover class composition, and, for the pond
polygons, the share of their dry area flagged.

Outputs: outputs/tables/patch_landuse_check.csv, patch_landuse_worldcover.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy import ndimage

from src.utils.paths import get_project_root, get_output_path
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
WC = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare / sparse vegetation",
      80: "Permanent water", 90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen", 0: "No data"}
DISTRICTS = ("Ada East", "Ada West", "Ningo-Prampram")


def main() -> None:
    root = get_project_root()
    fl, sat = root / "data/processed/flood_extent", root / "data/processed/satellite"
    with rasterio.open(fl / "fused_flood_confidence.tif") as s:
        fu, tr, crs, shape = s.read(1), s.transform, s.crs, s.shape
    flood = np.isin(fu, [1, 2, 3]); pre = fu == 4
    lab, n = ndimage.label(flood, structure=np.ones((3, 3)))
    diag = pd.read_csv(root / "outputs/tables/flood_patch_diagnostics.csv")
    assert len(diag) == n, (len(diag), n)
    sel = diag[diag.district.isin(DISTRICTS) & (diag.frac_unosat_flood * 1.0 < 0.03) & (diag.frac_in_unosat_domain > 0.5)]
    unres = np.isin(lab, sel.component.values)
    other = flood & ~unres
    log.info("unresolved: %d components, %.2f km2; rest of extent %.2f km2", len(sel), unres.sum() / 1e4, other.sum() / 1e4)
    osm = gpd.read_file(root / "data/raw/landuse/osm_landuse_songor.gpkg").to_crs(crs)
    sp = osm[osm.landuse == "salt_pond"]
    pond = rasterize(((g, 1) for g in sp.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    pond30 = ndimage.binary_dilation(pond, iterations=3)
    wet = osm[osm.natural == "wetland"]
    wetl = rasterize(((g, 1) for g in wet.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    wc = rasterio.open(sat / "worldcover_2021.tif").read(1)
    d = gpd.read_file(root / "data/raw/admin/ghana_districts.gpkg").to_crs(crs)
    d = d[d.NAME_2.isin(DISTRICTS)]
    dist = rasterize(((g, 1) for g in d.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), str(fl / "fused_flood_confidence.tif"), Resampling.bilinear)
    bg = dist & ~flood & ~pre & ~(np.isfinite(slope) & (slope > 5))
    rows = []
    for name, m in (("Unresolved lagoon-margin patches", unres), ("Rest of the flood extent", other), ("Background land, same districts", bg)):
        rows.append({"group": name, "area_km2": m.sum() / 1e4, "in_OSM_salt_pond_pct": 100 * (m & pond).sum() / m.sum(),
                     "within_30m_of_salt_pond_pct": 100 * (m & pond30).sum() / m.sum(), "in_OSM_wetland_pct": 100 * (m & wetl).sum() / m.sum()})
    df = pd.DataFrame(rows)
    # ponds themselves
    land_in_pond = pond & ~pre
    df.loc[len(df)] = {"group": "OSM salt-pond area not in pre-existing water", "area_km2": land_in_pond.sum() / 1e4, "in_OSM_salt_pond_pct": 100.0,
                       "within_30m_of_salt_pond_pct": 100.0, "in_OSM_wetland_pct": np.nan}
    df["flagged_pct_of_group"] = [np.nan, np.nan, np.nan, 100 * (land_in_pond & flood).sum() / land_in_pond.sum()]
    df.to_csv(get_output_path("tables/patch_landuse_check.csv"), index=False)
    log.info("\n%s", df.round(2).to_string(index=False))
    comp = {}
    for name, m in (("Unresolved lagoon-margin patches", unres), ("Rest of the flood extent", other), ("Background land, same districts", bg),
                    ("Unresolved patches inside salt ponds", unres & pond30), ("Unresolved patches outside salt ponds", unres & ~pond30)):
        v = wc[m]
        comp[name] = {WC.get(k, str(k)): 100 * (v == k).sum() / max(len(v), 1) for k in np.unique(wc[m | bg])}
    cdf = pd.DataFrame(comp).fillna(0).round(1)
    cdf.to_csv(get_output_path("tables/patch_landuse_worldcover.csv"))
    log.info("\n%s", cdf.to_string())


if __name__ == "__main__":
    main()
