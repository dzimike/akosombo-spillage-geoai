from __future__ import annotations
"""
Exposure under the UNOSAT FL20231018GHA flood extent and under a conventional change rule, on the same
layers and sampling as label_definition_exposure.py, for comparison with our primary extent.

Rows (restricted to the UNOSAT analysis extent so the areas are comparable):
  our primary fused extent | UNOSAT FloodExtent | change rule (dVV < -3 and dVH < -3 dB, 23 Oct scene vs pre-event
  composite) | our 23 Oct scene classification (open-water thresholds)

Output: outputs/tables/unosat_exposure.csv
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
from src.validation.compare_unosat import UN, rd
from src.validation.threshold_exposure_direct import ROAD_SAMPLE_SPACING_M, interpolate_road_points, reproject_population_conserving

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
POP_SCALE = 366077 / 426499


def main() -> None:
    root = get_project_root()
    sat, fl = root / "data/processed/satellite", root / "data/processed/flood_extent"
    ref = sat / "s1_pre_event_vv.tif"
    with rasterio.open(ref) as s:
        tr, crs, shape = s.transform, s.crs, s.shape
    def ras(name):
        g = gpd.read_file(root / UN.format(name)).to_crs(crs)
        return rasterize(((geom, 1) for geom in g.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    dom = ras("AnalysisExtent"); uf = ras("FloodExtent") & dom
    fused = rd(fl / "fused_flood_confidence.tif")
    static = static_water(ref, sat, fl)["union"]
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), str(ref), Resampling.bilinear)
    ok = ~static & ~(np.isfinite(slope) & (slope > 5))
    dvv = rd(sat / "s1_scene_20231023_vv.tif") - rd(sat / "s1_pre_event_vv.tif")
    dvh = rd(sat / "s1_scene_20231023_vh.tif") - rd(sat / "s1_pre_event_vh.tif")
    chg = remove_small_patches(((dvv < -3) & (dvh < -3) & ok & dom).astype(np.uint8)) == 1
    defs = {"our primary fused extent": np.isin(fused, [1, 2, 3]) & dom, "UNOSAT FloodExtent (23 Oct 2023)": uf,
            "change rule: dVV and dVH < -3 dB (23 Oct)": chg,
            "our open-water rule on the 23 Oct scene": rd(fl / "sar_flood_scene_20231023.tif") == 1}
    defs["our open-water rule on the 23 Oct scene"] &= dom
    bld = gpd.read_file(root / "data/raw/buildings/osm_buildings.gpkg").to_crs(crs); c = bld.geometry.centroid
    br, bc = rasterio.transform.rowcol(tr, c.x.values, c.y.values); br, bc = np.array(br), np.array(bc)
    okb = (br >= 0) & (br < shape[0]) & (bc >= 0) & (bc < shape[1])
    roads = gpd.read_file(root / "data/raw/roads/osm_roads.gpkg").to_crs(crs)
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].explode(index_parts=False).reset_index(drop=True)
    rx, ry, _, rlen = interpolate_road_points(roads, ROAD_SAMPLE_SPACING_M)
    rr, rc = rasterio.transform.rowcol(tr, rx, ry); rr, rc = np.array(rr), np.array(rc)
    okr = (rr >= 0) & (rr < shape[0]) & (rc >= 0) & (rc < shape[1])
    pop = reproject_population_conserving(root / "data/processed/population/population_grid_clipped.tif", ref) * POP_SCALE
    rows = []
    for k, m in defs.items():
        rows.append({"extent": k, "area_km2": m.sum() / 1e4, "buildings": int(m[br[okb], bc[okb]].sum()),
                     "roads_km": float(rlen[okr][m[rr[okr], rc[okr]]].sum() / 1000), "people": float(pop[m].sum())})
    df = pd.DataFrame(rows); df.to_csv(get_output_path("tables/unosat_exposure.csv"), index=False)
    log.info("\n%s", df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
