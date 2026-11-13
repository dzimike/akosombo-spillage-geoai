from __future__ import annotations
"""
Exposed area, buildings, people and road length by detection class of the fused extent
(1 = SAR change only, 2 = optical only, 3 = both), and the area overlap between the open-water definition and the
change-based extent. Same sampling as the other exposure scripts (building centroids, road points every 20 m,
total-preserving population rescaled to the 366,077 total).

Outputs: outputs/tables/exposure_by_detection_class.csv, outputs/tables/definition_area_overlap.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.flood_mapping.water_masks import static_water
from src.utils.paths import get_project_root, get_output_path
from src.validation.threshold_exposure_direct import (ROAD_SAMPLE_SPACING_M, interpolate_road_points,
                                                      reproject_population_conserving)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
POP_SCALE = 366077 / 426499


def main() -> None:
    root = get_project_root()
    fl, sat = root / "data/processed/flood_extent", root / "data/processed/satellite"
    rd = lambda p: rasterio.open(p).read(1)
    ref = sat / "s1_pre_event_vv.tif"
    fu = rd(fl / "fused_flood_confidence.tif")
    with rasterio.open(ref) as s:
        tr, crs = s.transform, s.crs
    H, W = fu.shape
    bl = gpd.read_file(root / "data/raw/buildings/osm_buildings.gpkg").to_crs(crs)
    c = bl.geometry.centroid
    br, bc = rasterio.transform.rowcol(tr, c.x.values, c.y.values); br, bc = np.array(br), np.array(bc)
    okb = (br >= 0) & (br < H) & (bc >= 0) & (bc < W)
    roads = gpd.read_file(root / "data/raw/roads/osm_roads.gpkg").to_crs(crs)
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].explode(index_parts=False).reset_index(drop=True)
    rx, ry, _, rlen = interpolate_road_points(roads, ROAD_SAMPLE_SPACING_M)
    rr, rc = rasterio.transform.rowcol(tr, rx, ry); rr, rc = np.array(rr), np.array(rc)
    okr = (rr >= 0) & (rr < H) & (rc >= 0) & (rc < W)
    pop = reproject_population_conserving(root / "data/processed/population/population_grid_clipped.tif", ref)
    rows = []
    for code, name in ((1, "SAR change only"), (2, "Optical water only"), (3, "SAR and optical")):
        m = fu == code
        rows.append({"class": name, "area_km2": m.sum() / 1e4, "buildings": int(m[br[okb], bc[okb]].sum()),
                     "people": float(pop[m].sum() * POP_SCALE), "roads_km": float(rlen[okr][m[rr[okr], rc[okr]]].sum() / 1000)})
    df = pd.DataFrame(rows)
    df.to_csv(get_output_path("tables/exposure_by_detection_class.csv"), index=False)
    log.info("\n%s\n%s", df.round(2).to_string(index=False), df.drop(columns="class").sum().round(2).to_dict())
    # definition overlap
    pre = fu == 4
    static = static_water(ref, sat, fl)["union"]
    opt = rd(fl / "optical_water_mask.tif") == 1
    ow = rd(fl / "sar_openwater_flood_mask.tif") == 1
    old = ow | (opt & ~static)
    new = np.isin(fu, [1, 2, 3])
    ov = pd.DataFrame([{"open_water_total_km2": old.sum() / 1e4, "change_total_km2": new.sum() / 1e4,
                        "both_km2": (old & new).sum() / 1e4, "open_water_only_km2": (old & ~new).sum() / 1e4,
                        "change_only_km2": (new & ~old).sum() / 1e4,
                        "open_water_only_in_preexisting_km2": (old & ~new & pre).sum() / 1e4}])
    ov.to_csv(get_output_path("tables/definition_area_overlap.csv"), index=False)
    log.info("\n%s", ov.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
