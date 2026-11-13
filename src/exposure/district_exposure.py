from __future__ import annotations
"""
Flood exposure by administrative district (GADM level 2), non-overlapping.

The community analysis uses 10 km service buffers that overlap substantially, so the
same pixels, people, roads and wetland are counted for several communities. Districts
partition the land area of the analysis domain, so their totals add up. Each district is
clipped to the raster domain (the western tailrace margin and the sea are outside it).

For every district: land area in the domain, flood area (fused classes 1-3), OSM buildings
(total / centroid in flood), WorldPop population (total / in flood; rescaled to the 366,077
AOI total), and OSM road length (total / flooded, 20 m point sampling).

Output: outputs/tables/district_exposure.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from src.utils.paths import get_project_root, get_output_path
from src.validation.threshold_exposure_direct import (
    ROAD_SAMPLE_SPACING_M, interpolate_road_points, reproject_population_conserving,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
POP_SCALE = 366077 / 426499


def main() -> None:
    root = get_project_root()
    fused_p = root / "data/processed/flood_extent/fused_flood_confidence.tif"
    with rasterio.open(fused_p) as s:
        fused, tr, crs, shape = s.read(1), s.transform, s.crs, s.shape
    flood = np.isin(fused, [1, 2, 3])
    g = gpd.read_file(root / "data/raw/admin/gadm41_GHA.gpkg", layer="ADM_ADM_2").to_crs(crs)
    from shapely.geometry import box
    dom = box(tr.c, tr.f + shape[0] * tr.e, tr.c + shape[1] * tr.a, tr.f)
    g = g[g.intersects(dom)].reset_index(drop=True)
    g["gid"] = np.arange(1, len(g) + 1)
    ids = rasterize(((geom, int(i)) for geom, i in zip(g.geometry, g.gid)), out_shape=shape, transform=tr, fill=0, dtype="int32")
    land = (fused != 4) & (ids > 0)
    pop = reproject_population_conserving(root / "data/processed/population/population_grid_clipped.tif", fused_p) * POP_SCALE

    bld = gpd.read_file(root / "data/raw/buildings/osm_buildings.gpkg").to_crs(crs)
    c = bld.geometry.centroid
    br, bc = rasterio.transform.rowcol(tr, c.x.values, c.y.values); br, bc = np.array(br), np.array(bc)
    okb = (br >= 0) & (br < shape[0]) & (bc >= 0) & (bc < shape[1])
    b_id = np.zeros(len(bld), int); b_id[okb] = ids[br[okb], bc[okb]]
    b_fl = np.zeros(len(bld), bool); b_fl[okb] = flood[br[okb], bc[okb]]

    roads = gpd.read_file(root / "data/raw/roads/osm_roads.gpkg").to_crs(crs)
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].explode(index_parts=False).reset_index(drop=True)
    rx, ry, _, rlen = interpolate_road_points(roads, ROAD_SAMPLE_SPACING_M)
    rr, rc = rasterio.transform.rowcol(tr, rx, ry); rr, rc = np.array(rr), np.array(rc)
    okr = (rr >= 0) & (rr < shape[0]) & (rc >= 0) & (rc < shape[1])
    r_id = np.zeros(len(rx), int); r_id[okr] = ids[rr[okr], rc[okr]]
    r_fl = np.zeros(len(rx), bool); r_fl[okr] = flood[rr[okr], rc[okr]]

    rows = []
    for _, d in g.iterrows():
        m = ids == d.gid
        rows.append({"region": d.NAME_1, "district": d.NAME_2,
                     "land_area_km2": float((m & land).sum() / 1e4), "flood_km2": float((m & flood).sum() / 1e4),
                     "buildings_total": int((b_id == d.gid).sum()), "buildings_exposed": int(((b_id == d.gid) & b_fl).sum()),
                     "population_total": float(pop[m & (fused != 4)].sum()), "population_exposed": float(pop[m & flood].sum()),
                     "roads_total_km": float(rlen[r_id == d.gid].sum() / 1000), "roads_flooded_km": float(rlen[(r_id == d.gid) & r_fl].sum() / 1000)})
    df = pd.DataFrame(rows)
    df = df[(df.land_area_km2 > 1)].sort_values("flood_km2", ascending=False)
    df.to_csv(get_output_path("tables/district_exposure.csv"), index=False)
    log.info("\n%s", df.round(1).to_string(index=False))
    log.info("totals: flood %.1f km2 (in districts), buildings exposed %d, people %.0f, road %.1f km",
             df.flood_km2.sum(), df.buildings_exposed.sum(), df.population_exposed.sum(), df.roads_flooded_km.sum())


if __name__ == "__main__":
    main()
