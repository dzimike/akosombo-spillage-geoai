from __future__ import annotations
"""
Flooded buildings in Mepe, from International Charter activation 844 (NASRDA map, 21 Oct 2023).

The map (data/raw/validation/charter844/img_28582238.jpg) shows OSM building footprints coloured orange as
'affected buildings' on a WorldView-2 image of 18 Oct 2023 (analyst interpretation, not field-validated).
The map is in plain lon/lat (EPSG:4326); the graticule ticks give an exact linear georeference:
   0 deg 25' E at x = 581.5 px, 0 deg 26' E at x = 3160.5 px;  6 deg 5' N at y = 771.5 px, 6 deg 4' N at y = 3347.5 px.
Orange polygons inside the map frame are labelled into connected components, georeferenced, and matched to the
OSM building footprints (orange component centroid inside a footprint, or footprint centroid inside the component).

Outputs: data/processed/validation/charter_mepe_flooded_buildings.gpkg (matched OSM footprints, flagged),
         outputs/tables/charter_mepe_buildings.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from shapely.geometry import Point

from src.utils.paths import get_project_root, get_output_path

Image.MAX_IMAGE_PIXELS = None
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
FRAME = (245, 4863, 331, 3725)   # x0, x1, y0, y1 of the map frame in pixels


def px_to_lonlat(x, y):
    return 25 / 60 + (x - 581.5) / (2579.0 * 60), 6 + 5 / 60 - (y - 771.5) / (2576.0 * 60)


def main() -> None:
    root = get_project_root()
    im = np.array(Image.open(root / "data/raw/validation/charter844/img_28582238.jpg").convert("RGB")).astype(int)
    x0, x1, y0, y1 = FRAME
    sub = im[y0 + 6:y1 - 5, x0 + 6:x1 - 5]
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    orange = (r > 215) & (g > 60) & (g < 135) & (b < 70)
    log.info("orange pixels %d (%.2f%% of frame)", orange.sum(), 100 * orange.mean())
    lab, n = ndimage.label(ndimage.binary_closing(orange, iterations=2))
    sizes = ndimage.sum(orange, lab, range(1, n + 1))
    keep = [i + 1 for i, s_ in enumerate(sizes) if s_ >= 60]           # drop specks (< ~30 m2)
    log.info("components %d, kept (>=60 px) %d; area px median %.0f", n, len(keep), np.median([sizes[i - 1] for i in keep]))
    cent = ndimage.center_of_mass(orange, lab, keep)
    rows = []
    for (cy, cx), k in zip(cent, keep):
        lon, lat = px_to_lonlat(cx + x0 + 6, cy + y0 + 6)
        rows.append({"lon": lon, "lat": lat, "area_px": float(sizes[k - 1])})
    comp = gpd.GeoDataFrame(rows, geometry=[Point(r_["lon"], r_["lat"]) for r_ in rows], crs=4326).to_crs(32630)
    # all OSM buildings inside the frame
    lo0, la0 = px_to_lonlat(x0, y0); lo1, la1 = px_to_lonlat(x1, y1)
    bld = gpd.read_file(root / "data/raw/buildings/osm_buildings.gpkg")
    fr = gpd.GeoSeries([__import__("shapely.geometry", fromlist=["box"]).box(lo0, la1, lo1, la0)], crs=4326).to_crs(bld.crs).iloc[0]
    bld = bld[bld.intersects(fr)].reset_index(drop=True).to_crs(32630)
    log.info("OSM buildings in the map frame: %d; orange components: %d", len(bld), len(comp))
    # flag buildings whose footprint contains / is within 3 m of an orange component centroid
    comp_buf = comp.copy(); comp_buf["geometry"] = comp.geometry.buffer(3)
    j = gpd.sjoin(bld, comp_buf[["geometry"]], how="left", predicate="intersects")
    bld["charter_affected"] = bld.index.isin(j.dropna(subset=["index_right"]).index)
    nflag = int(bld.charter_affected.sum())
    log.info("OSM buildings flagged affected by the Charter map: %d of %d", nflag, len(bld))
    out = root / "data/processed/validation"; out.mkdir(parents=True, exist_ok=True)
    bld[["geometry", "charter_affected"]].to_file(out / "charter_mepe_flooded_buildings.gpkg", driver="GPKG")
    pd.DataFrame([{"orange_components": len(comp), "osm_buildings_in_frame": len(bld), "osm_buildings_flagged": nflag}]).to_csv(
        get_output_path("tables/charter_mepe_buildings.csv"), index=False)


if __name__ == "__main__":
    main()
