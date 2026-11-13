from __future__ import annotations
"""
CRS validation and reprojection helpers.

All distance/area analysis must use EPSG:32630 (WGS 84 / UTM Zone 30N).
Geographic storage uses EPSG:4326.
"""

import geopandas as gpd
from pyproj import CRS

GEO_CRS = "EPSG:4326"
PROJ_CRS = "EPSG:32630"


def assert_crs(gdf: gpd.GeoDataFrame, expected: str = PROJ_CRS) -> None:
    """Raise ValueError if gdf is not in the expected CRS."""
    actual = gdf.crs.to_epsg() if gdf.crs else None
    expected_epsg = int(expected.split(":")[1])
    if actual != expected_epsg:
        raise ValueError(f"Expected CRS EPSG:{expected_epsg}, got EPSG:{actual}")


def to_projected(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject GeoDataFrame to the analysis CRS (EPSG:32630)."""
    return gdf.to_crs(PROJ_CRS)


def to_geographic(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject GeoDataFrame to the storage CRS (EPSG:4326)."""
    return gdf.to_crs(GEO_CRS)


def check_raster_crs(raster_path: str, expected_epsg: int = 32630) -> bool:
    """Return True if the raster has the expected CRS."""
    import rasterio
    with rasterio.open(raster_path) as src:
        actual = CRS(src.crs).to_epsg()
    return actual == expected_epsg


def rasters_are_aligned(paths: list[str]) -> bool:
    """Return True if all rasters share the same CRS, transform, and shape."""
    import rasterio
    metas = []
    for p in paths:
        with rasterio.open(p) as src:
            metas.append((src.crs, src.transform, src.width, src.height))
    return len(set(str(m) for m in metas)) == 1
