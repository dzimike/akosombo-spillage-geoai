"""
Community buffer and assignment utilities.

Assigns exposure results to the nearest affected community and builds
community-level buffer polygons for spatial summaries.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.utils.crs import PROJ_CRS, to_projected


def assign_to_nearest_community(
    gdf: gpd.GeoDataFrame,
    communities: gpd.GeoDataFrame,
    id_col: str = "community_name",
) -> gpd.GeoDataFrame:
    """
    Spatially join each feature in gdf to the nearest community centroid.

    Both inputs must be in the same projected CRS.
    """
    # TODO: implement using sjoin_nearest once geopandas>=0.10
    result = gpd.sjoin_nearest(gdf, communities[[id_col, "geometry"]], how="left")
    return result


def build_community_buffers(
    communities: gpd.GeoDataFrame, buffer_m: float = 5000
) -> gpd.GeoDataFrame:
    """
    Create circular buffer polygons around community centroids.

    Parameters
    ----------
    communities : GeoDataFrame in projected CRS
    buffer_m : buffer radius in metres
    """
    proj = to_projected(communities) if communities.crs.to_epsg() != 32630 else communities
    result = proj.copy()
    result["geometry"] = proj.geometry.buffer(buffer_m)
    result["buffer_m"] = buffer_m
    return result
