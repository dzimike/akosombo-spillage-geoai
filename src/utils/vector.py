from __future__ import annotations
"""
Vector geometry validation and utility helpers.
"""

import geopandas as gpd
from shapely.validation import make_valid


def fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attempt to repair invalid geometries in-place and return the GeoDataFrame."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g
    )
    return gdf


def drop_empty_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop rows with null or empty geometries."""
    return gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()


def validate_vector(gdf: gpd.GeoDataFrame) -> list[str]:
    """Return a list of validation warnings for a GeoDataFrame."""
    warnings = []
    invalid = (~gdf.geometry.is_valid).sum()
    empty = gdf.geometry.is_empty.sum()
    null = gdf.geometry.isna().sum()
    if invalid:
        warnings.append(f"{invalid} invalid geometries")
    if empty:
        warnings.append(f"{empty} empty geometries")
    if null:
        warnings.append(f"{null} null geometries")
    return warnings
