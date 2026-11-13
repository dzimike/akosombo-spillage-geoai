"""
Exposure overlay logic tests using synthetic geometries.

Verifies that spatial overlay correctly flags exposed features.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString

from src.exposure.building_exposure import compute_building_exposure
from src.exposure.road_exposure import compute_road_exposure
from src.utils.crs import PROJ_CRS


def make_flood_gdf(polygon):
    return gpd.GeoDataFrame(
        {"confidence_class": ["high"]},
        geometry=[polygon],
        crs=PROJ_CRS,
    )


def test_building_inside_flood_is_exposed():
    flood = make_flood_gdf(Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]))
    buildings = gpd.GeoDataFrame(
        {"building_id": [1, 2]},
        geometry=[Point(50, 50), Point(200, 200)],
        crs=PROJ_CRS,
    )
    result = compute_building_exposure(buildings, flood)
    assert result.loc[result.geometry == Point(50, 50), "flood_status"].values[0] == "exposed"
    assert result.loc[result.geometry == Point(200, 200), "flood_status"].values[0] == "not_exposed"


def test_road_inside_flood_has_flooded_length():
    flood = make_flood_gdf(Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]))
    road_inside = LineString([(10, 10), (90, 10)])
    road_outside = LineString([(200, 200), (300, 200)])
    roads = gpd.GeoDataFrame(
        {"highway": ["primary", "primary"]},
        geometry=[road_inside, road_outside],
        crs=PROJ_CRS,
    )
    result = compute_road_exposure(roads, flood)
    assert result.loc[0, "flooded_length_km"] > 0, "Road inside flood should have positive flooded length"
    assert result.loc[1, "flooded_length_km"] == 0.0, "Road outside flood should have zero flooded length"


def test_all_buildings_outside_flood_are_not_exposed():
    flood = make_flood_gdf(Polygon([(1000, 1000), (1100, 1000), (1100, 1100), (1000, 1100)]))
    buildings = gpd.GeoDataFrame(
        {"building_id": [1]},
        geometry=[Point(0, 0)],
        crs=PROJ_CRS,
    )
    result = compute_building_exposure(buildings, flood)
    assert all(result["flood_status"] == "not_exposed")
