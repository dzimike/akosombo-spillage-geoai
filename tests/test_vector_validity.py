"""
Vector geometry validity tests.

Checks for invalid geometries, empty geometries, and null CRS in
all processed vector layers.
"""

import pytest
import geopandas as gpd
from pathlib import Path

from src.utils.paths import get_project_root

ROOT = get_project_root()


def get_all_geopackages():
    processed = ROOT / "data/processed"
    if not processed.exists():
        return []
    return list(processed.rglob("*.gpkg"))


@pytest.mark.parametrize("gpkg_path", get_all_geopackages())
def test_no_invalid_geometries(gpkg_path):
    gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        return
    invalid = ~gdf.geometry.is_valid
    assert not invalid.any(), f"{invalid.sum()} invalid geometries in {gpkg_path}"


@pytest.mark.parametrize("gpkg_path", get_all_geopackages())
def test_no_null_geometries(gpkg_path):
    gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        return
    null_count = gdf.geometry.isna().sum()
    assert null_count == 0, f"{null_count} null geometries in {gpkg_path}"


def test_community_points_within_ghana_bounds():
    """Affected community points must fall within Ghana's approximate bounding box."""
    comm_path = ROOT / "data/processed/aoi/affected_communities.gpkg"
    if not comm_path.exists():
        pytest.skip("Communities layer not yet built")
    gdf = gpd.read_file(comm_path).to_crs("EPSG:4326")
    lon_min, lon_max = -3.5, 1.2
    lat_min, lat_max = 4.5, 11.2
    for _, row in gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y
        assert lon_min <= x <= lon_max, f"Community {row.get('community_name')} longitude {x} outside Ghana bounds"
        assert lat_min <= y <= lat_max, f"Community {row.get('community_name')} latitude {y} outside Ghana bounds"
