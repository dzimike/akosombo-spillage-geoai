"""
CRS validation tests.

Checks that all processed vector layers have the expected CRS.
"""

import pytest
import geopandas as gpd
from pathlib import Path
import os

from src.utils.crs import PROJ_CRS, GEO_CRS
from src.utils.paths import get_project_root

ROOT = get_project_root()


def get_processed_vectors():
    """Collect all .gpkg files under data/processed/."""
    p = ROOT / "data" / "processed"
    if not p.exists():
        return []
    return list(p.rglob("*.gpkg"))


@pytest.mark.parametrize("gpkg_path", get_processed_vectors())
def test_processed_vector_has_crs(gpkg_path):
    """All processed vectors must have a defined CRS."""
    gdf = gpd.read_file(gpkg_path)
    assert gdf.crs is not None, f"No CRS defined: {gpkg_path}"


def test_aoi_uses_geographic_crs():
    """AOI export should be stored in EPSG:4326."""
    aoi_path = ROOT / "data/processed/aoi/lower_volta_aoi.gpkg"
    if not aoi_path.exists():
        pytest.skip("AOI not yet built")
    gdf = gpd.read_file(aoi_path)
    assert gdf.crs.to_epsg() == 4326, f"Expected EPSG:4326, got EPSG:{gdf.crs.to_epsg()}"


def test_affected_communities_has_valid_geometries():
    """All community points must have valid, non-empty geometries."""
    comm_path = ROOT / "data/processed/aoi/affected_communities.gpkg"
    if not comm_path.exists():
        pytest.skip("Communities layer not yet built")
    gdf = gpd.read_file(comm_path)
    assert gdf.geometry.notna().all(), "Null geometries found in communities layer"
    assert gdf.geometry.is_valid.all(), "Invalid geometries found in communities layer"
    assert len(gdf) >= 5, "Expected at least 5 community points"
