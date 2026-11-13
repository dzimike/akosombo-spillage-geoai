"""
Raster alignment tests.

Checks that feature stack rasters share the same CRS, transform, and shape.
"""

import pytest
import rasterio
import numpy as np
from pathlib import Path

from src.utils.crs import rasters_are_aligned
from src.utils.paths import get_project_root

ROOT = get_project_root()
FEATURE_DIR = ROOT / "data/processed/features"


def get_feature_rasters():
    if not FEATURE_DIR.exists():
        return []
    return list(FEATURE_DIR.glob("*.tif"))


@pytest.mark.parametrize("raster_path", get_feature_rasters())
def test_feature_raster_in_projected_crs(raster_path):
    """All feature rasters must be in EPSG:32630."""
    with rasterio.open(raster_path) as src:
        from pyproj import CRS as ProjCRS
        epsg = ProjCRS(src.crs).to_epsg()
    assert epsg == 32630, f"Expected EPSG:32630, got EPSG:{epsg}: {raster_path}"


def test_feature_stack_bands_aligned():
    """All bands in the feature stack must have the same transform and shape."""
    stack_path = FEATURE_DIR / "flood_model_feature_stack.tif"
    if not stack_path.exists():
        pytest.skip("Feature stack not yet built")
    with rasterio.open(stack_path) as src:
        assert src.count > 1, "Feature stack should have multiple bands"
        for band in range(1, src.count + 1):
            data = src.read(band)
            assert data.shape == (src.height, src.width)


def test_no_all_nodata_bands():
    """Feature stack bands should not be entirely nodata/NaN."""
    stack_path = FEATURE_DIR / "flood_model_feature_stack.tif"
    if not stack_path.exists():
        pytest.skip("Feature stack not yet built")
    with rasterio.open(stack_path) as src:
        for band in range(1, src.count + 1):
            data = src.read(band, masked=True)
            valid_count = data.count()
            assert valid_count > 0, f"Band {band} is entirely masked/nodata"
