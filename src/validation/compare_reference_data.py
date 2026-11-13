"""
Compare SAR flood map against humanitarian assessment maps or
manually interpreted reference data.
"""

import logging
import geopandas as gpd
import pandas as pd

from src.utils.crs import to_projected, assert_crs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def compare_extents(
    sar_flood: gpd.GeoDataFrame,
    reference_flood: gpd.GeoDataFrame,
) -> dict[str, float]:
    """
    Compare two flood extent polygons and return area overlap statistics.

    Both inputs must be in the same projected CRS.
    """
    assert_crs(sar_flood)
    assert_crs(reference_flood)

    sar_union = sar_flood.geometry.unary_union
    ref_union = reference_flood.geometry.unary_union

    intersection = sar_union.intersection(ref_union).area / 1e6  # km2
    sar_area = sar_union.area / 1e6
    ref_area = ref_union.area / 1e6
    union_area = sar_union.union(ref_union).area / 1e6
    iou = intersection / union_area if union_area > 0 else 0

    return {
        "sar_area_km2": round(sar_area, 2),
        "reference_area_km2": round(ref_area, 2),
        "intersection_km2": round(intersection, 2),
        "iou": round(iou, 4),
        "percent_agreement": round(100 * intersection / ref_area, 1) if ref_area > 0 else 0,
    }
