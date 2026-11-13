"""
Permanent water mask from JRC Global Surface Water dataset.

Used to exclude permanent water bodies from flood detections.
"""

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

JRC_OCCURRENCE_THRESHOLD = 50  # percent occurrence; pixels above this are permanent water


def get_jrc_permanent_water_mask(ee, aoi_geom, threshold: int = JRC_OCCURRENCE_THRESHOLD):
    """
    Return a GEE binary permanent water mask from JRC Global Surface Water.

    Pixels with water occurrence >= threshold are classified as permanent.
    """
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    permanent = jrc.gte(threshold).rename("permanent_water")
    return permanent


def apply_permanent_water_mask_local(
    flood_mask: np.ndarray, jrc_occurrence: np.ndarray, threshold: int = JRC_OCCURRENCE_THRESHOLD
) -> np.ndarray:
    """Remove permanent water pixels from a local binary flood mask array."""
    permanent = jrc_occurrence >= threshold
    return np.where(permanent, 0, flood_mask).astype(np.uint8)
