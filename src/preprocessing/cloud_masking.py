"""
Cloud and shadow masking utilities for optical imagery.

Used by preprocess_sentinel2.py and any Landsat preprocessing.
"""

import numpy as np


def apply_scl_mask(image_array: np.ndarray, scl_array: np.ndarray) -> np.ndarray:
    """
    Mask cloud and shadow pixels using Sentinel-2 Scene Classification Layer.

    SCL classes masked: 3 (cloud shadow), 8 (medium cloud), 9 (high cloud), 10 (thin cirrus).
    Returns array with masked pixels set to NaN.
    """
    bad_pixels = np.isin(scl_array, [3, 8, 9, 10])
    masked = image_array.astype(float)
    masked[bad_pixels] = np.nan
    return masked


def apply_qa_pixel_mask_landsat(image_array: np.ndarray, qa_pixel: np.ndarray) -> np.ndarray:
    """
    Apply Landsat Collection 2 QA_PIXEL cloud/shadow mask.

    Bits 3 (cloud shadow) and 4 (cloud) are used.
    """
    cloud_shadow = (qa_pixel >> 3) & 1
    cloud = (qa_pixel >> 4) & 1
    bad = (cloud_shadow == 1) | (cloud == 1)
    masked = image_array.astype(float)
    masked[bad] = np.nan
    return masked


def estimate_cloud_fraction(scl_array: np.ndarray) -> float:
    """Return the fraction of cloud/shadow pixels in the SCL array."""
    total = scl_array.size
    cloudy = np.isin(scl_array, [3, 8, 9, 10]).sum()
    return float(cloudy / total) if total > 0 else 0.0
