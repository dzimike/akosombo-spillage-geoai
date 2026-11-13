"""
Otsu and adaptive thresholding utilities for SAR flood detection.
"""

import numpy as np


def otsu_threshold(array: np.ndarray) -> float:
    """Compute Otsu threshold on a finite-value array."""
    from skimage.filters import threshold_otsu
    return float(threshold_otsu(array[np.isfinite(array)]))


def adaptive_threshold(array: np.ndarray, window_size: int = 512, offset: float = 0.0) -> np.ndarray:
    """
    Compute an adaptive (local) threshold map.

    Each pixel's threshold is the local mean minus offset.
    Returns a boolean mask where True = likely water.
    """
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(array.astype(float), size=window_size)
    return array < (local_mean - offset)


def threshold_sensitivity_range(array: np.ndarray, n_steps: int = 10) -> list[dict]:
    """
    Return flood extent area estimates across a range of thresholds.

    Used for uncertainty/sensitivity analysis (Phase 9).
    """
    finite = array[np.isfinite(array)]
    low, high = np.percentile(finite, [5, 95])
    thresholds = np.linspace(low, high, n_steps)
    results = []
    for t in thresholds:
        count = int((array < t).sum())
        results.append({"threshold": float(t), "flooded_pixels": count})
    return results
