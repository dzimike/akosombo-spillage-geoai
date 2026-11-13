from __future__ import annotations
"""
Counts of acquisition dates with a SAR backscatter drop, from the per-date change flags
(satellite/s1_change_by_date.tif; bit 1 = dVV & dVH < -2 dB, bit 2 = -3 dB, bit 3 = -4 dB).

The release period is 15 September - 30 October 2023 (Volta River Authority). Counting inside it, not the full
15 Sep - 15 Nov event window, keeps dry-season onset in November from satisfying the persistence criterion (see
validation/s1_change_timeseries.py and validation/release_window_variant.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

RELEASE_WINDOW = ("2023-09-15", "2023-10-30")
BIT = {2: 2, 3: 4, 4: 8}


def count_flags(sat_dir: Path, k_db: int = 3, window: tuple[str, str] = RELEASE_WINDOW) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (flagged-date count, observed-date count, dates) inside the window."""
    bands = pd.read_csv(sat_dir / "s1_change_by_date_bands.csv")
    sel = [i + 1 for i, d in enumerate(bands.date) if window[0] <= d <= window[1]]
    with rasterio.open(sat_dir / "s1_change_by_date.tif") as src:
        cnt = np.zeros(src.shape, np.uint8)
        nobs = np.zeros(src.shape, np.uint8)
        for i in sel:
            code = src.read(i)
            cnt += ((code & BIT[k_db]) > 0).astype(np.uint8)
            nobs += ((code & 1) > 0).astype(np.uint8)
    return cnt, nobs, [str(bands.date[i - 1]) for i in sel]
