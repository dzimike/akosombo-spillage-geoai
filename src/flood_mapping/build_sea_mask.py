from __future__ import annotations
"""
Open-sea mask from the gap-free GLO-30 DEM.

Sea = pixels at or below SEA_LEVEL_M elevation that are connected (4-neighbour,
on the 10 m grid) to the southern boundary of the domain and form a component of
at least MIN_COMPONENT_PIXELS. GLO-30 is water-flattened to 0 m over the Gulf of
Guinea, so this follows the coastline without any optical or SAR input.

Why it is needed: the pre-event-optical permanent-water proxy missed about a
quarter of the sea surface (cloud-gapped pre-event composite), and the optical
event water test then labelled that open ocean as 'flood' (70 % of the original
fused flood area).

Output: data/processed/flood_extent/sea_mask.tif (uint8, 1 = sea)
"""
import logging

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage

from src.utils.paths import get_project_root
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
SEA_LEVEL_M = 0.5
MIN_COMPONENT_PIXELS = 1_000_000


def sea_mask(elev: np.ndarray) -> np.ndarray:
    lab, _ = ndimage.label(np.isfinite(elev) & (elev <= SEA_LEVEL_M))
    sizes = np.bincount(lab.ravel())
    south = np.unique(lab[-1, :])
    keep = [i for i in south if i > 0 and sizes[i] >= MIN_COMPONENT_PIXELS]
    return np.isin(lab, keep)


def main() -> None:
    root = get_project_root()
    ref = root / "data/processed/satellite/s1_pre_event_vv.tif"
    elev = read_aligned(str(root / "data/processed/dem/elevation.tif"), str(ref), Resampling.nearest)
    sea = sea_mask(elev)
    with rasterio.open(ref) as s:
        prof = s.profile.copy()
    prof.update(dtype="uint8", nodata=255, compress="lzw", count=1)
    out = root / "data/processed/flood_extent/sea_mask.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(sea.astype(np.uint8), 1)
    log.info("sea mask: %.1f km2 (%.1f%% of domain) -> %s", sea.sum() / 1e4, 100 * sea.mean(), out)


if __name__ == "__main__":
    main()
