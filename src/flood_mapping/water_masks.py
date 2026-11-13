from __future__ import annotations
"""
Shared definition of 'water that is not flood' for the flood-mapping stage.

static water  = JRC GSW occurrence >= 50 %          (permanent rivers, lagoons, sea)
              | open-sea mask from the GLO-30 DEM   (build_sea_mask.py)
              | pre-event optical water             (NDWI >= 0 and MNDWI >= 0 in the
                                                     pre-event composite, where valid)

preexisting_water() additionally includes SAR pre-event open water (sar_pre_water_mask.tif,
written by sar_change_detection.py); the fusion step labels that union class 4, so that a pixel
is only 'flood' if the event window shows it newly wet (SAR change or optical water) and it was
not water before the release.
"""
from pathlib import Path

import numpy as np
from rasterio.enums import Resampling

from src.utils.raster import read_aligned

JRC_THRESHOLD = 50


def static_water(ref_path: Path, sat_dir: Path, flood_dir: Path) -> dict[str, np.ndarray]:
    ref = str(ref_path)
    jrc_p, sea_p = sat_dir / "jrc_permanent_water.tif", flood_dir / "sea_mask.tif"
    for p in (jrc_p, sea_p):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing - run download_jrc_water.py / build_sea_mask.py first")
    jrc = read_aligned(str(jrc_p), ref, Resampling.nearest)
    comps = {
        "jrc": np.isfinite(jrc) & (jrc >= JRC_THRESHOLD),
        "sea": read_aligned(str(sea_p), ref, Resampling.nearest) == 1,
    }
    nd, mn = sat_dir / "s2_ndwi_pre.tif", sat_dir / "s2_mndwi_pre.tif"
    if nd.exists() and mn.exists():
        ndwi, mndwi = read_aligned(str(nd), ref), read_aligned(str(mn), ref)
        comps["optical_pre"] = np.isfinite(ndwi) & (ndwi >= 0) & np.isfinite(mndwi) & (mndwi >= 0)
    comps["union"] = np.logical_or.reduce([v for v in comps.values()])
    return comps


def preexisting_water(ref_path: Path, sat_dir: Path, flood_dir: Path) -> dict[str, np.ndarray]:
    """static_water components plus SAR pre-event open water; 'union' is the pre-existing-water mask (class 4)."""
    comps = static_water(ref_path, sat_dir, flood_dir)
    p = flood_dir / "sar_pre_water_mask.tif"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run sar_change_detection.py first")
    comps["sar_pre"] = read_aligned(str(p), str(ref_path), Resampling.nearest) == 1
    comps["union"] = np.logical_or.reduce([v for k, v in comps.items() if k != "union"])
    return comps
