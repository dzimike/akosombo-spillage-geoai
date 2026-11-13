from __future__ import annotations
"""
Conventional single-scene change mask: 23 Oct 2023 scene minus the pre-event composite, dVV < -3 dB and dVH < -3 dB,
outside static water, slope <= 5 deg, ten-pixel minimum mapping unit (not tuned to any reference).

Written to data/processed/flood_extent/sar_flood_scene_change_20231023.tif (the file sar_flood_scene_20231023.tif is the
OPEN-WATER rule applied to the same scene).
"""
import logging

import numpy as np
import rasterio
from rasterio.enums import Resampling

from src.flood_mapping.sar_change_detection import remove_small_patches
from src.flood_mapping.water_masks import static_water
from src.utils.paths import get_project_root
from src.utils.raster import read_aligned

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    root = get_project_root()
    sat, fl = root / "data/processed/satellite", root / "data/processed/flood_extent"
    rd = lambda p: rasterio.open(p).read(1)
    ref = sat / "s1_pre_event_vv.tif"
    dvv = rd(sat / "s1_scene_20231023_vv.tif") - rd(sat / "s1_pre_event_vv.tif")
    dvh = rd(sat / "s1_scene_20231023_vh.tif") - rd(sat / "s1_pre_event_vh.tif")
    static = static_water(ref, sat, fl)["union"]
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), str(ref), Resampling.bilinear)
    flat = ~(np.isfinite(slope) & (slope > 5))
    ok = ~static & flat & np.isfinite(dvv) & np.isfinite(dvh)
    m = remove_small_patches((((dvv < -3) & (dvh < -3)) & ok).astype(np.uint8))
    prof = rasterio.open(ref).profile.copy()
    prof.update(dtype="uint8", count=1, nodata=255, compress="lzw")
    with rasterio.open(fl / "sar_flood_scene_change_20231023.tif", "w", **prof) as d:
        d.write(m, 1)
    log.info("single-scene change mask: %.1f km2", m.sum() / 1e4)


if __name__ == "__main__":
    main()
