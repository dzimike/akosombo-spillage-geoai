from __future__ import annotations
"""
Phase 4 – Task 4.1: SAR-Based Flood Detection (change-based definition)

Flood = backscatter DROP relative to the pre-event state, on flat terrain, outside pre-existing water.

1. Change evidence (preprocessing/s1_change_by_date.py -> satellite/s1_change_by_date.tif): every acquisition is
   compared with the median of the pre-event acquisitions of the SAME relative orbit (7x7 focal mean, dB). A
   pixel is 'changed' on a date if dVV < -K and dVH < -K.
2. SAR flood = changed on at least PERSISTENCE acquisition dates inside the release period (15 Sep - 30 Oct 2023;
   7 dates of the two orbits) (default K = 3 dB, 2 dates, chosen a priori; the K x persistence grid is scored in
   validation/evaluate_change_definitions.py). The release period rather than the 15 Sep - 15 Nov event window is
   used because seasonal dry-season change in November can satisfy the persistence criterion on its own (the
   per-date series, validation/s1_change_timeseries.py, shows 0.8 of the western scatter patches flagged
   from November through January and 0.06-0.15 already in July-August).
3. Pre-existing water (never flood): JRC occurrence >= 50 %, open-sea mask, pre-event optical water, and
   SAR open water in the pre-event window (VH < T_VH and VV < T_VV; Youden J vs JRC water / land).
   The same open-water rule applied to the event window is retained as a sensitivity extent
   (sar_openwater_flood_mask.tif = event open water outside static water, the pre-change definition).
4. Slope <= 5 deg, minimum mapping unit 10 px.

Why not open water in the event window: the old rule found 72 km2 of which 58 % lay in the Songor-Ada
wetland that was already SAR-dark before the release, and it agreed with the analyst-reviewed UNOSAT extent
at F1 0.13 (change rule: 0.41-0.46). It also cannot see flooding below a roof line or among vegetation, where
the backscatter does not fall below a water threshold (Mepe, Charter 844: AUC 0.48-0.62 for VV/VH).

Outputs
-------
data/processed/flood_extent/sar_flood_mask_binary.tif      change-based flood (255 = no change evidence)
data/processed/flood_extent/sar_openwater_flood_mask.tif   event open water outside static water (sensitivity)
data/processed/flood_extent/sar_event_water_mask.tif       water in event window
data/processed/flood_extent/sar_pre_water_mask.tif         water in pre-event window
data/processed/flood_extent/sar_new_water_mask.tif         event open water that was not pre-event water
data/processed/flood_extent/sar_flood_extent.gpkg
data/processed/flood_extent/sar_threshold_report.csv
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from src.utils.paths import load_config, get_processed_path, ensure_dir
from src.utils.logging_utils import Timer, log_run
from src.utils.raster import read_aligned
from src.flood_mapping.water_masks import static_water
from src.flood_mapping.change_counts import count_flags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIN_MAPPING_UNIT_PIXELS = 10
SLOPE_THRESHOLD_DEGREES = 5.0
MNDWI_WATER_THRESHOLD = 0.0
K_DB = 3                  # dB drop in both VV and VH relative to the orbit-matched pre-event median
PERSISTENCE = 2           # minimum number of event-window dates on which the drop is observed
CHANGE_BAND = {2: 1, 3: 2, 4: 3}   # K -> 1-based band in s1_change_stack.tif (cnt2, cnt3, cnt4)


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def compute_change_images(
    vv_pre: np.ndarray, vv_event: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return dB difference and linear ratio change images (NaN-safe)."""
    diff = vv_event - vv_pre
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isfinite(vv_pre) & (vv_pre != 0),
                         vv_event / vv_pre, np.nan)
    return diff, ratio


def otsu_threshold(array: np.ndarray) -> float:
    """Otsu threshold on finite values of array."""
    from skimage.filters import threshold_otsu
    flat = array[np.isfinite(array)].ravel()
    return float(threshold_otsu(flat))


def youden_threshold(water_vals: np.ndarray, land_vals: np.ndarray) -> tuple[float, float]:
    """Backscatter threshold (dB) maximising TPR - FPR for 'value < threshold = water'."""
    ts = np.linspace(-40.0, -5.0, 351)
    w, l = np.sort(water_vals), np.sort(land_vals)
    j = np.searchsorted(w, ts) / len(w) - np.searchsorted(l, ts) / len(l)
    i = int(np.argmax(j))
    return float(ts[i]), float(j[i])


def apply_permanent_water_mask(
    flood_mask: np.ndarray, permanent_water: np.ndarray
) -> np.ndarray:
    """Zero-out pixels classified as permanent water."""
    return np.where(permanent_water.astype(bool), 0, flood_mask).astype(np.uint8)


def apply_slope_mask(
    flood_mask: np.ndarray, slope: np.ndarray,
    threshold: float = SLOPE_THRESHOLD_DEGREES,
) -> np.ndarray:
    """Zero-out steep-slope false positives."""
    steep = np.isfinite(slope) & (slope > threshold)
    return np.where(steep, 0, flood_mask).astype(np.uint8)


def remove_small_patches(
    flood_mask: np.ndarray, min_pixels: int = MIN_MAPPING_UNIT_PIXELS
) -> np.ndarray:
    """Remove connected components smaller than the minimum mapping unit."""
    from scipy import ndimage
    labeled, _ = ndimage.label(flood_mask)
    component_sizes = ndimage.sum(flood_mask, labeled,
                                  range(1, labeled.max() + 1))
    small_ids = [i + 1 for i, s in enumerate(component_sizes)
                 if s < min_pixels]
    mask_small = np.isin(labeled, small_ids)
    return np.where(mask_small, 0, flood_mask).astype(np.uint8)


def raster_to_polygon(
    flood_mask: np.ndarray, transform, crs
):
    """Convert binary raster mask to GeoDataFrame of polygons."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    polys = []
    for geom, val in shapes(flood_mask, transform=transform):
        if val == 1:
            polys.append(shape(geom))
    gdf = gpd.GeoDataFrame({"geometry": polys}, crs=crs)
    if not gdf.empty:
        gdf["area_ha"] = (gdf.geometry.area / 1e4).round(4)
    return gdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write(path, arr, profile, nodata=255):
    prof = profile.copy()
    prof.update(dtype="uint8", count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.uint8), 1)


def main(config_path: str | None = None, k_db: int = K_DB, persistence: int = PERSISTENCE) -> None:
    config    = load_config(config_path)
    sat_dir   = get_processed_path("satellite", config)
    dem_dir   = get_processed_path("dem", config)
    flood_dir = get_processed_path("flood_extent", config)
    ensure_dir(flood_dir)

    paths = {k: sat_dir / f"s1_{k}.tif" for k in
             ("pre_event_vv", "pre_event_vh", "event_vv", "event_vh")}
    slope_path = dem_dir / "slope.tif"
    stack_path = sat_dir / "s1_change_by_date.tif"
    for p in [*paths.values(), slope_path, stack_path]:
        if not p.exists():
            log.error("Required file missing: %s", p)
            return
    ref = paths["pre_event_vv"]

    with Timer() as t:
        with rasterio.open(ref) as src:
            profile, transform, crs = src.profile.copy(), src.transform, src.crs
        arr = {}
        for k, p in paths.items():
            with rasterio.open(p) as src:
                arr[k] = src.read(1).astype(np.float32)
        valid = np.logical_and.reduce([np.isfinite(a) for a in arr.values()])
        log.info("Valid S1 pixels (all four composites): %.1f%%", 100 * valid.mean())

        # ---- Static water + slope -------------------------------------------
        comps = static_water(ref, sat_dir, flood_dir)
        static = comps["union"]
        for k, v in comps.items():
            log.info("  static water %-12s %.1f km2", k, v.sum() / 1e4)
        slope = read_aligned(str(slope_path), str(ref), Resampling.bilinear)
        flat = ~(np.isfinite(slope) & (slope > SLOPE_THRESHOLD_DEGREES))

        # ---- Backscatter water thresholds (Youden J: JRC water vs land) -------
        land = valid & ~static & flat
        jrc = comps["jrc"] & valid
        t_vh, j_vh = youden_threshold(arr["event_vh"][jrc][::7], arr["event_vh"][land][::7])
        t_vv, j_vv = youden_threshold(arr["event_vv"][jrc][::7], arr["event_vv"][land][::7])
        log.info("Youden thresholds on event backscatter: VH %.2f dB (J=%.3f), VV %.2f dB (J=%.3f)",
                 t_vh, j_vh, t_vv, j_vv)

        def water(prefix):
            return valid & (arr[f"{prefix}_vh"] < t_vh) & (arr[f"{prefix}_vv"] < t_vv)
        event_w, pre_w = water("event"), water("pre_event")
        log.info("  event water %.1f km2 | pre-event water %.1f km2 (excl. static water)",
                 (event_w & ~static).sum() / 1e4, (pre_w & ~static).sum() / 1e4)

        # ---- Change evidence (orbit-matched pre-event median, K dB, persistence inside the release period) -----
        cnt, n_obs, dates = count_flags(sat_dir, k_db)
        changed = (cnt >= persistence) & valid
        log.info("  change evidence K=%d dB, >=%d of %d release-period dates %s: %.1f km2 (before water/slope masks)",
                 k_db, persistence, len(dates), dates, changed.sum() / 1e4)

        # ---- Flood = change outside pre-existing water; masks; MMU --------------
        preexisting = static | pre_w
        flood = changed & ~preexisting & flat
        flood_mask = remove_small_patches(flood.astype(np.uint8))
        flood_mask = np.where(valid, flood_mask, 255).astype(np.uint8)
        n_flood = int((flood_mask == 1).sum())
        area_km2 = n_flood * 100 / 1e6
        log.info("  flood (change, outside pre-existing water): %d px = %.2f km2", n_flood, area_km2)
        ow = remove_small_patches((event_w & ~static & flat).astype(np.uint8))
        ow_km2 = float(ow.sum()) * 100 / 1e6
        new_mask = remove_small_patches((event_w & ~static & flat & ~pre_w).astype(np.uint8))
        new_km2 = float(new_mask.sum()) * 100 / 1e6
        both = int(((flood_mask == 1) & (ow == 1)).sum()) * 100 / 1e6
        log.info("  event open-water extent (old definition): %.2f km2; new-water-only %.2f km2; overlap with change flood %.2f km2",
                 ow_km2, new_km2, both)

        _write(flood_dir / "sar_flood_mask_binary.tif", flood_mask, profile)
        _write(flood_dir / "sar_openwater_flood_mask.tif", ow, profile)
        _write(flood_dir / "sar_event_water_mask.tif", event_w, profile)
        _write(flood_dir / "sar_pre_water_mask.tif", pre_w, profile)
        _write(flood_dir / "sar_new_water_mask.tif", new_mask, profile)

        flood_poly = raster_to_polygon((flood_mask == 1).astype(np.uint8), transform, crs)
        n_polys = len(flood_poly)
        out_gpkg = flood_dir / "sar_flood_extent.gpkg"
        if not flood_poly.empty:
            flood_poly.to_file(str(out_gpkg), driver="GPKG")

        with open(flood_dir / "sar_threshold_report.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["threshold_vh_dB_youden", round(float(t_vh), 4)])
            w.writerow(["youden_J_vh", round(float(j_vh), 4)])
            w.writerow(["threshold_vv_dB_youden", round(float(t_vv), 4)])
            w.writerow(["youden_J_vv", round(float(j_vv), 4)])
            w.writerow(["event_water_km2_excl_static", round((event_w & ~static).sum() / 1e4, 2)])
            w.writerow(["pre_event_water_km2_excl_static", round((pre_w & ~static).sum() / 1e4, 2)])
            w.writerow(["change_k_dB", k_db])
            w.writerow(["change_persistence_dates", persistence])
            w.writerow(["openwater_event_extent_km2", round(ow_km2, 4)])
            w.writerow(["flood_pixels_final", n_flood])
            w.writerow(["flood_area_km2", round(area_km2, 4)])
            w.writerow(["new_water_only_km2", round(new_km2, 4)])
            w.writerow(["flood_polygons", n_polys])
            w.writerow(["pixel_size_m", 10])
            w.writerow(["slope_mask_threshold_deg", SLOPE_THRESHOLD_DEGREES])
            w.writerow(["mmu_pixels", MIN_MAPPING_UNIT_PIXELS])
            w.writerow(["permanent_water_source", "JRC>=50 | GLO-30 sea mask | pre-event NDWI+MNDWI | pre-event SAR open water"])
    log.info("SAR flood detection complete in %.1f s", t.elapsed)

    log_run(script=__file__, input_paths=[str(p) for p in paths.values()],
            output_paths=[str(flood_dir / "sar_flood_mask_binary.tif")],
            parameters={"k_dB": k_db, "persistence": persistence, "vh_dB": round(float(t_vh), 3), "vv_dB": round(float(t_vv), 3),
                        "area_km2": round(area_km2, 2), "n_polygons": n_polys},
            status="success", runtime_seconds=t.elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR-based flood detection")
    parser.add_argument("--config", default=None)
    parser.add_argument("--k", type=int, default=K_DB, choices=sorted(CHANGE_BAND))
    parser.add_argument("--persistence", type=int, default=PERSISTENCE)
    args = parser.parse_args()
    main(args.config, args.k, args.persistence)
