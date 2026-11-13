from __future__ import annotations
"""
Phase 4 – Task 4.3: Flood Extent Fusion

Combines SAR and optical flood detections into a multi-source confidence map.

Classification scheme
---------------------
0 = not flooded
1 = SAR-only (medium–high confidence on flat terrain)
2 = optical-only (lower confidence — cloud shadows possible)
3 = SAR + optical agreement (high confidence)
4 = permanent water (JRC GSW >= 50 %, open sea, pre-event optical water)
9 = no data

Confidence classes (separate output band)
------------------------------------------
3 = high   (class 3, OR class 1 on slope ≤ 5°)
2 = medium (class 1 on slope 5–15°, or class 2 on slope ≤ 15°)
1 = low    (class 1 or 2 on steep terrain)
4 = permanent water
0 = not flooded

Usage
-----
    python src/flood_mapping/flood_extent_fusion.py --config config/config.yaml

Outputs
-------
data/processed/flood_extent/fused_flood_confidence.tif
data/processed/flood_extent/fused_flood_extent.gpkg
data/processed/flood_extent/fused_flood_stats.csv
outputs/maps/flood_extent_confidence_map.png
"""

import argparse
import csv
import logging

import numpy as np
import rasterio
from rasterio.enums import Resampling

from src.utils.paths import load_config, get_processed_path, get_output_path, ensure_dir
from src.utils.logging_utils import Timer, log_run
from src.utils.raster import read_aligned
from src.flood_mapping.water_masks import preexisting_water

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Class codes
CLASS_NO_DATA       = 9
CLASS_NOT_FLOODED   = 0
CLASS_SAR_ONLY      = 1
CLASS_OPTICAL_ONLY  = 2
CLASS_AGREEMENT     = 3
CLASS_PERM_WATER    = 4

# Slope thresholds for confidence assignment
SLOPE_HIGH_CONF   = 5.0   # degrees — flat floodplain
SLOPE_MEDIUM_CONF = 15.0  # degrees — gentle slope


# ---------------------------------------------------------------------------
# Core fusion functions
# ---------------------------------------------------------------------------

def fuse_flood_masks(
    sar_mask: np.ndarray,
    optical_mask: np.ndarray,
    permanent_water: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Produce a 6-class fusion raster (uint8)."""
    fused = np.full_like(sar_mask, CLASS_NOT_FLOODED, dtype=np.uint8)
    fused = np.where(sar_mask == 1,    CLASS_SAR_ONLY,     fused)
    fused = np.where(optical_mask == 1, CLASS_OPTICAL_ONLY, fused)
    fused = np.where((sar_mask == 1) & (optical_mask == 1), CLASS_AGREEMENT, fused)
    fused = np.where(permanent_water == 1, CLASS_PERM_WATER, fused)
    fused = np.where(~valid_mask, CLASS_NO_DATA, fused)
    return fused


def assign_confidence(
    fused: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    """Map class codes to 3-level confidence (0 not flooded, 1 low … 3 high)."""
    conf = np.zeros_like(fused, dtype=np.uint8)

    # SAR + optical agreement → high confidence
    conf = np.where(fused == CLASS_AGREEMENT, 3, conf)

    # SAR-only: slope-dependent confidence
    sar_only = fused == CLASS_SAR_ONLY
    conf = np.where(sar_only & (slope <= SLOPE_HIGH_CONF),   3, conf)
    conf = np.where(sar_only & (slope > SLOPE_HIGH_CONF) &
                               (slope <= SLOPE_MEDIUM_CONF), 2, conf)
    conf = np.where(sar_only & (slope > SLOPE_MEDIUM_CONF),  1, conf)

    # Optical-only: slope-dependent confidence
    opt_only = fused == CLASS_OPTICAL_ONLY
    conf = np.where(opt_only & (slope <= SLOPE_MEDIUM_CONF), 2, conf)
    conf = np.where(opt_only & (slope > SLOPE_MEDIUM_CONF),  1, conf)

    # Permanent water
    conf = np.where(fused == CLASS_PERM_WATER, CLASS_PERM_WATER, conf)

    return conf


def raster_to_polygon(fused: np.ndarray, transform, crs, flood_classes=(1, 2, 3)):
    """Vectorise flooded pixels (specified classes) to GeoDataFrame."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    flood_binary = np.isin(fused, list(flood_classes)).astype(np.uint8)
    rows = []
    for geom, val in shapes(flood_binary, transform=transform):
        if val == 1:
            rows.append(shape(geom))
    gdf = gpd.GeoDataFrame({"geometry": rows}, crs=crs)
    if not gdf.empty:
        gdf["area_ha"] = (gdf.geometry.area / 1e4).round(4)
    return gdf


# ---------------------------------------------------------------------------
# Quick overview map
# ---------------------------------------------------------------------------

def save_overview_map(fused: np.ndarray, transform, crs, out_path: str) -> None:
    """Save a simple colour-coded confidence map PNG using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        from rasterio.transform import array_bounds
        import pyproj

        bounds = array_bounds(fused.shape[0], fused.shape[1], transform)
        # Convert bounds to geographic for labelling
        transformer = pyproj.Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True)

        cmap = mcolors.ListedColormap([
            "#d9d9d9",  # 0 not flooded
            "#2196F3",  # 1 SAR-only
            "#4CAF50",  # 2 optical-only
            "#F44336",  # 3 agreement (high confidence)
            "#1565C0",  # 4 permanent water
        ])
        bounds_vals = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds_vals, cmap.N)

        display = np.where(fused == CLASS_NO_DATA, 0, fused).astype(np.float32)

        fig, ax = plt.subplots(figsize=(10, 12), dpi=150)
        im = ax.imshow(display, cmap=cmap, norm=norm,
                       extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
                       origin="upper")

        legend_patches = [
            mpatches.Patch(color="#2196F3", label="SAR only"),
            mpatches.Patch(color="#4CAF50", label="Optical only"),
            mpatches.Patch(color="#F44336", label="SAR + Optical (high confidence)"),
            mpatches.Patch(color="#1565C0", label="Permanent water"),
            mpatches.Patch(color="#d9d9d9", label="Not flooded"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
        ax.set_title("Akosombo Spillage 2023 — Fused Flood Extent\n"
                     "(SAR Sentinel-1 + Optical Sentinel-2)", fontsize=11)
        ax.set_xlabel("Easting (m, EPSG:32630)")
        ax.set_ylabel("Northing (m, EPSG:32630)")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        log.info("  Saved overview map: %s", out_path)
    except Exception as exc:
        log.warning("Could not save overview map: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    config    = load_config(config_path)
    flood_dir = get_processed_path("flood_extent", config)
    dem_dir   = get_processed_path("dem", config)
    sat_dir   = get_processed_path("satellite", config)
    maps_dir  = get_output_path("maps", config)
    ensure_dir(flood_dir)
    ensure_dir(maps_dir)

    sar_mask_path    = flood_dir / "sar_flood_mask_binary.tif"
    optical_mask_path = flood_dir / "optical_water_mask.tif"
    slope_path       = dem_dir / "slope.tif"
    jrc_path         = sat_dir / "jrc_permanent_water.tif"
    mndwi_pre_path   = sat_dir / "s2_mndwi_pre.tif"
    ndwi_pre_path    = sat_dir / "s2_ndwi_pre.tif"

    for p in [sar_mask_path, optical_mask_path, slope_path]:
        if not p.exists():
            log.error("Required file missing: %s", p)
            return

    with Timer() as t:
        # ---- Load SAR mask as reference grid ------------------------------
        log.info("Loading flood masks…")
        with rasterio.open(sar_mask_path) as src:
            sar_mask  = src.read(1).astype(np.uint8)
            profile   = src.profile.copy()
            transform = src.transform
            crs       = src.crs

        # Align optical mask to SAR grid (should already match, but ensure it)
        optical_mask = read_aligned(
            str(optical_mask_path), str(sar_mask_path), Resampling.nearest
        ).astype(np.uint8)
        optical_mask = np.where(np.isnan(optical_mask.astype(float)), 0,
                                optical_mask).astype(np.uint8)

        log.info("  SAR flood pixels:     %d", (sar_mask == 1).sum())
        log.info("  Optical water pixels: %d", (optical_mask == 1).sum())

        # ---- Permanent water -------------------------------------------------
        # JRC GSW >= 50 %, open sea (GLO-30), pre-event optical water and SAR open water in the
        # pre-event window: everything that was already water before the release (class 4).
        comps = preexisting_water(sar_mask_path, sat_dir, flood_dir)
        perm_water = comps["union"].astype(np.uint8)
        pw_source = "JRC | sea | pre-event optical | pre-event SAR open water"
        for k, v in comps.items():
            log.info("  permanent water %-12s %.1f km2", k, v.sum() / 1e4)

        # ---- Slope (aligned to SAR grid) ----------------------------------
        slope = read_aligned(str(slope_path), str(sar_mask_path), Resampling.bilinear)
        slope = np.where(~np.isfinite(slope), 0.0, slope).astype(np.float32)

        # ---- Valid data mask (has SAR data) --------------------------------
        with rasterio.open(sar_mask_path) as src:
            sar_raw = src.read(1)
        valid_mask = sar_raw != 255   # nodata = 255

        # ---- Fuse masks ---------------------------------------------------
        log.info("Fusing flood masks…")
        fused = fuse_flood_masks(sar_mask, optical_mask, perm_water, valid_mask)
        conf  = assign_confidence(fused, slope)

        # Class pixel counts
        class_labels = {
            CLASS_NOT_FLOODED:  "not_flooded",
            CLASS_SAR_ONLY:     "sar_only",
            CLASS_OPTICAL_ONLY: "optical_only",
            CLASS_AGREEMENT:    "sar_optical_agreement",
            CLASS_PERM_WATER:   "permanent_water",
            CLASS_NO_DATA:      "no_data",
        }
        for code, label in class_labels.items():
            n = int((fused == code).sum())
            km2 = n * 100 / 1e6
            log.info("  %-30s  %8d px  %.2f km²", label, n, km2)

        # ---- Write 2-band raster (fused + confidence) --------------------
        out_tif = flood_dir / "fused_flood_confidence.tif"
        out_profile = profile.copy()
        out_profile.update(dtype="uint8", count=2, nodata=255, compress="lzw")
        with rasterio.open(out_tif, "w", **out_profile) as dst:
            dst.write(fused, 1)
            dst.write(conf, 2)
            dst.update_tags(1, description="fused flood class (0-4, 9)")
            dst.update_tags(2, description="confidence level (0 no flood, 1-3 low/med/high, 4 perm water)")
        log.info("  Saved: %s", out_tif)

        # ---- Vectorise flooded classes ------------------------------------
        log.info("Vectorising fused flood extent…")
        flood_poly = raster_to_polygon(fused, transform, crs,
                                       flood_classes=(1, 2, 3))
        n_polys = len(flood_poly)
        total_ha = flood_poly["area_ha"].sum() if not flood_poly.empty else 0
        log.info("  Flood polygons: %d  total area: %.2f ha (%.2f km²)",
                 n_polys, total_ha, total_ha / 100)

        out_gpkg = flood_dir / "fused_flood_extent.gpkg"
        if not flood_poly.empty:
            flood_poly.to_file(str(out_gpkg), driver="GPKG")
            log.info("  Saved: %s", out_gpkg)

        # ---- Stats CSV ----------------------------------------------------
        stats_path = flood_dir / "fused_flood_stats.csv"
        total_flood_px = int(np.isin(fused, [1, 2, 3]).sum())
        total_flood_km2 = total_flood_px * 100 / 1e6
        with open(stats_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "pixels", "area_km2", "pct_of_aoi"])
            total_valid = int(valid_mask.sum())
            for code, label in class_labels.items():
                n   = int((fused == code).sum())
                km2 = round(n * 100 / 1e6, 4)
                pct = round(100 * n / total_valid, 3) if total_valid else 0
                w.writerow([label, n, km2, pct])
            w.writerow(["total_flooded_all_sources", total_flood_px,
                        round(total_flood_km2, 4),
                        round(100 * total_flood_px / total_valid, 3) if total_valid else 0])
        log.info("  Saved stats: %s", stats_path)

        # ---- Overview map -------------------------------------------------
        map_path = str(maps_dir / "flood_extent_confidence_map.png")
        save_overview_map(fused, transform, crs, map_path)

    log.info("Flood extent fusion complete in %.1f s", t.elapsed)

    log_run(
        script=__file__,
        input_paths=[str(sar_mask_path), str(optical_mask_path)],
        output_paths=[str(out_tif), str(out_gpkg)],
        parameters={"total_flood_km2": round(total_flood_km2, 2),
                    "n_polygons": n_polys,
                    "perm_water_source": pw_source},
        status="success",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse SAR and optical flood extents")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
