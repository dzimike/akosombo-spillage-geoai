from __future__ import annotations
"""
Phase 4 – Task 4.4: Flood Depth Proxy

Produces a terrain-informed flood depth *proxy* within the detected flood extent.
This is NOT a calibrated flood depth map.  All outputs are labelled "proxy" and
must not be interpreted as true water depth without ground-truth gauge data.

Methodology
-----------
Within the fused flood extent (classes 1, 2, 3):
  * Relative elevation < low_thresh  → deep proxy   (class 3)
  * Relative elevation [low, high)   → moderate proxy (class 2)
  * Relative elevation ≥ high_thresh → shallow proxy (class 1)
Secondary refinement: high flow accumulation (percentile > 90) within moderate
  and shallow proxy areas is downgraded one class (more likely channelised flow).

Classes
-------
0 = outside flood extent
1 = shallow proxy
2 = moderate proxy
3 = deep proxy

Usage
-----
    python src/flood_mapping/flood_depth_proxy.py --config config/config.yaml

Outputs
-------
data/processed/flood_depth_proxy/flood_depth_proxy.tif
data/processed/flood_depth_proxy/flood_depth_proxy.gpkg
data/processed/flood_depth_proxy/flood_depth_proxy_stats.csv
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROXY_SHALLOW  = 1
PROXY_MODERATE = 2
PROXY_DEEP     = 3

# Relative elevation thresholds (metres above local minimum)
REL_ELEV_LOW  = 1.0   # below → deep
REL_ELEV_HIGH = 3.0   # above → shallow

FLOW_ACCUM_PERCENTILE = 90  # above → channel, refine proxy class down


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def classify_depth_proxy(
    relative_elevation: np.ndarray,
    flow_accum: np.ndarray,
    flood_mask: np.ndarray,
    low_thresh: float = REL_ELEV_LOW,
    high_thresh: float = REL_ELEV_HIGH,
) -> np.ndarray:
    """
    Classify flood depth proxy within the flood extent.

    IMPORTANT: Output is a terrain proxy only — do not equate with true depth.
    """
    proxy = np.zeros_like(flood_mask, dtype=np.uint8)
    within = flood_mask.astype(bool)

    low  = relative_elevation < low_thresh
    mid  = (relative_elevation >= low_thresh) & (relative_elevation < high_thresh)
    high = relative_elevation >= high_thresh

    proxy = np.where(within & low,  PROXY_DEEP,     proxy)
    proxy = np.where(within & mid,  PROXY_MODERATE, proxy)
    proxy = np.where(within & high, PROXY_SHALLOW,  proxy)

    # Refine: high flow accumulation → likely channelised → deepen proxy
    fa_finite = flow_accum[np.isfinite(flow_accum)]
    if fa_finite.size > 0:
        fa_thresh = float(np.percentile(fa_finite, FLOW_ACCUM_PERCENTILE))
        high_fa   = np.isfinite(flow_accum) & (flow_accum >= fa_thresh)
        proxy = np.where(within & high_fa & (proxy == PROXY_SHALLOW),  PROXY_MODERATE, proxy)
        proxy = np.where(within & high_fa & (proxy == PROXY_MODERATE), PROXY_DEEP,     proxy)

    return proxy


def raster_to_polygon(proxy: np.ndarray, transform, crs):
    """Vectorise each depth class to a GeoDataFrame."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    rows = []
    for geom, val in shapes(proxy, transform=transform):
        if val in (PROXY_SHALLOW, PROXY_MODERATE, PROXY_DEEP):
            rows.append({"geometry": shape(geom),
                         "depth_proxy": int(val),
                         "depth_label": {1: "shallow", 2: "moderate",
                                         3: "deep"}[int(val)]})
    gdf = gpd.GeoDataFrame(rows, crs=crs)
    if not gdf.empty:
        gdf["area_ha"] = (gdf.geometry.area / 1e4).round(4)
    return gdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    config    = load_config(config_path)
    flood_dir = get_processed_path("flood_extent", config)
    dem_dir   = get_processed_path("dem", config)
    depth_dir = get_processed_path("flood_depth_proxy", config)
    ensure_dir(depth_dir)

    fused_path  = flood_dir / "fused_flood_confidence.tif"
    rel_elev_path = dem_dir / "relative_elevation.tif"
    flow_path     = dem_dir / "flow_accumulation.tif"

    for p in [fused_path, rel_elev_path, flow_path]:
        if not p.exists():
            log.error("Required file missing: %s", p)
            return

    with Timer() as t:
        # ---- Load fused flood mask ----------------------------------------
        log.info("Loading fused flood mask…")
        with rasterio.open(fused_path) as src:
            fused_class = src.read(1).astype(np.uint8)
            profile     = src.profile.copy()
            transform   = src.transform
            crs         = src.crs

        # Flooded = classes 1 (SAR), 2 (optical), 3 (agreement)
        flood_mask = np.isin(fused_class, [1, 2, 3]).astype(np.uint8)
        n_flood = int(flood_mask.sum())
        log.info("  Flooded pixels: %d  (%.2f km²)",
                 n_flood, n_flood * 100 / 1e6)

        # ---- Load and align DEM derivatives to flood mask grid ------------
        log.info("Aligning DEM derivatives to flood mask grid (30 m → 10 m)…")
        rel_elev   = read_aligned(str(rel_elev_path), str(fused_path),
                                  Resampling.bilinear)
        flow_accum = read_aligned(str(flow_path),     str(fused_path),
                                  Resampling.bilinear)

        # Replace nodata
        rel_elev   = np.where(np.isfinite(rel_elev),   rel_elev,   0.0).astype(np.float32)
        flow_accum = np.where(np.isfinite(flow_accum), flow_accum, 0.0).astype(np.float32)
        log.info("  Relative elevation range: %.2f – %.2f m",
                 rel_elev[flood_mask == 1].min() if n_flood else 0,
                 rel_elev[flood_mask == 1].max() if n_flood else 0)

        # ---- Classify depth proxy ----------------------------------------
        log.info("Classifying flood depth proxy…")
        proxy = classify_depth_proxy(rel_elev, flow_accum, flood_mask)

        for cls, label in [(PROXY_SHALLOW, "shallow"),
                           (PROXY_MODERATE, "moderate"),
                           (PROXY_DEEP, "deep")]:
            n   = int((proxy == cls).sum())
            km2 = n * 100 / 1e6
            log.info("  %-12s  %8d px  %.3f km²", label, n, km2)

        # ---- Write proxy raster ------------------------------------------
        out_tif = depth_dir / "flood_depth_proxy.tif"
        out_profile = profile.copy()
        out_profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
        with rasterio.open(out_tif, "w", **out_profile) as dst:
            dst.write(proxy, 1)
            dst.update_tags(description=(
                "Flood depth proxy: 1=shallow, 2=moderate, 3=deep. "
                "NOT calibrated — terrain proxy only."))
        log.info("  Saved: %s", out_tif)

        # ---- Vectorise ---------------------------------------------------
        log.info("Vectorising depth proxy…")
        proxy_poly = raster_to_polygon(proxy, transform, crs)
        n_polys    = len(proxy_poly)
        log.info("  Polygons: %d", n_polys)

        out_gpkg = depth_dir / "flood_depth_proxy.gpkg"
        if not proxy_poly.empty:
            proxy_poly.to_file(str(out_gpkg), driver="GPKG")
            log.info("  Saved: %s", out_gpkg)

        # ---- Stats CSV ---------------------------------------------------
        stats_path = depth_dir / "flood_depth_proxy_stats.csv"
        with open(stats_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["depth_class", "label", "pixels", "area_km2",
                        "pct_of_flooded"])
            for cls, label in [(PROXY_SHALLOW, "shallow"),
                               (PROXY_MODERATE, "moderate"),
                               (PROXY_DEEP, "deep")]:
                n   = int((proxy == cls).sum())
                km2 = round(n * 100 / 1e6, 4)
                pct = round(100 * n / n_flood, 2) if n_flood else 0
                w.writerow([cls, label, n, km2, pct])
            w.writerow(["note", "PROXY ONLY — not calibrated flood depth", "", "", ""])
        log.info("  Saved: %s", stats_path)

    log.info("Flood depth proxy complete in %.1f s", t.elapsed)

    log_run(
        script=__file__,
        input_paths=[str(fused_path), str(rel_elev_path), str(flow_path)],
        output_paths=[str(out_tif), str(out_gpkg)],
        parameters={"rel_elev_low": REL_ELEV_LOW, "rel_elev_high": REL_ELEV_HIGH,
                    "flow_accum_pct": FLOW_ACCUM_PERCENTILE},
        status="success",
        runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create flood depth proxy")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
