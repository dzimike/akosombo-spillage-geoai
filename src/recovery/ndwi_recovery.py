from __future__ import annotations
"""
Phase 8 – Task 8.2: Water Recession Monitoring (MNDWI)

Measures how much floodwater has receded between the event window and the
post-event window using MNDWI (Modified NDWI).

Recession score = MNDWI_event − MNDWI_post
  Positive = water signal weakened (receded)
  Negative = water signal strengthened (still inundated / new water)

Recession classes (over flooded pixels only)
--------------------------------------------
0 = was_dry       (MNDWI_event ≤ 0 — not water during event)
1 = fully_receded (MNDWI_post < WATER_THR and was flooded)
2 = partially     (MNDWI_post reduced but still ≥ WATER_THR)
3 = still_flooded (MNDWI_post ≥ WATER_THR, barely changed)
9 = nodata

Outputs
-------
data/processed/recovery/water_recession.tif       — float32 recession_score
data/processed/recovery/water_recession_stats.csv — per-community stats
outputs/maps/water_recession_map.png
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
import pandas as pd

from src.utils.paths import load_config, get_processed_path, get_output_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WATER_THR          = 0.0    # MNDWI > 0 = open water
COMMUNITY_BUFFER_M = 10_000


def compute_recession(mndwi_event, mndwi_post):
    """Return recession_score = mndwi_event − mndwi_post (positive = receded)."""
    return (mndwi_event - mndwi_post).astype(np.float32)


def classify_recession(mndwi_event, mndwi_post, flood_mask):
    cls = np.full(mndwi_event.shape, 9, dtype=np.uint8)
    was_water = np.isfinite(mndwi_event) & (mndwi_event > WATER_THR) & flood_mask
    cls = np.where(flood_mask & ~was_water, 0, cls)                          # was_dry
    cls = np.where(was_water & np.isfinite(mndwi_post)
                   & (mndwi_post < WATER_THR),                1, cls)         # fully_receded
    cls = np.where(was_water & np.isfinite(mndwi_post)
                   & (mndwi_post >= WATER_THR)
                   & (mndwi_post < mndwi_event - 0.05),       2, cls)         # partially
    cls = np.where(was_water & np.isfinite(mndwi_post)
                   & (mndwi_post >= mndwi_event - 0.05),      3, cls)         # still_flooded
    cls = np.where(~np.isfinite(mndwi_event) | ~np.isfinite(mndwi_post), 9, cls)
    return cls


def community_stats(recession_score, cls, flood_mask, communities_gdf,
                    template_path: str) -> pd.DataFrame:
    import rasterio.features
    from shapely.geometry import mapping

    comms = communities_gdf.to_crs("EPSG:32630")
    comms = comms.copy()
    comms["geometry"] = comms.geometry.buffer(COMMUNITY_BUFFER_M)

    with rasterio.open(template_path) as src:
        H, W, tf = src.height, src.width, src.transform

    rows = []
    for _, row in comms.iterrows():
        buf = rasterio.features.geometry_mask(
            [mapping(row.geometry)], out_shape=(H, W), transform=tf, invert=True)
        within = buf & flood_mask & np.isfinite(recession_score)
        n = int(within.sum())
        rows.append({
            "community_name":        row.community_name,
            "n_flood_pixels":        n,
            "mean_recession_score":  round(float(recession_score[within].mean()), 4) if n else np.nan,
            "pct_fully_receded":     round(100 * float((cls[within] == 1).mean()), 2) if n else np.nan,
            "pct_partially_receded": round(100 * float((cls[within] == 2).mean()), 2) if n else np.nan,
            "pct_still_flooded":     round(100 * float((cls[within] == 3).mean()), 2) if n else np.nan,
            "water_recession_index": round(
                float((cls[within] == 1).mean()) * 1.0 +
                float((cls[within] == 2).mean()) * 0.5, 4) if n else np.nan,
        })
    return pd.DataFrame(rows)


def save_map(recession_score, flood_mask, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    display = np.where(flood_mask & np.isfinite(recession_score), recession_score, np.nan)
    masked  = np.ma.masked_invalid(display)
    cmap    = plt.cm.Blues
    cmap.set_bad("lightgrey", alpha=0.35)

    fig, ax = plt.subplots(figsize=(10, 12))
    vmax = float(np.nanpercentile(recession_score[flood_mask], 95)) if flood_mask.any() else 1.0
    im   = ax.imshow(masked, cmap=cmap, vmin=-0.2, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Water Recession Score (MNDWI_event − MNDWI_post)")
    ax.set_title("Water Recession — Akosombo Spillage 2023\n"
                 "Post-flood period (Nov 2023 – Jan 2024)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Map: %s", output_path)


def main(config_path: str | None = None) -> None:
    import geopandas as gpd

    config  = load_config(config_path)
    sat_dir = get_processed_path("satellite", config)
    rec_dir = get_processed_path("recovery", config)
    ensure_dir(rec_dir)
    ensure_dir(get_output_path("maps", config))

    template    = sat_dir / "s2_ndwi_event.tif"
    mndwi_evt_p = sat_dir / "s2_mndwi_event.tif"
    mndwi_pst_p = sat_dir / "s2_mndwi_post.tif"
    flood_p     = get_processed_path("flood_extent/fused_flood_confidence.tif", config)
    comm_p      = get_processed_path("aoi/affected_communities.gpkg", config)

    for p in [mndwi_evt_p, mndwi_pst_p, flood_p]:
        if not p.exists():
            log.error("Missing: %s", p); return

    with Timer() as t:
        log.info("Loading MNDWI rasters…")

        def load(path):
            with rasterio.open(path) as s:
                arr = s.read(1).astype(np.float32)
                nd  = s.nodata
            return np.where((nd is not None) & (arr == nd), np.nan, arr)

        mndwi_event = load(mndwi_evt_p)
        mndwi_post  = load(mndwi_pst_p)

        with rasterio.open(flood_p) as src:
            flood_class = src.read(1).astype(np.uint8)
        flood_mask = np.isin(flood_class, [1, 2, 3])
        log.info("Flooded pixels: %d", int(flood_mask.sum()))

        recession = compute_recession(mndwi_event, mndwi_post)
        cls       = classify_recession(mndwi_event, mndwi_post, flood_mask)

        valid = flood_mask & np.isfinite(recession)
        log.info("Valid flooded pixels for recession: %d", int(valid.sum()))
        if valid.any():
            log.info("  Mean recession score: %.4f", float(recession[valid].mean()))
            for code, label in [(0,"was_dry"),(1,"fully_receded"),
                                (2,"partially"),(3,"still_flooded"),(9,"nodata")]:
                n   = int((cls[flood_mask] == code).sum())
                pct = 100 * n / max(int(flood_mask.sum()), 1)
                log.info("  Class %d %-15s %8d px  (%.1f%%)", code, label, n, pct)

        # ---- Write raster ------------------------------------------------
        with rasterio.open(template) as src:
            profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
        out_tif = rec_dir / "water_recession.tif"
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(np.where(flood_mask & np.isfinite(recession), recession, np.nan)
                      .astype(np.float32), 1)
            dst.update_tags(description="Water recession score = MNDWI_event - MNDWI_post; positive = receded")
        log.info("Raster: %s", out_tif)

        # ---- Community stats ---------------------------------------------
        stats_path = None
        if comm_p.exists():
            comms      = gpd.read_file(comm_p)
            df_stats   = community_stats(recession, cls, flood_mask,
                                         comms, str(template))
            stats_path = rec_dir / "water_recession_stats.csv"
            df_stats.to_csv(stats_path, index=False)
            log.info("Community stats:\n%s", df_stats.to_string(index=False))

        # ---- Map ---------------------------------------------------------
        save_map(recession, flood_mask,
                 get_output_path("maps/water_recession_map.png", config))

    log.info("Water recession complete in %.1f s", t.elapsed)
    log_run(
        script=__file__,
        input_paths=[str(mndwi_evt_p), str(mndwi_pst_p)],
        output_paths=[str(out_tif)] + ([str(stats_path)] if stats_path else []),
        parameters={"water_thr": WATER_THR},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water recession monitoring")
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
