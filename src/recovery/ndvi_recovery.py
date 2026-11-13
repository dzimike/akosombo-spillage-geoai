from __future__ import annotations
"""
Phase 8 – Task 8.1: Vegetation Recovery (NDVI)

Computes per-pixel NDVI loss (pre→event) and recovery (event→post) over
previously flooded pixels and classifies each into a recovery category.

Recovery ratio = (ndvi_post − ndvi_event) / (ndvi_pre − ndvi_event)
  1.0 = fully recovered  |  0.0 = zero recovery  |  >1.0 = above-pre level

Recovery classes
----------------
0 = no_loss        (ndvi_pre − ndvi_event < LOSS_THRESHOLD)
1 = full_recovery  (recovery_ratio ≥ 0.80)
2 = partial        (0.30 ≤ recovery_ratio < 0.80)
3 = no_recovery    (recovery_ratio < 0.30)
9 = nodata / outside flood extent

Outputs
-------
data/processed/recovery/ndvi_recovery.tif       — float32 recovery_ratio
data/processed/recovery/ndvi_recovery_stats.csv — per-community stats
outputs/maps/ndvi_recovery_map.png
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

LOSS_THRESHOLD       = 0.05   # min NDVI drop counted as vegetation loss
FULL_RECOVERY_THR    = 0.80   # recovery_ratio ≥ this = fully recovered
PARTIAL_RECOVERY_THR = 0.30   # recovery_ratio ≥ this = partial
COMMUNITY_BUFFER_M   = 10_000


def compute_recovery(ndvi_pre, ndvi_event, ndvi_post):
    loss  = ndvi_pre - ndvi_event                       # positive where NDVI dropped
    gain  = ndvi_post - ndvi_event                      # positive where NDVI bounced back
    denom = np.where(np.abs(loss) > 0.001, loss, np.nan)
    ratio = np.clip(gain / denom, 0.0, 2.0)
    return ratio.astype(np.float32), loss.astype(np.float32)


def classify(ratio, loss):
    cls = np.full_like(ratio, 9, dtype=np.uint8)
    cls = np.where(loss < LOSS_THRESHOLD, 0, cls)
    cls = np.where((loss >= LOSS_THRESHOLD) & (ratio >= FULL_RECOVERY_THR),    1, cls)
    cls = np.where((loss >= LOSS_THRESHOLD) & (ratio >= PARTIAL_RECOVERY_THR)
                   & (ratio < FULL_RECOVERY_THR),                               2, cls)
    cls = np.where((loss >= LOSS_THRESHOLD) & (ratio < PARTIAL_RECOVERY_THR),  3, cls)
    cls = np.where(~np.isfinite(ratio) | ~np.isfinite(loss), 9, cls)
    return cls


def community_stats(ratio, loss, cls, flood_mask, communities_gdf,
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
        buf_mask = rasterio.features.geometry_mask(
            [mapping(row.geometry)], out_shape=(H, W), transform=tf, invert=True)
        within = buf_mask & flood_mask & np.isfinite(ratio)
        n = int(within.sum())
        rows.append({
            "community_name":      row.community_name,
            "n_flood_pixels":      n,
            "mean_ndvi_loss":      round(float(loss[within].mean()), 4) if n else np.nan,
            "mean_recovery_ratio": round(float(ratio[within].mean()), 4) if n else np.nan,
            "pct_full_recovery":   round(100 * float((cls[within] == 1).mean()), 2) if n else np.nan,
            "pct_partial":         round(100 * float((cls[within] == 2).mean()), 2) if n else np.nan,
            "pct_no_recovery":     round(100 * float((cls[within] == 3).mean()), 2) if n else np.nan,
        })
    return pd.DataFrame(rows)


def save_map(ratio, flood_mask, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    display = np.where(flood_mask & np.isfinite(ratio), ratio, np.nan)
    masked  = np.ma.masked_invalid(display)
    cmap    = plt.cm.RdYlGn
    cmap.set_bad("lightgrey", alpha=0.35)

    fig, ax = plt.subplots(figsize=(10, 12))
    im   = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("NDVI Recovery Ratio  (0 = no recovery · 1 = full)")
    ax.set_title("Vegetation Recovery — Akosombo Spillage 2023\n"
                 "NDVI change: event-period → post-flood", fontsize=12)
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

    template   = sat_dir / "s2_ndwi_event.tif"
    ndvi_pre_p = sat_dir / "s2_ndvi_pre.tif"
    ndvi_evt_p = sat_dir / "s2_ndvi_event.tif"
    ndvi_pst_p = sat_dir / "s2_ndvi_post.tif"
    flood_p    = get_processed_path("flood_extent/fused_flood_confidence.tif", config)
    comm_p     = get_processed_path("aoi/affected_communities.gpkg", config)

    for p in [ndvi_pre_p, ndvi_evt_p, ndvi_pst_p, flood_p]:
        if not p.exists():
            log.error("Missing: %s", p); return

    with Timer() as t:
        log.info("Loading NDVI rasters…")

        def load(path):
            with rasterio.open(path) as s:
                arr = s.read(1).astype(np.float32)
                nd  = s.nodata
            return np.where((nd is not None) & (arr == nd), np.nan, arr)

        ndvi_pre   = load(ndvi_pre_p)
        ndvi_event = load(ndvi_evt_p)
        ndvi_post  = load(ndvi_pst_p)

        with rasterio.open(flood_p) as src:
            flood_class = src.read(1).astype(np.uint8)
        flood_mask = np.isin(flood_class, [1, 2, 3])
        log.info("Flooded pixels: %d", int(flood_mask.sum()))

        ratio, loss = compute_recovery(ndvi_pre, ndvi_event, ndvi_post)
        cls = classify(ratio, loss)

        valid = flood_mask & np.isfinite(ratio)
        log.info("Valid flooded pixels for recovery: %d", int(valid.sum()))
        if valid.any():
            log.info("  Mean NDVI loss:    %.4f", float(loss[valid].mean()))
            log.info("  Mean recovery ratio: %.4f", float(ratio[valid].mean()))
            for code, label in [(0,"no_loss"),(1,"full"),(2,"partial"),(3,"no_recovery")]:
                n   = int((cls[valid] == code).sum())
                pct = 100 * n / max(int(valid.sum()), 1)
                log.info("  Class %d %-15s %8d px  (%.1f%%)", code, label, n, pct)

        # ---- Write raster ------------------------------------------------
        with rasterio.open(template) as src:
            profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
        out_tif = rec_dir / "ndvi_recovery.tif"
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(np.where(flood_mask & np.isfinite(ratio), ratio, np.nan)
                      .astype(np.float32), 1)
            dst.update_tags(
                description="NDVI recovery ratio (post-event)/(pre-event loss); 1=full recovery")
        log.info("Raster: %s", out_tif)

        # ---- Community stats ---------------------------------------------
        stats_path = None
        if comm_p.exists():
            comms      = gpd.read_file(comm_p)
            df_stats   = community_stats(ratio, loss, cls, flood_mask,
                                         comms, str(template))
            stats_path = rec_dir / "ndvi_recovery_stats.csv"
            df_stats.to_csv(stats_path, index=False)
            log.info("Community stats:\n%s", df_stats.to_string(index=False))

        # ---- Map ---------------------------------------------------------
        save_map(ratio, flood_mask, get_output_path("maps/ndvi_recovery_map.png", config))

    log.info("NDVI recovery complete in %.1f s", t.elapsed)
    log_run(
        script=__file__,
        input_paths=[str(ndvi_pre_p), str(ndvi_evt_p), str(ndvi_pst_p)],
        output_paths=[str(out_tif)] + ([str(stats_path)] if stats_path else []),
        parameters={"loss_threshold": LOSS_THRESHOLD,
                    "full_recovery_thr": FULL_RECOVERY_THR},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDVI vegetation recovery")
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
