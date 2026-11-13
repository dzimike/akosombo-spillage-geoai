from __future__ import annotations
"""
Phase 8 – Task 8.3: Road Accessibility Recovery

For each previously flooded road segment, checks whether the road corridor
was dry in the post-event period using two signals:
  1. Post-event MNDWI < 0  (optical dryness proxy, primary)
  2. Post-event SAR VV closer to pre-event level (SAR water recession, secondary)

Accessible = MNDWI_post < 0 at road pixel OR SAR change normalised < threshold.

Per-community metric
---------------------
road_accessibility_index = accessible_km / total_flooded_km
  1.0 = all flooded roads now accessible
  0.0 = all flooded roads still inundated

Outputs
-------
data/processed/recovery/road_accessibility_stats.csv
outputs/maps/road_accessibility_map.png
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

from src.utils.paths import load_config, get_processed_path, get_output_path, ensure_dir
from src.utils.logging_utils import Timer, log_run
from src.utils.raster import read_aligned
from rasterio.enums import Resampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WATER_THR          = 0.0    # MNDWI < this = dry
COMMUNITY_BUFFER_M = 10_000


def road_accessibility_per_segment(
    exposed_roads: gpd.GeoDataFrame,
    mndwi_post: np.ndarray,
    template_path: str,
) -> gpd.GeoDataFrame:
    """
    Rasterize each road segment centroid and sample post-event MNDWI.
    Accessible = MNDWI_post < WATER_THR at the road pixel.
    """
    with rasterio.open(template_path) as src:
        tf, H, W = src.transform, src.height, src.width

    roads = exposed_roads.copy()

    # Sample MNDWI_post at road midpoints
    def sample_raster(geom, arr, transform):
        try:
            from rasterio.transform import rowcol
            mid = geom.interpolate(0.5, normalized=True)
            r, c = rowcol(transform, mid.x, mid.y)
            if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                return float(arr[r, c])
        except Exception:
            pass
        return np.nan

    roads["mndwi_post"] = [
        sample_raster(geom, mndwi_post, tf) for geom in roads.geometry
    ]
    roads["accessible"] = (roads["mndwi_post"] < WATER_THR).astype(int)
    roads.loc[~np.isfinite(roads["mndwi_post"]), "accessible"] = -1  # unknown
    return roads


def community_accessibility(roads: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate per-community road accessibility from road-level results."""
    rows = []
    for comm in roads["community_name"].dropna().unique():
        seg = roads[roads["community_name"] == comm]
        flooded_km   = float(seg["flooded_length_km"].sum())
        known        = seg[seg["accessible"] >= 0]
        accessible_km = float(known[known["accessible"] == 1]["flooded_length_km"].sum())
        unknown_km    = float(seg[seg["accessible"] < 0]["flooded_length_km"].sum())
        index = accessible_km / flooded_km if flooded_km > 0 else np.nan
        rows.append({
            "community_name":        comm,
            "total_flooded_km":      round(flooded_km, 3),
            "accessible_km":         round(accessible_km, 3),
            "unknown_km":            round(unknown_km, 3),
            "road_accessibility_index": round(index, 4) if np.isfinite(index) else np.nan,
        })
    # Add any communities without flooded roads
    all_comms = ["Mepe", "Battor", "Sogakope", "Mafi Kumase", "Adidome", "Ada Foah"]
    present   = {r["community_name"] for r in rows}
    for c in all_comms:
        if c not in present:
            rows.append({"community_name": c, "total_flooded_km": 0.0,
                         "accessible_km": 0.0, "unknown_km": 0.0,
                         "road_accessibility_index": 1.0})
    return pd.DataFrame(rows).sort_values("community_name").reset_index(drop=True)


def save_map(roads: gpd.GeoDataFrame, communities: gpd.GeoDataFrame,
             output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12, 10))

    color_map = {1: "#2196F3", 0: "#F44336", -1: "#9E9E9E"}
    label_map = {1: "Accessible (dry)", 0: "Still inundated", -1: "Unknown"}

    for status, group in roads.groupby("accessible"):
        group.to_crs("EPSG:32630").plot(
            ax=ax, color=color_map.get(status, "#9E9E9E"),
            linewidth=0.8, label=label_map.get(status, str(status)))

    communities.to_crs("EPSG:32630").plot(
        ax=ax, color="black", markersize=40, marker="*", zorder=5)
    for _, row in communities.to_crs("EPSG:32630").iterrows():
        ax.annotate(row.community_name,
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=l)
               for s, (c, l) in {1: ("#2196F3","Accessible"),
                                  0: ("#F44336","Still inundated"),
                                  -1: ("#9E9E9E","Unknown")}.items()]
    ax.legend(handles=patches, loc="lower right")
    ax.set_title("Road Accessibility Recovery — Akosombo Spillage 2023", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Map: %s", output_path)


def main(config_path: str | None = None) -> None:
    config  = load_config(config_path)
    sat_dir = get_processed_path("satellite", config)
    rec_dir = get_processed_path("recovery", config)
    ensure_dir(rec_dir)
    ensure_dir(get_output_path("maps", config))

    template      = sat_dir / "s2_ndwi_event.tif"
    mndwi_post_p  = sat_dir / "s2_mndwi_post.tif"
    roads_p       = get_processed_path("exposure/exposed_roads.gpkg", config)
    comm_p        = get_processed_path("aoi/affected_communities.gpkg", config)

    for p in [mndwi_post_p, roads_p]:
        if not p.exists():
            log.error("Missing: %s", p); return

    with Timer() as t:
        log.info("Loading post-event MNDWI…")
        with rasterio.open(mndwi_post_p) as src:
            mndwi_post = src.read(1).astype(np.float32)
            nd = src.nodata
        if nd is not None:
            mndwi_post = np.where(mndwi_post == nd, np.nan, mndwi_post)

        log.info("Loading exposed roads (%s)…", roads_p.name)
        roads = gpd.read_file(str(roads_p))
        # Project roads to same CRS as raster
        with rasterio.open(template) as src:
            raster_crs = src.crs
        roads = roads.to_crs(raster_crs)
        log.info("  Exposed road segments: %d", len(roads))

        log.info("Assessing accessibility from post-event MNDWI…")
        roads = road_accessibility_per_segment(roads, mndwi_post, str(template))

        n_acc = int((roads["accessible"] == 1).sum())
        n_inn = int((roads["accessible"] == 0).sum())
        n_unk = int((roads["accessible"] == -1).sum())
        acc_km = float(roads[roads["accessible"] == 1]["flooded_length_km"].sum())
        tot_km = float(roads["flooded_length_km"].sum())
        log.info("  Accessible: %d segments (%.1f km)", n_acc, acc_km)
        log.info("  Still inundated: %d segments", n_inn)
        log.info("  Unknown: %d segments", n_unk)
        log.info("  Overall accessibility index: %.3f", acc_km / tot_km if tot_km else 0)

        # ---- Community table ---------------------------------------------
        df_comm    = community_accessibility(roads)
        stats_path = rec_dir / "road_accessibility_stats.csv"
        df_comm.to_csv(stats_path, index=False)
        log.info("Community accessibility:\n%s", df_comm.to_string(index=False))

        # ---- Map ---------------------------------------------------------
        if comm_p.exists():
            comms = gpd.read_file(comm_p).to_crs(raster_crs)
        else:
            comms = gpd.GeoDataFrame()
        save_map(roads, comms,
                 get_output_path("maps/road_accessibility_map.png", config))

    log.info("Road accessibility complete in %.1f s", t.elapsed)
    log_run(
        script=__file__,
        input_paths=[str(mndwi_post_p), str(roads_p)],
        output_paths=[str(stats_path)],
        parameters={"water_thr": WATER_THR,
                    "overall_accessibility": round(acc_km / tot_km, 4) if tot_km else None},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road accessibility recovery")
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
