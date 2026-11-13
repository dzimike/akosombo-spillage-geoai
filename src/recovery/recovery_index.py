from __future__ import annotations
"""
Phase 8 – Task 8.4: Composite Community Recovery Index

Combines per-community scores from Tasks 8.1–8.3 into a single normalised
Recovery Index (0–1) using weights from model_config.yaml.

Component mapping
-----------------
vegetation_recovery       → mean_recovery_ratio       (ndvi_recovery_stats.csv)
water_recession           → water_recession_index     (water_recession_stats.csv)
road_accessibility        → road_accessibility_index  (road_accessibility_stats.csv)
settlement_activity_proxy → pct_buildings_exposed⁻¹   (community_exposure_summary.gpkg)
                            (communities with fewer exposed buildings = quicker recovery proxy)

Weights (from model_config.yaml)
---------------------------------
vegetation_recovery:       0.30
water_recession:           0.30
road_accessibility:        0.25
settlement_activity_proxy: 0.15

Outputs
-------
outputs/tables/community_recovery_index.csv
outputs/maps/community_recovery_index_map.png
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.paths import load_config, get_project_root, get_output_path, \
    get_processed_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def normalise_col(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; constant series → 0.5."""
    lo, hi = series.min(), series.max()
    if hi > lo:
        return (series - lo) / (hi - lo)
    return pd.Series(0.5, index=series.index)


def compute_index(df: pd.DataFrame, weights: dict[str, float],
                  components: dict[str, str]) -> pd.Series:
    """Weighted normalised recovery index."""
    total_w = 0.0
    index   = pd.Series(np.zeros(len(df)), index=df.index)
    for comp_name, col in components.items():
        if col not in df.columns:
            log.warning("Component '%s' (col '%s') not found — skipping", comp_name, col)
            continue
        w  = weights.get(comp_name, 0.0)
        nv = normalise_col(df[col].fillna(df[col].median()))
        index    += w * nv
        total_w  += w
        log.info("  %-30s  weight=%.2f  range=[%.3f, %.3f]",
                 comp_name, w, float(df[col].min()), float(df[col].max()))
    return (index / total_w).round(4) if total_w > 0 else index


def save_map(df_idx: pd.DataFrame, communities_gdf, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    comms = communities_gdf.to_crs("EPSG:32630").merge(
        df_idx[["community_name", "recovery_index"]], on="community_name", how="left")

    cmap  = cm.RdYlGn
    norm  = mcolors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(10, 9))
    for _, row in comms.iterrows():
        color = cmap(norm(row.recovery_index)) if np.isfinite(row.recovery_index) else "grey"
        ax.scatter(row.geometry.x, row.geometry.y, s=400, color=color,
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(
            f"{row.community_name}\n(RI={row.recovery_index:.2f})",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9, fontweight="bold",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Recovery Index (0 = least recovered · 1 = most recovered)")

    ax.set_title("Community Recovery Index — Akosombo Spillage 2023", fontsize=13)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Map: %s", output_path)


def main(config_path: str | None = None) -> None:
    import geopandas as gpd

    proj_root = get_project_root()
    config    = load_config(config_path)

    with open(proj_root / "config" / "model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    weights = model_cfg["recovery_weights"]
    log.info("Recovery weights: %s", weights)

    rec_dir  = get_processed_path("recovery", config)
    exp_dir  = get_processed_path("exposure", config)
    comm_p   = get_processed_path("aoi/affected_communities.gpkg", config)

    ensure_dir(get_output_path("tables", config))
    ensure_dir(get_output_path("maps", config))

    # ---- Load component tables ------------------------------------------
    ndvi_stats_p  = rec_dir / "ndvi_recovery_stats.csv"
    water_stats_p = rec_dir / "water_recession_stats.csv"
    road_stats_p  = rec_dir / "road_accessibility_stats.csv"
    exp_summary_p = exp_dir / "community_exposure_summary.gpkg"

    dfs = {}
    for name, path in [("ndvi",  ndvi_stats_p),
                       ("water", water_stats_p),
                       ("road",  road_stats_p)]:
        if path.exists():
            dfs[name] = pd.read_csv(path)
            log.info("Loaded %s stats: %d rows", name, len(dfs[name]))
        else:
            log.warning("Missing: %s — component will be skipped", path)

    # Settlement activity proxy: inverse of pct_buildings_exposed
    # (less exposure → faster recovery proxy)
    if exp_summary_p.exists():
        exp_df = gpd.read_file(str(exp_summary_p))[
            ["community_name", "pct_buildings_exposed"]].copy()
        exp_df["settlement_proxy"] = 1.0 - (
            exp_df["pct_buildings_exposed"].fillna(0) / 100.0)
        dfs["settlement"] = exp_df
    else:
        log.warning("Exposure summary not found — settlement proxy skipped")

    # ---- Merge all components on community_name -------------------------
    all_comms = ["Mepe", "Battor", "Sogakope", "Mafi Kumase", "Adidome", "Ada Foah"]
    df = pd.DataFrame({"community_name": all_comms})

    if "ndvi"  in dfs:
        df = df.merge(dfs["ndvi"][["community_name","mean_recovery_ratio"]],
                      on="community_name", how="left")
    if "water" in dfs:
        df = df.merge(dfs["water"][["community_name","water_recession_index"]],
                      on="community_name", how="left")
    if "road"  in dfs:
        df = df.merge(dfs["road"][["community_name","road_accessibility_index"]],
                      on="community_name", how="left")
    if "settlement" in dfs:
        df = df.merge(dfs["settlement"][["community_name","settlement_proxy"]],
                      on="community_name", how="left")

    # ---- Compute index --------------------------------------------------
    components = {
        "vegetation_recovery":      "mean_recovery_ratio",
        "water_recession":          "water_recession_index",
        "road_accessibility":       "road_accessibility_index",
        "settlement_activity_proxy":"settlement_proxy",
    }
    with Timer() as t:
        df["recovery_index"] = compute_index(df, weights, components)

    df = df.sort_values("recovery_index", ascending=False).reset_index(drop=True)
    df["recovery_rank"] = range(1, len(df) + 1)

    out_csv = get_output_path("tables/community_recovery_index.csv", config)
    df.to_csv(out_csv, index=False)
    log.info("\nCommunity Recovery Index:\n%s",
             df[["community_name","recovery_index","recovery_rank"]].to_string(index=False))

    # ---- Map ------------------------------------------------------------
    if comm_p.exists():
        comms = gpd.read_file(comm_p)
        save_map(df, comms, get_output_path("maps/community_recovery_index_map.png", config))

    log.info("Recovery index complete in %.1f s", t.elapsed)
    log_run(
        script=__file__,
        input_paths=[str(p) for p in [ndvi_stats_p, water_stats_p, road_stats_p] if p.exists()],
        output_paths=[str(out_csv)],
        parameters=weights,
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Composite recovery index")
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
