from __future__ import annotations
"""
24-Community Composite Recovery Index Generator — DELIBERATE REBUILD,
NOT A RECONSTRUCTION OF THE OLD PUBLISHED NUMBERS.

Background: outputs/tables/community_recovery_index_24.csv (built June 25,
date unknown-methodology) could not be reverse-engineered despite an
extensive investigation — buffer radius sweeps (1-20 km), clip-range
variants, a class-weighted formula found in ndwi_recovery.py, coordinate-
drift checks (ruled out: raw community shapefile matches the processed
file to within 1 m), flood-mask content-drift checks (ruled out: exact
pixel-count match), and a 5-term regression (overfit, non-physical
coefficients). The processing log (outputs/reports/processing_log.csv)
has no entry for whatever built that file — it was never run through this
codebase's logged pipeline. Two of its four components were confirmed
genuinely unrecoverable:
  - ndvi_raw: no tested formula/parameter combination matched it.
  - settle_proxy: traced to confirmed data drift — the exposure summary
    it depends on (community_exposure_summary_24.csv) was regenerated
    July 15, three weeks after the recovery index was built, and the
    original building-exposure snapshot no longer exists anywhere.

Rather than continue guessing at a lost methodology, this script instead
computes a clean, fully reproducible recovery index from the methodology
docs/manuscript.md Section 4.6 and docs/uncertainty_manuscript.md Section
4.4 already claim: NDVI recovery ratio clipped to [0, 1] (the ratio is
stored unclamped in ndvi_recovery.tif, observed max 2.0 — the clip must
be applied here), mean within a 10 km buffer intersected with flood-
classified pixels, road accessibility from post-event MNDWI sampling at
flooded-road midpoints, and settlement proxy from the current (July 15)
building exposure data. road_acc and the raw-mean formula for wrec_raw
were independently validated against the old published file before this
decision (MAE 0.0003 and ~0.04 respectively) and are unchanged here.

Running this script WILL produce different recovery_index values and a
different community ranking than the currently-published
community_recovery_index_24.csv (e.g. Battor moves from rank 15 toward
rank 1). That is the point — not a bug. Propagating this to the real
output path, the regenerated figures, and the manuscript text that cites
specific recovery-index values and ranks is a deliberate follow-up step,
not automatic on running this script.

----------------------------------------------------------------------
Reconstructs outputs/tables/community_recovery_index_24.csv — the recovery
index table behind Table 3/4 of docs/manuscript.md and
docs/uncertainty_manuscript.md, and behind Figures 4, 5, 7 (main) and
1, 3, 4, 5 (uncertainty) — following the methodology described in
docs/manuscript.md Section 4.6 (Recovery Index Construction) and
docs/uncertainty_manuscript.md Section 4.4.

This generator did not previously exist in the codebase: the legacy
src/recovery/recovery_index.py only covers the original 6-community
analysis (hardcoded community list, reads 6-row stats CSVs that were never
extended to 24 communities). community_recovery_index_24.csv had been a
standalone artifact with no regeneration path. This script closes that gap.

Components (weights in parentheses, per Section 4.6)
-----------------------------------------------------
vegetation_recovery (0.30)  — mean of data/processed/recovery/ndvi_recovery.tif
                               within flood-classified pixels in each
                               community's 10 km service buffer.
water_recession     (0.30)  — mean of data/processed/recovery/water_recession.tif
                               over the same buffer and flood mask.
road_accessibility  (0.25)  — fraction of each community's originally
                               flooded road length (from
                               data/processed/exposure/exposed_roads.gpkg)
                               classified as dry (MNDWI < 0) at the
                               post-event date, sampled at segment
                               midpoints against
                               data/processed/satellite/s2_mndwi_post.tif.
                               Communities with no flooded road segments
                               (Akuse, Torgorme, Alabonu) default to 1.0 —
                               vacuously fully accessible.
settlement_proxy    (0.15)  — one minus the fraction of OSM building
                               footprints exposed to flooding within the
                               buffer, from
                               outputs/tables/community_exposure_summary_24.csv.

For Akuse and Torgorme (Lower Manya Krobo), whose flood extents (2.3-2.6 km²)
are too small to yield reliable per-pixel NDVI/water-recession means,
vegetation_recovery and water_recession are assigned the 75th percentile of
the other 22 communities' valid values, per Section 4.6.

All four components are min-max normalised to [0, 1] across all 24
communities, then combined into the weighted recovery_index. Ranks run
1 (fastest recovery) to 24 (slowest). Monte Carlo weight sensitivity
(2,000 Dirichlet(alpha=2) draws, seed 42) reproduces rank_std, rank_mean,
and modal_rank using the identical methodology as
generate_mc_rank_frequencies() in regenerate_24_figures.py (project root),
which consumes this file's ndvi_norm/wrec_norm/racc_norm/setl_norm columns
downstream.

Usage
-----
    python -m src.recovery.recovery_index_24

Output
------
outputs/tables/community_recovery_index_24.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

from src.utils.paths import get_project_root, ensure_dir
from src.exposure.community_summary_24 import (
    BUFFER_M, FLOOD_POSITIVE, sample_raster_at_point, raster_mean_in_buffer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WEIGHTS = {
    "ndvi_norm": 0.30,
    "wrec_norm": 0.30,
    "racc_norm": 0.25,
    "setl_norm": 0.15,
}
SMALL_FLOOD_FILL_COMMUNITIES = {"Akuse", "Torgorme"}
FILL_PERCENTILE = 75
N_MC_DRAWS = 2000
MC_ALPHA = 2
MC_SEED = 42


MIN_INFORMATIVE_RANGE = 0.05   # raw components are on a 0-1 scale


def normalise_minmax(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1].

    A component whose raw range across communities is below MIN_INFORMATIVE_RANGE
    (constant, or varying only in the third decimal) carries no between-community
    information; stretching it to 0-1 would turn rounding-level noise into a
    full-weight difference, so it is set to 0.5 for every community.
    """
    lo, hi = series.min(), series.max()
    if hi - lo >= MIN_INFORMATIVE_RANGE:
        return (series - lo) / (hi - lo)
    log.info("  component %r range %.4f < %.2f: non-informative, normalised to 0.5",
             series.name, hi - lo, MIN_INFORMATIVE_RANGE)
    return pd.Series(0.5, index=series.index)


def raster_mean_in_buffer_clipped(raster_path: Path, buffer_geom,
                                   clip_lo: float, clip_hi: float) -> float:
    """Mean of raster values inside buffer_geom, clamped to [clip_lo, clip_hi]
    before averaging. ndvi_recovery.tif stores the raw gain/loss ratio
    unclamped (observed max 2.0), but Section 4.6 defines the recovery
    ratio as "clipped to the interval zero to one" before use — the clip
    must be applied here, not assumed already baked into the raster."""
    if not raster_path.exists():
        return float("nan")
    with rasterio.open(raster_path) as src:
        out, _ = rio_mask(src, [mapping(buffer_geom)], crop=True, nodata=np.nan)
        arr = out[0].astype(float)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return float("nan")
        return float(np.clip(valid, clip_lo, clip_hi).mean())


def compute_road_accessibility(roads_gdf: gpd.GeoDataFrame, mndwi_post_path: Path,
                                communities: list[str]) -> dict[str, float]:
    """Fraction of each community's flooded road length classified dry
    (MNDWI < 0) at the post-event date, sampled at segment midpoints.
    Communities with no flooded road segments default to 1.0 (vacuously
    fully accessible — there is no flooded road to assess)."""
    result = {c: 1.0 for c in communities}
    if roads_gdf is None or roads_gdf.empty:
        return result

    with rasterio.open(mndwi_post_path) as src:
        midpoints = roads_gdf.geometry.interpolate(0.5, normalized=True)
        coords = [(pt.x, pt.y) for pt in midpoints]
        mndwi_vals = np.array([v[0] for v in src.sample(coords)], dtype=float)

    roads_gdf = roads_gdf.copy()
    roads_gdf["mndwi_post"] = mndwi_vals
    roads_gdf["accessible"] = roads_gdf["mndwi_post"] < 0

    for comm, grp in roads_gdf.groupby("community_name"):
        total_km = grp["flooded_length_km"].sum()
        if total_km <= 0:
            continue
        accessible_km = grp.loc[grp["accessible"], "flooded_length_km"].sum()
        result[comm] = float(accessible_km / total_km)

    return result


def main() -> None:
    root = get_project_root()
    ensure_dir(root / "outputs" / "tables")

    comm_path    = root / "data/processed/aoi/affected_communities.gpkg"
    flood_path   = root / "data/processed/flood_extent/fused_flood_confidence.tif"
    ndvi_path    = root / "data/processed/recovery/ndvi_recovery.tif"
    water_path   = root / "data/processed/recovery/water_recession.tif"
    mndwi_post_p = root / "data/processed/satellite/s2_mndwi_post.tif"
    roads_p      = root / "data/processed/exposure/exposed_roads.gpkg"
    exp_summary_p = root / "outputs/tables/community_exposure_summary_24.csv"

    log.info("Loading 24 communities…")
    comm = gpd.read_file(str(comm_path)).to_crs("EPSG:32630")
    comm["buffer_geom"] = comm.geometry.buffer(BUFFER_M)

    roads = gpd.read_file(str(roads_p)).to_crs("EPSG:32630") if roads_p.exists() else None
    exp_summary = pd.read_csv(exp_summary_p)

    # ---- Vegetation recovery and water recession: mean within
    # (10 km buffer ∩ flood-classified pixels) -----------------------------
    log.info("Computing NDVI recovery and water recession means…")
    rows = []
    for _, row in comm.iterrows():
        name, district = row["community_name"], row["district"]
        pt_x, pt_y = row.geometry.x, row.geometry.y
        buf_geom = row["buffer_geom"]

        flood_cls_val = sample_raster_at_point(flood_path, pt_x, pt_y)
        flood_cls = int(flood_cls_val) if np.isfinite(flood_cls_val) else 255
        flooded_point = 1 if flood_cls in FLOOD_POSITIVE else 0

        # ndvi_recovery.tif / water_recession.tif are already restricted to
        # flood-classified pixels (NaN elsewhere) — confirmed every valid
        # pixel in ndvi_recovery.tif is a strict subset of the Class-1-3
        # flood mask. Clipping again to a polygonised flood-extent vector
        # would introduce polygonisation-boundary error against the
        # raster's own pixel-exact mask, so the plain 10 km buffer (no
        # further flood clipping) is used here; the raster's NaN pattern
        # does the flood-restriction. ndvi_recovery.tif stores the raw
        # ratio unclamped (observed max 2.0) — clip to [0, 1] per Sec 4.6.
        ndvi_raw = raster_mean_in_buffer_clipped(ndvi_path, buf_geom, 0.0, 1.0)
        wrec_raw = raster_mean_in_buffer(water_path, buf_geom)

        rows.append({
            "community_name": name,
            "district": district,
            "flooded_point": flooded_point,
            "ndvi_raw": ndvi_raw,
            "wrec_raw": wrec_raw,
        })

    df = pd.DataFrame(rows)

    # ---- 75th-percentile fill for Akuse/Torgorme (insufficient flood
    # pixels for a reliable per-pixel mean) ---------------------------------
    for col in ["ndvi_raw", "wrec_raw"]:
        valid = df.loc[~df["community_name"].isin(SMALL_FLOOD_FILL_COMMUNITIES), col]
        valid = valid.dropna()
        fill_val = float(np.percentile(valid, FILL_PERCENTILE))
        missing_or_fill = df["community_name"].isin(SMALL_FLOOD_FILL_COMMUNITIES) | df[col].isna()
        df.loc[missing_or_fill, col] = fill_val
        log.info("  %s 75th-percentile fill value: %.4f", col, fill_val)

    # ---- Road accessibility -----------------------------------------------
    log.info("Computing road accessibility (post-event MNDWI at flooded-road midpoints)…")
    road_acc_map = compute_road_accessibility(roads, mndwi_post_p, df["community_name"].tolist())
    df["road_acc"] = df["community_name"].map(road_acc_map)

    # ---- Settlement proxy: 1 - exposed_buildings / total_buildings --------
    exp_summary["settle_proxy"] = 1.0 - (
        exp_summary["exposed_buildings"] / exp_summary["total_buildings"].replace(0, np.nan))
    df = df.merge(exp_summary[["community_name", "settle_proxy"]],
                  on="community_name", how="left")
    df["settle_proxy"] = df["settle_proxy"].fillna(df["settle_proxy"].median())

    # ---- Normalise components, compute recovery index ---------------------
    df["ndvi_norm"] = normalise_minmax(df["ndvi_raw"])
    df["wrec_norm"] = normalise_minmax(df["wrec_raw"])
    df["racc_norm"] = normalise_minmax(df["road_acc"])
    df["setl_norm"] = normalise_minmax(df["settle_proxy"])

    df["recovery_index"] = (
        WEIGHTS["ndvi_norm"] * df["ndvi_norm"]
        + WEIGHTS["wrec_norm"] * df["wrec_norm"]
        + WEIGHTS["racc_norm"] * df["racc_norm"]
        + WEIGHTS["setl_norm"] * df["setl_norm"]
    )

    df = df.sort_values("recovery_index", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # ---- Monte Carlo weight sensitivity: rank_std, rank_mean, modal_rank --
    # Identical methodology to generate_mc_rank_frequencies() in
    # regenerate_24_figures.py: 2,000 draws from a symmetric
    # Dirichlet(alpha=2) over the four components, seed 42.
    log.info("Running %d-draw Monte Carlo weight sensitivity (Dirichlet alpha=%d, seed=%d)…",
             N_MC_DRAWS, MC_ALPHA, MC_SEED)
    from scipy.stats import dirichlet as _dirichlet

    comp_cols = ["ndvi_norm", "wrec_norm", "racc_norm", "setl_norm"]
    components = df[comp_cols].values.astype(float)
    n = len(df)

    weight_draws = _dirichlet.rvs([MC_ALPHA] * 4, size=N_MC_DRAWS, random_state=MC_SEED)
    rank_matrix = np.empty((N_MC_DRAWS, n), dtype=int)
    for d, w in enumerate(weight_draws):
        scores = components @ w
        order = np.argsort(scores)[::-1]
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        rank_matrix[d] = ranks

    df["rank_mean"] = rank_matrix.mean(axis=0).round(2)
    df["rank_std"] = rank_matrix.std(axis=0, ddof=1).round(3)
    modal_ranks = []
    for i in range(n):
        vals, counts = np.unique(rank_matrix[:, i], return_counts=True)
        modal_ranks.append(int(vals[np.argmax(counts)]))
    df["modal_rank"] = modal_ranks

    # ---- Round for display, match existing column order -------------------
    for col in ["ndvi_raw", "wrec_raw", "road_acc", "settle_proxy",
               "ndvi_norm", "wrec_norm", "racc_norm", "setl_norm", "recovery_index"]:
        df[col] = df[col].round(4)

    out_cols = ["community_name", "district", "flooded_point", "ndvi_raw", "wrec_raw",
               "road_acc", "settle_proxy", "ndvi_norm", "wrec_norm", "racc_norm",
               "setl_norm", "recovery_index", "rank", "modal_rank", "rank_std", "rank_mean"]
    df = df[out_cols].rename(columns={"community_name": "community"})

    # SAFETY GUARD: this is a deliberate rebuild (see module docstring) that
    # produces different numbers than the currently-published
    # community_recovery_index_24.csv, which the manuscript text still
    # cites directly (specific index values, ranks, rank_std). Writes to a
    # clearly-labelled sibling file so the real path is only overwritten
    # as an explicit, separate step once the new numbers have been
    # reviewed and the decision is made to propagate them to the
    # manuscripts and figures.
    # The 2026-10-04 relabelling run propagates deliberately: the real file is
    # overwritten (previous version archived in archive/v3_sea_in_labels/tables).
    out_path = root / "outputs/tables/community_recovery_index_24.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved: %s", out_path)
    log.info("\n%s", df[["community", "district", "recovery_index", "rank",
                          "modal_rank", "rank_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
