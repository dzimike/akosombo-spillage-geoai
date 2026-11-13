from __future__ import annotations
"""
Recovery index v2 (2026-10-14): non-overlapping service areas, informative components only, SAR-based persistence.

What v1 (recovery_index_24.py) had wrong, found when it was re-run on the change-based flood mask:
  * 10 km community buffers overlap, so neighbouring communities (all seven South Tongu communities) average nearly the
    same flood pixels and differ only by how the buffers clip them;
  * road accessibility (every community 1.0: no flooded road segment inside the buffer is wet post-event) and the
    settlement proxy (0.998-1.000) carry no between-community information and were set to 0.5, so the index was a
    two-component NDVI + MNDWI score, with the optical components limited by cloud and by pre-event validity of 47 %.

v2:
  service areas   Voronoi cells of the 24 communities intersected with their 10 km buffers (non-overlapping), on the
                  project grid; communities outside the raster (Akuse, Torgorme) are not assessed
  flood pixels    final fused flood (classes 1-3) inside each service area; areas with < MIN_FLOOD_PX pixels (5 ha)
                  are not assessed
  components      sar_recovery     1 - net persistence, where net persistence is the flagged share (dVV & dVH < -3 dB)
                                       of flood pixels minus the flagged share of non-flooded land in the same service
                                       area (a control for the dry-season change that affects all land after November),
                                       on post-event dates (16 Nov 2023 - 31 Jan 2024), divided by the same net share on
                                       release-period dates (15 Sep - 30 Oct); clipped to [0, 1]
                  ndvi_recovery    mean of ndvi_recovery.tif (post/event NDVI ratio clipped to [0,1]) on flood pixels
                  mndwi_recession  mean of water_recession.tif on flood pixels
  weights         0.4 / 0.3 / 0.3 (SAR is observed on every date regardless of cloud); 2,000 Dirichlet(2) draws give
                  rank_mean and rank_std
Outputs: outputs/tables/community_recovery_index_v2.csv, outputs/tables/recovery_v1_v2_comparison.csv
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.stats import dirichlet, spearmanr
from shapely.ops import voronoi_diagram
from shapely.geometry import MultiPoint

from src.utils.paths import get_project_root, get_output_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
BUFFER_M = 10_000
MIN_FLOOD_PX = 500
WEIGHTS = np.array([0.4, 0.3, 0.3])
POST_START, EVENT_START, EVENT_END = "2023-11-16", "2023-09-15", "2023-10-30"


def main() -> None:
    root = get_project_root()
    sat, fl, rec = root / "data/processed/satellite", root / "data/processed/flood_extent", root / "data/processed/recovery"
    with rasterio.open(fl / "fused_flood_confidence.tif") as s:
        fu, tr, crs, shape = s.read(1), s.transform, s.crs, s.shape
    flood = np.isin(fu, [1, 2, 3])
    comm = gpd.read_file(root / "data/processed/aoi/affected_communities.gpkg").to_crs(crs).reset_index(drop=True)
    pts = MultiPoint(list(comm.geometry))
    env = pts.buffer(BUFFER_M * 3)
    cells = list(voronoi_diagram(pts, envelope=env).geoms)
    cell_of = {}
    for i, g in enumerate(comm.geometry):
        for c in cells:
            if c.contains(g):
                cell_of[i] = c; break
    area_geoms = [cell_of[i].intersection(g.buffer(BUFFER_M)) for i, g in enumerate(comm.geometry)]
    lab = rasterize(((g, i + 1) for i, g in enumerate(area_geoms)), out_shape=shape, transform=tr, fill=0, dtype="uint8")
    n = len(comm)
    nflood = np.bincount(lab[flood], minlength=n + 1)[1:]
    log.info("flood pixels by service area: %s", dict(zip(comm.community_name, nflood)))

    # SAR persistence from the per-date flags
    from rasterio.enums import Resampling
    from src.utils.raster import read_aligned
    slope = read_aligned(str(root / "data/processed/dem/slope.tif"), str(fl / "fused_flood_confidence.tif"), Resampling.bilinear)
    control = ~flood & (fu != 4) & ~(np.isfinite(slope) & (slope > 5))
    bands = pd.read_csv(sat / "s1_change_by_date_bands.csv")
    ev_share, po_share, ev_ctl, po_ctl = [], [], [], []
    with rasterio.open(sat / "s1_change_by_date.tif") as src:
        for i, (orbit, date) in enumerate(zip(bands.orbit, bands.date), 1):
            if not (EVENT_START <= date <= EVENT_END or date >= POST_START):
                continue
            code = src.read(i)
            ok = (code & 1) > 0
            fl4 = (code & 4) > 0
            def share(m):
                v = ok & m
                den = np.bincount(lab[v], minlength=n + 1)[1:].astype(float)
                num = np.bincount(lab[fl4 & v], minlength=n + 1)[1:].astype(float)
                return np.where(den >= 100, num / np.maximum(den, 1), np.nan)
            post = date >= POST_START
            (po_share if post else ev_share).append(share(flood))
            (po_ctl if post else ev_ctl).append(share(control))
    ev = np.nanmean(np.vstack(ev_share), axis=0); po = np.nanmean(np.vstack(po_share), axis=0)
    evc = np.nanmean(np.vstack(ev_ctl), axis=0); poc = np.nanmean(np.vstack(po_ctl), axis=0)
    persist = (po - poc) / np.maximum(ev - evc, 1e-6)

    def mean_on_flood(path, clip=None):
        with rasterio.open(path) as s_:
            a = s_.read(1).astype(float)
        a = np.where(flood & np.isfinite(a), a, np.nan)
        if clip:
            a = np.clip(a, *clip)
        m = np.isfinite(a)
        return np.bincount(lab[m], weights=a[m], minlength=n + 1)[1:] / np.maximum(np.bincount(lab[m], minlength=n + 1)[1:], 1)
    ndvi = mean_on_flood(rec / "ndvi_recovery.tif", (0, 1))
    wrec = mean_on_flood(rec / "water_recession.tif")
    df = pd.DataFrame({"community": comm.community_name, "district": comm.district, "flood_px": nflood,
                       "event_flag_share": ev, "post_flag_share": po, "event_control_share": evc, "post_control_share": poc,
                       "persistence_ratio": persist,
                       "sar_recovery": np.clip(1 - persist, 0, 1), "ndvi_recovery": ndvi, "mndwi_recession": wrec})
    ok = df.flood_px >= MIN_FLOOD_PX
    df.loc[~ok, ["sar_recovery", "ndvi_recovery", "mndwi_recession"]] = np.nan
    comps = ["sar_recovery", "ndvi_recovery", "mndwi_recession"]
    z = df[comps].copy()
    for c in comps:
        lo, hi = z[c].min(), z[c].max()
        z[c] = (z[c] - lo) / (hi - lo) if hi - lo > 1e-9 else 0.5
    df["recovery_index_v2"] = (z.values * WEIGHTS).sum(axis=1)
    df.loc[~ok, "recovery_index_v2"] = np.nan
    a = df.loc[ok, "recovery_index_v2"].index
    zz = z.loc[a].values
    draws = dirichlet.rvs([2, 2, 2], size=2000, random_state=42)
    rm = np.empty((2000, len(a)), int)
    for d, w in enumerate(draws):
        sc = zz @ w; order = np.argsort(-sc); r = np.empty(len(a), int); r[order] = np.arange(1, len(a) + 1); rm[d] = r
    df["rank_v2"] = np.nan; df.loc[a, "rank_v2"] = df.loc[a, "recovery_index_v2"].rank(ascending=False)
    df["rank_mean"] = np.nan; df.loc[a, "rank_mean"] = rm.mean(0).round(2)
    df["rank_std"] = np.nan; df.loc[a, "rank_std"] = rm.std(0, ddof=1).round(2)
    df = df.sort_values("recovery_index_v2", ascending=False)
    v1 = pd.read_csv(root / "outputs/tables/community_recovery_index_24.csv")[["community", "recovery_index", "rank"]].rename(
        columns={"recovery_index": "recovery_index_v1", "rank": "rank_v1"})
    df = df.merge(v1, on="community", how="left")
    ass = df.dropna(subset=["rank_v2"])
    df["rank_v1_assessed"] = np.nan
    df.loc[ass.index, "rank_v1_assessed"] = ass.rank_v1.rank(method="first")
    both = df.dropna(subset=["rank_v2", "rank_v1"])
    rho = spearmanr(both.rank_v1, both.rank_v2).statistic
    log.info("assessed %d of %d communities; Spearman v1 vs v2 ranks %.2f", len(both), n, rho)
    # correlation between components
    cc = df[comps].corr(method="spearman").round(2)
    log.info("component Spearman correlations:\n%s", cc.to_string())
    log.info("\n%s", df[["community", "district", "flood_px", "persistence_ratio", "sar_recovery", "ndvi_recovery",
                        "mndwi_recession", "recovery_index_v2", "rank_v2", "rank_std", "rank_v1"]].round(3).to_string(index=False))
    df.round(4).to_csv(get_output_path("tables/community_recovery_index_v2.csv"), index=False)
    pd.DataFrame([{"n_assessed": len(both), "spearman_v1_v2": rho}]).to_csv(get_output_path("tables/recovery_v1_v2_comparison.csv"), index=False)


if __name__ == "__main__":
    main()
