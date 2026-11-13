from __future__ import annotations
"""
Figures for the narrowed main manuscript.

fig2_methodological_workflow.png  mapping -> exposure -> sensitivity workflow
fig4_exposure_hotspots.png        exposure by (non-overlapping) administrative district
fig5_s1_water_timeseries.png      per-scene Sentinel-1 open-water area in three reach windows
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.utils.paths import get_project_root
from src.visualization.palette import AGREE, ASSET, CALIBRATION, CLASS, DEFINITION

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
DPI = 600
DESIGNATED = {"Ada East", "North Tongu", "Central Tongu", "South Tongu", "Shai Osudoku"}


def workflow(out):
    fig, ax = plt.subplots(figsize=(11, 5.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(-0.5, 5.6)
    def box(x, y, w, h, text, fc):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec="#555", lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#555", lw=1.1))
    box(0.1, 4.2, 2.6, 1.1, "Sentinel-1 GRD (VV/VH, dB)\nascending, 2 orbits\npre 6 / event 9 / post 13 scenes\n7×7 focal mean, median", "#CFE3F3")
    box(0.1, 2.7, 2.6, 1.1, "Sentinel-2 L2A\ncloud-masked composites\nNDWI, MNDWI (pre / event / post)", "#FCEBC7")
    box(0.1, 0.6, 2.6, 1.6, "Pre-existing water\nJRC occurrence ≥ 50 %\n+ open-sea mask (GLO-30)\n+ pre-event optical water\n+ pre-event SAR open water\n(slope ≤ 5° mask from GLO-30)", "#E4F1FA")
    box(3.4, 4.2, 2.5, 1.1, "SAR change\ndVV and dVH < −3 dB vs\norbit-matched pre-event\nmedian, on ≥ 2 dates", "#CFE3F3")
    box(3.4, 2.7, 2.5, 1.1, "Optical water\nNDWI ≥ 0 and MNDWI ≥ 0", "#FCEBC7")
    box(6.6, 3.3, 2.0, 1.4, "Fusion, 5 classes\nSAR only / optical only /\nboth / pre-existing water /\nnot flooded", "#CBEBE1")
    box(9.0, 3.3, 1.9, 1.4, "Exposure\nbuildings, roads,\npopulation, cropland,\nfacilities; by district", "#F3DDE9")
    box(3.4, 0.6, 5.2, 1.6, "Sensitivity and independent comparison\nchange threshold 2–4 dB x 1–3 dates; previous open-water rule\nUNOSAT (23 Oct), Charter 844 TanDEM-X (20 Oct) and\nWorldView-2 flooded buildings (Mepe, 18 Oct)", "#EDEDED")
    arr(2.7, 4.75, 3.4, 4.75); arr(2.7, 3.25, 3.4, 3.25); arr(2.7, 2.2, 3.4, 2.85)
    arr(5.9, 4.75, 6.6, 4.2); arr(5.9, 3.25, 6.6, 3.8); arr(8.6, 4.0, 9.0, 4.0)
    arr(7.6, 3.3, 7.0, 2.2); ax.text(1.4, 0.3, "removed from both sensor masks\nand set as class 4", ha="center", fontsize=7, color="#555")
    ax.text(5.5, -0.3, "All layers on the 10 m UTM 30N grid (EPSG:32630); reference products are analyst-derived, not field-validated", ha="center", fontsize=7.5, color="#555")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig); log.info("saved %s", out)


def exposure(root, out):
    d = pd.read_csv(root / "outputs/tables/district_exposure.csv")
    d = d[d.flood_km2 >= 1.0].sort_values("flood_km2")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    spec = [("flood_km2", "Flood area (km²)", "{:.1f}"), ("buildings_exposed", "Exposed buildings", "{:,.0f}"),
            ("population_exposed", "Exposed population", "{:,.0f}"), ("roads_flooded_km", "Flooded road (km)", "{:.1f}")]
    for ax, (c, lab, fmt), akey in zip(axes, spec, ("area", "buildings", "people", "roads")):
        bars = ax.barh(d.district, d[c], color=ASSET[akey], height=0.68)
        for b_, dn in zip(bars, d.district):
            b_.set_alpha(1.0 if dn in DESIGNATED else 0.4)
        for y, v in enumerate(d[c]):
            ax.text(v + d[c].max() * 0.02, y, fmt.format(v), va="center", fontsize=8)
        ax.set_xlim(0, d[c].max() * 1.25); ax.set_xlabel(lab, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", ls=":", alpha=0.4); ax.tick_params(labelsize=8)
    for ax, t in zip(axes, "abcd"):
        ax.set_title(t, loc="left", fontweight="bold", fontsize=11)
    fig.legend(handles=[mpatches.Patch(color="#555555", label="District containing designated communities"),
                        mpatches.Patch(color="#555555", alpha=0.4, label="Other district")], loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig); log.info("saved %s", out)


def timeseries(root, out):
    t = pd.read_csv(root / "outputs/tables/s1_water_timeseries.csv")
    t = t[t.relative_orbit == 74].copy(); t["date"] = pd.to_datetime(t["date"])
    names = {"A_Mepe_Battor": "Mepe–Battor reach", "B_Sogakope_Sokpoe": "Sogakope–Sokpoe reach", "C_Ada_Songor": "Ada–Songor wetland"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (k, lab) in zip(axes, names.items()):
        g = t[t.window == k].sort_values("date")
        ax.plot(g.date, g.water_km2, "o-", color=CLASS["sar"], ms=3.5, lw=1.4)
        ax.axvspan(pd.Timestamp("2023-09-15"), pd.Timestamp("2023-10-30"), color="#DDDDDD", alpha=0.6, lw=0)
        ax.axvline(pd.Timestamp("2023-10-09"), color="#555555", ls="--", lw=1)
        ax.set_title(lab, fontsize=10); ax.set_ylabel("Open water outside permanent water (km²)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(ls=":", alpha=0.4); ax.tick_params(labelsize=8)
        ax.set_ylim(0, g.water_km2.max() * 1.15)
        ax.set_xticks(pd.to_datetime(["2023-07-01", "2023-09-01", "2023-11-01", "2024-01-01"]))
        ax.set_xticklabels(["Jul", "Sep", "Nov", "Jan"], fontsize=8)
    axes[0].text(pd.Timestamp("2023-09-17"), axes[0].get_ylim()[1] * 0.93, "release\n15 Sep–30 Oct", fontsize=7, color="#555555", va="top")
    fig.tight_layout(); fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig); log.info("saved %s", out)


def uncertainty_decomposition(root, out):
    """Range of flood area and exposure under three analyst choices, log axis."""
    dd = pd.read_csv(root / "outputs/tables/threshold_exposure_direct_prior_corrected.csv").set_index("threshold")
    dd["population_exposed"] = dd["population_exposed"] * (366077 / 426499)
    ld = pd.read_csv(root / "outputs/tables/label_definition_exposure.csv").set_index("definition")
    cols = [("Buildings", "buildings_exposed", "buildings"), ("Population", "population_exposed", "population"),
            ("Roads (km)", "roads_exposed_km", "roads_km"), ("Flood area (km²)", "flood_area_km2", "flood_km2")]
    hp = dd.loc[0.50:0.95]
    thr = [100 * (hp[c].max() - hp[c].min()) / dd.loc[0.50, c] for _, c, _ in cols]
    chg = []
    for _, _, k in cols:
        v = [ld.loc[x, k] for x in ("primary", "k2_m2", "k4_m2", "k3_m1", "k3_m3", "k4_m1")]
        chg.append(100 * (max(v) - min(v)) / ld.loc["primary", k])
    new = [abs(100 * (ld.loc["open_water_rule", k] / ld.loc["primary", k] - 1)) for _, _, k in cols]
    x = np.arange(len(cols)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    series = [(thr, "Probability threshold 0.50–0.95 (prior-corrected)", CALIBRATION["corrected"], None), (chg, "Change rule: 2–4 dB x 1–3 dates (range)", DEFINITION["change"], "//"),
              (new, "Definition: open-water rule instead of change", DEFINITION["openwater"], "xx")]
    for i, (vals, lab, colr, h) in enumerate(series):
        bars = ax.bar(x + (i - 1) * w, vals, w, color=colr, alpha=0.9 if h is None else 0.55, hatch=h, edgecolor="white", label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_yscale("log"); ax.set_ylim(1, 600)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cols], fontsize=9)
    ax.set_ylabel("Change or range (% of primary value)", fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.4, which="both"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout(); fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig); log.info("saved %s", out)


def unosat_agreement(root, out):
    """Where our fused extent and the UNOSAT FloodExtent (23 Oct 2023) agree and disagree: overview map and two 10 m close-ups."""
    import geopandas as gpd, rasterio
    from matplotlib.colors import ListedColormap
    from matplotlib.gridspec import GridSpec
    from rasterio.features import rasterize
    from src.validation.compare_unosat import UN
    with rasterio.open(root / "data/processed/flood_extent/fused_flood_confidence.tif") as s:
        fused, tr, crs, shape = s.read(1), s.transform, s.crs, s.shape
    def ras(name):
        g = gpd.read_file(root / UN.format(name)).to_crs(crs)
        return rasterize(((geom, 1) for geom in g.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    dom = ras("AnalysisExtent"); uf = ras("FloodExtent") & dom
    ours = np.isin(fused, [1, 2, 3]) & dom; perm = (fused == 4) & dom
    cat = np.zeros(shape, np.uint8)             # 0 outside, 1 analysed dry land, 2 pre-existing water, 3 ours only, 4 UNOSAT only, 5 both
    cat[dom] = 1; cat[dom & perm] = 2
    cat[ours & ~uf] = 3; cat[uf & ~ours] = 4; cat[ours & uf] = 5
    cm = ListedColormap(["white", AGREE["none"], AGREE["pre"], AGREE["ours"], AGREE["unosat"], AGREE["both"]])
    x0, y1 = tr.c, tr.f
    def bounds(km):            # km box (E0, N0, E1, N1) -> pixel slices
        c0, c1 = int((km[0] * 1e3 - x0) / 10), int((km[2] * 1e3 - x0) / 10)
        r0, r1 = int((y1 - km[3] * 1e3) / 10), int((y1 - km[1] * 1e3) / 10)
        return slice(r0, r1), slice(c0, c1)
    boxes = {"a": (875, 662, 899, 680), "b": (897, 640, 922, 658)}
    titles = {"a": "River reach, Mepe to Adidome", "b": "Eastern flats and delta"}
    km2 = lambda m: m.sum() / 1e4
    fig = plt.figure(figsize=(13, 7.4))
    gs = GridSpec(2, 2, width_ratios=[1.55, 1], hspace=0.22, wspace=0.06, left=0.05, right=0.985, top=0.96, bottom=0.12, figure=fig)
    ax = fig.add_subplot(gs[:, 0])
    k = 4; H, W = (shape[0] // k) * k, (shape[1] // k) * k
    red = lambda m: m[:H, :W].reshape(H // k, k, W // k, k)
    # majority-priority down-sampling: keep the rarest present class so small patches stay visible
    blk = cat[:H, :W].reshape(H // k, k, W // k, k)
    pr = np.zeros((H // k, W // k), np.uint8)
    for v in (1, 2, 5, 3, 4):                      # later classes overwrite: UNOSAT-only and ours-only win over background
        pr[(blk == v).any(axis=(1, 3))] = v
    rows = np.where((pr > 0).any(axis=1))[0]; cols = np.where((pr > 0).any(axis=0))[0]
    ext = [x0, x0 + W * 10, y1 - H * 10, y1]
    ax.imshow(pr, cmap=cm, vmin=0, vmax=5, extent=ext, origin="upper", interpolation="nearest")
    ax.set_xlim(x0 + cols.min() * k * 10, x0 + (cols.max() + 1) * k * 10); ax.set_ylim(y1 - (rows.max() + 1) * k * 10, y1 - rows.min() * k * 10)
    ax.set_xlabel("Easting (km, UTM 30N)"); ax.set_ylabel("Northing (km, UTM 30N)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}")); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.set_title("Overview of the common domain", fontsize=10, loc="left")
    xl = ax.get_xlim(); yl = ax.get_ylim()
    ax.plot([xl[0] + 3000, xl[0] + 13000], [yl[0] + 2500] * 2, color="black", lw=3); ax.text(xl[0] + 8000, yl[0] + 3200, "10 km", ha="center", fontsize=8)
    for key, bx in boxes.items():
        ax.add_patch(plt.Rectangle((bx[0] * 1e3, bx[1] * 1e3), (bx[2] - bx[0]) * 1e3, (bx[3] - bx[1]) * 1e3, fill=False, ec="black", lw=1.2))
        ax.text(bx[0] * 1e3 + 500, bx[3] * 1e3 - 500, key, fontsize=12, fontweight="bold", va="top")
    for i, (key, bx) in enumerate(boxes.items()):
        a2 = fig.add_subplot(gs[i, 1]); sl = bounds(bx)
        sub = cat[sl]
        a2.imshow(sub, cmap=cm, vmin=0, vmax=5, extent=[bx[0], bx[2], bx[1], bx[3]], origin="upper", interpolation="nearest")
        a2.set_xticks([]); a2.set_yticks([])
        for s_ in a2.spines.values():
            s_.set_edgecolor("black"); s_.set_linewidth(1.0)
        both_, oo, uo = km2(sub == 5), km2(sub == 3), km2(sub == 4)
        a2.set_title(f"{key}  {titles[key]}", fontsize=10, loc="left")
        a2.text(0.5, -0.04, f"both {both_:.1f}, ours only {oo:.1f}, UNOSAT only {uo:.1f} km²", transform=a2.transAxes, ha="center", va="top", fontsize=8.5)
        a2.plot([bx[0] + 1, bx[0] + 6], [bx[1] + 1] * 2, color="black", lw=2.5); a2.text(bx[0] + 3.5, bx[1] + 1.4, "5 km", ha="center", fontsize=7.5)
    km = lambda m: m.sum() / 1e4
    fig.legend(handles=[mpatches.Patch(color=AGREE["both"], label=f"Both ({km(ours & uf):.1f} km²)"), mpatches.Patch(color=AGREE["ours"], label=f"Our extent only ({km(ours & ~uf):.1f} km²)"),
                        mpatches.Patch(color=AGREE["unosat"], label=f"UNOSAT only ({km(uf & ~ours):.1f} km²)"), mpatches.Patch(color=AGREE["pre"], label="Our pre-existing water"),
                        mpatches.Patch(color=AGREE["none"], label="Analysed, flooded in neither")], loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig); log.info("saved %s", out)


def main():
    root = get_project_root(); fd = root / "outputs/figures"
    workflow(fd / "fig2_methodological_workflow.png")
    exposure(root, fd / "fig4_exposure_hotspots.png")
    timeseries(root, fd / "fig5_s1_water_timeseries.png")
    uncertainty_decomposition(root, fd / "uncertainty_paper/fig5_uncertainty_decomposition.png")
    unosat_agreement(root, fd / "fig6_unosat_agreement.png")


if __name__ == "__main__":
    main()
