from __future__ import annotations
"""
Graphical method overview for the main paper: research gap, three numbered steps, every step shown with image chips
from the same 24 km x 18 km window of the Lower Volta floodplain (Mepe-Battor to Adidome).

Chips: S1 pre-event and 23 Oct VV, change, S2 NDWI | change-date count, SAR change mask, optical water, fused classes |
exposure, agreement with UNOSAT, K x M sensitivity grid, Mepe building check.
"""
import logging
import textwrap

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from src.flood_mapping.change_counts import count_flags
from src.flood_mapping.water_masks import static_water
from src.utils.paths import get_project_root
from src.validation.compare_unosat import UN
from src.validation.threshold_exposure_direct import ROAD_SAMPLE_SPACING_M, interpolate_road_points
from src.visualization.palette import AGREE, CLASS, DEFINITION

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
BOX = (875000, 662000, 899000, 680000)          # xmin, ymin, xmax, ymax (UTM 30N)
DS = 3                                           # display down-sampling


def main() -> None:
    root = get_project_root()
    sat, fl = root / "data/processed/satellite", root / "data/processed/flood_extent"
    with rasterio.open(fl / "fused_flood_confidence.tif") as s:
        tr, crs = s.transform, s.crs
        win = from_bounds(*BOX, transform=tr).round_offsets().round_lengths()
        fu = s.read(1, window=win)
    wtr = rasterio.windows.transform(win, tr)
    H, W = fu.shape
    rd = lambda p: rasterio.open(p).read(1, window=win)
    ext = [BOX[0] / 1e3, BOX[2] / 1e3, BOX[1] / 1e3, BOX[3] / 1e3]
    pre_vv, sc_vv = rd(sat / "s1_pre_event_vv.tif"), rd(sat / "s1_scene_20231023_vv.tif")
    ndwi = rd(sat / "s2_ndwi_event.tif")
    cnt, _, dates = count_flags(sat)
    cnt = cnt[win.row_off:win.row_off + H, win.col_off:win.col_off + W]
    sar = rd(fl / "sar_flood_mask_binary.tif") == 1
    opt = rd(fl / "optical_water_mask.tif") == 1
    ow = rd(fl / "sar_openwater_flood_mask.tif") == 1
    static = static_water(sat / "s1_pre_event_vv.tif", sat, fl)["union"][win.row_off:win.row_off + H, win.col_off:win.col_off + W]
    pre = fu == 4
    old = ow | (opt & ~static)
    new = np.isin(fu, [1, 2, 3])
    ds = lambda a: a[::DS, ::DS]
    def ras(name):
        g = gpd.read_file(root / UN.format(name)).to_crs(crs)
        return rasterize(((geom, 1) for geom in g.geometry), out_shape=(H, W), transform=wtr, fill=0, dtype="uint8").astype(bool)
    uf = ras("FloodExtent")
    # exposure points
    bl = gpd.read_file(root / "data/raw/buildings/osm_buildings.gpkg").to_crs(crs)
    cen = bl.geometry.centroid
    r, c = rasterio.transform.rowcol(wtr, cen.x.values, cen.y.values); r, c = np.array(r), np.array(c)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    expo = np.zeros(len(bl), bool); expo[ok] = new[r[ok], c[ok]]
    eb = cen[expo]
    roads = gpd.read_file(root / "data/raw/roads/osm_roads.gpkg").to_crs(crs)
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].explode(index_parts=False).reset_index(drop=True)
    rx, ry, _, rl = interpolate_road_points(roads, ROAD_SAMPLE_SPACING_M)
    rr, rc = rasterio.transform.rowcol(wtr, rx, ry); rr, rc = np.array(rr), np.array(rc)
    okr = (rr >= 0) & (rr < H) & (rc >= 0) & (rc < W)
    fr = np.zeros(len(rx), bool); fr[okr] = new[rr[okr], rc[okr]]
    # Mepe frame
    mb = gpd.read_file(root / "data/processed/validation/charter_mepe_flooded_buildings.gpkg").to_crs(crs)
    mat = pd.read_csv(root / "outputs/tables/definition_reference_matrix.csv")
    grid_area = {}
    for _, row in mat.iterrows():
        n = row.definition
        if n.startswith("Release period, K="):
            k = int(n.split("K=")[1].split(" dB")[0]); m = int(n.split(">= ")[1].split(" dates")[0]); grid_area[(k, m)] = row.area_km2
    f1 = {k: mat.loc[mat.definition.str.startswith(k), "UNOSAT 23 Oct F1"].iloc[0] for k in ("Open-water rule (event", "Release period, K=3 dB, >= 2")}

    cls_cmap = ListedColormap(["#EEEEEE", CLASS["sar"], CLASS["optical"], CLASS["both"], CLASS["pre"]])
    def show(ax, arr, cmap="gray", vmin=None, vmax=None, title=None):
        ax.imshow(ds(arr), extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_edgecolor("#888"); s_.set_linewidth(0.4)
        if title:
            ax.set_title(textwrap.fill(title, 33), fontsize=6.2, pad=1.5, linespacing=1.05)
    def classmap(arr_cls):
        return arr_cls
    fig = plt.figure(figsize=(6.85, 7.7))
    gs = GridSpec(4, 4, height_ratios=[0.74, 1, 1, 1], hspace=0.42, wspace=0.06, left=0.02, right=0.985, top=0.968, bottom=0.02, figure=fig)
    # ---- banner
    axb = fig.add_subplot(gs[0, :]); axb.axis("off")
    sub = gs[0, :].subgridspec(1, 4, width_ratios=[0.25, 1.05, 1.05, 1.35], wspace=0.12)
    axt = fig.add_subplot(sub[0, 0]); axt.axis("off")
    for j, (a, ttl, col, val) in enumerate(((old, "Open-water definition", DEFINITION["openwater"], f1["Open-water rule (event"]),
                                            (new, "Change-based definition", DEFINITION["change"], f1["Release period, K=3 dB, >= 2"]))):
        ax = fig.add_subplot(sub[0, 1 + j])
        img = np.where(ds(a), 1, 0)
        ax.imshow(np.where(ds(pre), 2, img), extent=ext, cmap=ListedColormap(["#EEEEEE", col, CLASS["pre"]]), vmin=0, vmax=2, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=6.6, color=col, fontweight="bold", pad=2)
        ax.text(0.5, -0.07, f"F1 against UNOSAT: {val:.2f}", transform=ax.transAxes, ha="center", fontsize=6.2)
    axs = fig.add_subplot(sub[0, 3]); axs.axis("off")
    axs.text(0, 0.92, "Whole study area", fontsize=6.8, fontweight="bold", va="top")
    axs.text(0, 0.78, "Open-water definition   72.2 km²\nChange-based extent      44.2 km²\nShared by both            18.2 km²\n\n"
             "Exposure to the change extent:\n59 buildings, 5,400 people, 12 km of road", fontsize=6.0, va="top", linespacing=1.5)
    # ---- band helper
    def band(row, title, subtitle=""):
        fig.canvas.draw()
        a0 = fig.axes[-4 if False else 0]
    # Step 1
    ax = [fig.add_subplot(gs[1, j]) for j in range(4)]
    show(ax[0], pre_vv, "gray", -22, -2, "Sentinel-1 VV, pre-event median (orbit-matched)")
    show(ax[1], sc_vv, "gray", -22, -2, "Sentinel-1 VV, 23 Oct 2023")
    show(ax[2], sc_vv - pre_vv, "RdBu", -6, 6, "Change: 23 Oct minus pre-event (dB)")
    show(ax[3], ndwi, "BrBG", -0.6, 0.6, "Sentinel-2 NDWI, event composite")
    ax[0].plot([ext[0] + 1, ext[0] + 6], [ext[2] + 1, ext[2] + 1], color="white", lw=3); ax[0].text(ext[0] + 3.5, ext[2] + 1.5, "5 km", color="white", fontsize=5.3, ha="center")
    # Step 2
    bx = [fig.add_subplot(gs[2, j]) for j in range(4)]
    show(bx[0], np.where(pre, np.nan, cnt), "YlOrRd", 0, 7, "SAR: release-period dates with a drop > 3 dB (0-7)")
    show(bx[1], np.where(pre, 2, sar.astype(int)), ListedColormap(["#EEEEEE", CLASS["sar"], CLASS["pre"]]), 0, 2, "SAR change: ≥ 2 dates (K = 3 dB)")
    show(bx[2], np.where(pre, 2, (opt & ~pre).astype(int)), ListedColormap(["#EEEEEE", CLASS["optical"], CLASS["pre"]]), 0, 2, "Optical water: NDWI and MNDWI ≥ 0")
    fcls = np.zeros_like(fu); fcls[fu == 1] = 1; fcls[fu == 2] = 2; fcls[fu == 3] = 3; fcls[fu == 4] = 4
    show(bx[3], fcls, cls_cmap, 0, 4, "Fused classes outside pre-existing water")
    bx[3].legend(handles=[mp.Patch(color=CLASS["sar"], label="SAR only"), mp.Patch(color=CLASS["optical"], label="Optical only"),
                          mp.Patch(color=CLASS["both"], label="Both"), mp.Patch(color=CLASS["pre"], label="Pre-existing water")],
                 loc="lower left", fontsize=4.9, frameon=True, framealpha=0.9)
    # Step 3
    cx = [fig.add_subplot(gs[3, j]) for j in range(4)]
    show(cx[0], np.where(new, 1, np.where(pre, 2, 0)), ListedColormap(["#F4F4F4", "#9EC9E8", CLASS["pre"]]), 0, 2, "Exposure: buildings and road samples in the extent")
    rxk, ryk = rx[fr] / 1e3, ry[fr] / 1e3
    cx[0].scatter(rxk, ryk, s=0.5, color=ASSET_ROAD if (ASSET_ROAD := "#555555") else "k", zorder=3)
    cx[0].scatter(eb.x / 1e3, eb.y / 1e3, s=6, color="#CC3333", edgecolor="white", linewidth=0.2, zorder=4)
    cx[0].set_xlim(ext[0], ext[1]); cx[0].set_ylim(ext[2], ext[3])
    both = new & uf; oo = new & ~uf; uo = uf & ~new
    ag = np.zeros((H, W), int); ag[pre] = 4; ag[oo] = 1; ag[uo] = 2; ag[both] = 3
    show(cx[1], ag, ListedColormap([AGREE["none"], AGREE["ours"], AGREE["unosat"], AGREE["both"], AGREE["pre"]]), 0, 4, "Agreement with UNOSAT, 23 Oct")
    cx[1].legend(handles=[mp.Patch(color=AGREE["both"], label="Both"), mp.Patch(color=AGREE["ours"], label="Ours only"), mp.Patch(color=AGREE["unosat"], label="UNOSAT only")],
                 loc="lower left", fontsize=4.9, framealpha=0.9)
    # sensitivity grid
    ax = cx[2]
    G = np.array([[grid_area[(k, m)] for m in (1, 2, 3)] for k in (2, 3, 4)])
    ax.imshow(G, cmap="Blues", vmin=0, vmax=220, aspect=0.72)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{G[i, j]:.0f}", ha="center", va="center", fontsize=6.2, fontweight="bold" if (i, j) == (1, 1) else "normal",
                    color="white" if G[i, j] > 110 else "#222")
    ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, ec="black", lw=1.1))
    ax.set_xticks(range(3)); ax.set_xticklabels(["≥1", "≥2", "≥3"], fontsize=5.8); ax.set_yticks(range(3)); ax.set_yticklabels(["2 dB", "3 dB", "4 dB"], fontsize=5.8)
    ax.set_xlabel("Dates flagged (of 7)", fontsize=5.8)
    ax.set_title("Flooded area (km²):\nthreshold K x dates", fontsize=6.2, pad=1.5, linespacing=1.05)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    pos = cx[2].get_position(); cx[2].set_position([pos.x0 + 0.03, pos.y0 + 0.006, pos.width - 0.03, pos.height])
    # Mepe check
    ax = cx[3]
    x0, x1, y0, y1 = 878000, 881500, 671300, 674000
    sel = mb.cx[x0:x1, y0:y1]
    cols = np.where(sel.charter_affected, "#F0A030", "#CCCCCC")
    sel.plot(ax=ax, color=cols, edgecolor=cols, linewidth=0.3)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")
    ax.set_title(textwrap.fill("Mepe: 179 buildings flagged flooded (WorldView-2)", 33), fontsize=6.2, pad=1.5, linespacing=1.05)
    ax.text(0.5, 0.04, "orange: flagged; none in our extent", transform=ax.transAxes, ha="center", fontsize=5.0, bbox=dict(fc="white", ec="none", alpha=0.85))
    for s_ in ax.spines.values():
        s_.set_edgecolor("#888"); s_.set_linewidth(0.4)
    # ---- step frames and captions (figure coordinates)
    fig.canvas.draw()
    def frame(axs, label, desc, off=0.2):
        x0 = min(a.get_position().x0 for a in axs) - 0.012; x1 = max(a.get_position().x1 for a in axs) + 0.012
        y0 = min(a.get_position().y0 for a in axs) - 0.042; y1 = max(a.get_position().y1 for a in axs) + 0.046
        fig.add_artist(mp.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.0,rounding_size=0.008", fc="none", ec="#555", lw=0.7, ls=(0, (4, 3)),
                                         transform=fig.transFigure, clip_on=False))
        fig.text(x0 + 0.008, y1 + 0.004, label, fontsize=7.2, fontweight="bold", va="bottom", bbox=dict(fc="white", ec="#555", boxstyle="round,pad=0.25", lw=0.6))
        fig.text(x0 + off, y1 + 0.006, desc, fontsize=8.5, color="#444", va="bottom")
        return x0, x1, y0, y1
    f1_ = frame(ax_s1 := [fig.axes[i] for i in range(0)], "", "") if False else None
    boxes = []
    boxes.append(frame([a for a in fig.axes if a.get_subplotspec().rowspan.start == 1], "Step 1  Inputs", ""))
    boxes.append(frame([a for a in fig.axes if a.get_subplotspec().rowspan.start == 2], "Step 2  Detection and fusion", ""))
    boxes.append(frame([a for a in fig.axes if a.get_subplotspec().rowspan.start == 3], "Step 3  Exposure and checks", ""))
    for (xa, xb, ya, yb), (xc, xd, yc, yd) in zip(boxes[:-1], boxes[1:]):
        fig.add_artist(mp.FancyArrowPatch(((xa + xb) / 2, ya - 0.001), ((xc + xd) / 2, yd + 0.012), arrowstyle="-|>", mutation_scale=9, color="#444", lw=0.8,
                                          transform=fig.transFigure))
    out = root / "outputs/figures/fig_method_overview.png"
    fig.savefig(out, dpi=600)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
