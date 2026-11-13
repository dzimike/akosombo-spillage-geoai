"""
Figure 1 — Study area map, rebuilt in matplotlib (replaces QGIS-only workflow).

Adds a Ghana context inset (reviewer request) and fixes the legend/caption
colour mismatch present in the previous QGIS-exported version.

Usage: run inside the 'flood' conda environment (has rasterio/geopandas).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[2]
DPI = 600
PROJ_CRS = "EPSG:32630"

import sys
sys.path.insert(0, str(ROOT))
from src.visualization.palette import CLASS, DISTRICT
DISTRICT_COLORS = dict(DISTRICT)
DISTRICT_ORDER = ["North Tongu", "Shai-Osudoku", "Lower Manya Krobo",
                   "Ada East", "Central Tongu", "South Tongu"]

# Flood confidence class colours — MUST match the caption text exactly
CLASS_COLORS = {
    0: (CLASS["none"], "Not flooded"),
    1: (CLASS["sar"], "SAR only"),
    2: (CLASS["optical"], "Optical only"),
    3: (CLASS["both"], "SAR and optical agreement"),
    4: (CLASS["pre"], "Pre-existing water"),
}


def main():
    flood_path = ROOT / "data/processed/flood_extent/fused_flood_confidence.tif"
    comm_path  = ROOT / "data/processed/aoi/affected_communities.gpkg"
    aoi_path   = ROOT / "data/processed/aoi/lower_volta_aoi.gpkg"
    dam_path   = ROOT / "data/processed/aoi/dam_locations.gpkg"
    roads_path = ROOT / "data/raw/roads/osm_roads.gpkg"
    water_path = ROOT / "data/raw/roads/osm_waterways.gpkg"
    gadm0_path = ROOT / "data/raw/admin/gadm41_GHA.gpkg"
    out_path   = ROOT / "outputs/figures/fig1_study_area.png"

    comm  = gpd.read_file(comm_path).to_crs(PROJ_CRS)
    aoi   = gpd.read_file(aoi_path).to_crs(PROJ_CRS)
    dams  = gpd.read_file(dam_path).to_crs(PROJ_CRS)
    roads = gpd.read_file(roads_path).to_crs(PROJ_CRS)
    water = gpd.read_file(water_path).to_crs(PROJ_CRS)

    scale = 6
    with rasterio.open(flood_path) as src:
        H, W = src.height, src.width
        arr = src.read(1, out_shape=(H // scale, W // scale),
                       resampling=Resampling.nearest)
        left, bottom, right, top = src.bounds
        raster_crs = src.crs

    rgba = np.zeros((*arr.shape, 4), dtype=np.float32)
    for cls, (hexcol, _) in CLASS_COLORS.items():
        r, g, b = (int(hexcol[i:i+2], 16) / 255 for i in (1, 3, 5))
        mask = arr == cls
        rgba[mask] = [r, g, b, 1.0]
    nodata_mask = ~np.isin(arr, list(CLASS_COLORS.keys()))
    rgba[nodata_mask] = [0.15, 0.15, 0.15, 1.0]

    fig = plt.figure(figsize=(11, 12), dpi=DPI)
    ax = fig.add_axes([0.06, 0.06, 0.62, 0.84])

    ax.imshow(rgba, extent=[left, right, bottom, top], origin="upper",
              interpolation="nearest", zorder=1)

    roads_clip = gpd.clip(roads, aoi.geometry.union_all().buffer(2000))
    roads_clip.plot(ax=ax, color="#666666", linewidth=0.3, alpha=0.6, zorder=2)
    water_clip = gpd.clip(water, aoi.geometry.union_all().buffer(2000))
    water_clip.plot(ax=ax, color="#0D47A1", linewidth=0.6, alpha=0.8, zorder=3)

    aoi.boundary.plot(ax=ax, color="black", linewidth=1.1, linestyle="--", zorder=5)

    for _, row in comm.iterrows():
        x, y = row.geometry.x, row.geometry.y
        color = DISTRICT_COLORS.get(row.get("district", ""), "#888888")
        ax.plot(x, y, "o", color=color, ms=7, mec="white", mew=0.8, zorder=6)
        ax.annotate(row["community_name"], (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=6.3, color="black",
                    path_effects=[withStroke(linewidth=2, foreground="white")],
                    zorder=7)

    for _, row in dams.iterrows():
        x, y = row.geometry.x, row.geometry.y
        ax.plot(x, y, marker="*", color="#B8860B", markersize=18,
                mec="black", mew=0.6, zorder=8)
        ax.annotate(row["name"], (x, y), textcoords="offset points",
                    xytext=(8, 6), fontsize=7.5, fontweight="bold", color="#5C4400",
                    path_effects=[withStroke(linewidth=2.5, foreground="white")],
                    zorder=8)

    # View extent must cover the full AOI + all communities + both dams, not
    # just the flood raster's own bounding box (the raster was not extended
    # when the AOI was widened westward to include Lower Manya Krobo).
    aoi_b = aoi.total_bounds
    comm_b = comm.total_bounds
    dam_b = dams.total_bounds
    view_left   = min(left, aoi_b[0], comm_b[0], dam_b[0])
    view_bottom = min(bottom, aoi_b[1], comm_b[1], dam_b[1])
    view_right  = max(right, aoi_b[2], comm_b[2], dam_b[2])
    view_top    = max(top, aoi_b[3], comm_b[3], dam_b[3])
    pad_x = (view_right - view_left) * 0.03
    pad_y = (view_top - view_bottom) * 0.03
    view_left -= pad_x; view_right += pad_x
    view_bottom -= pad_y; view_top += pad_y

    ax.set_xlim(view_left, view_right)
    ax.set_ylim(view_bottom, view_top)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.set_xlabel("Easting (km, UTM 30N)", fontsize=9)
    ax.set_ylabel("Northing (km, UTM 30N)", fontsize=9)
    ax.tick_params(labelsize=7.5)

    # Scale bar
    bar_len_m = 20000
    bar_x0 = view_left + (view_right - view_left) * 0.04
    bar_y0 = view_bottom + (view_top - view_bottom) * 0.03
    ax.plot([bar_x0, bar_x0 + bar_len_m], [bar_y0, bar_y0], color="black", lw=3, zorder=9)
    ax.text(bar_x0 + bar_len_m / 2, bar_y0 + (view_top - view_bottom) * 0.012, "20 km",
            ha="center", fontsize=8, zorder=9)

    # North arrow
    na_x = view_right - (view_right - view_left) * 0.05
    na_y0 = view_bottom + (view_top - view_bottom) * 0.03
    na_y1 = na_y0 + (view_top - view_bottom) * 0.05
    ax.annotate("", xy=(na_x, na_y1), xytext=(na_x, na_y0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2), zorder=9)
    ax.text(na_x, na_y1 + (view_top - view_bottom) * 0.008, "N", ha="center", fontsize=9,
            fontweight="bold", zorder=9)

    # ---- Ghana context inset (reviewer request) ----------------------------
    inset_ax = fig.add_axes([0.71, 0.72, 0.24, 0.22])
    gadm0 = gpd.read_file(gadm0_path, layer="ADM_ADM_0").to_crs(PROJ_CRS)
    gadm0.plot(ax=inset_ax, facecolor="#EDEDED", edgecolor="black", linewidth=0.6)
    aoi.boundary.plot(ax=inset_ax, color="none")
    aoi_bounds = aoi.total_bounds
    box_w = (aoi_bounds[2] - aoi_bounds[0]) * 3
    box_h = (aoi_bounds[3] - aoi_bounds[1]) * 3
    cx = (aoi_bounds[0] + aoi_bounds[2]) / 2
    cy = (aoi_bounds[1] + aoi_bounds[3]) / 2
    inset_ax.add_patch(mpatches.Rectangle(
        (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
        facecolor="none", edgecolor="#C62828", linewidth=1.5, zorder=5))
    inset_ax.set_xticks([]); inset_ax.set_yticks([])
    inset_ax.set_title("Ghana", fontsize=8, fontweight="bold")
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("#888888")

    # ---- Legend --------------------------------------------------------
    legend_ax = fig.add_axes([0.71, 0.06, 0.27, 0.62])
    legend_ax.axis("off")

    class_handles = [mpatches.Patch(facecolor=c, edgecolor="none", label=lbl)
                      for _, (c, lbl) in CLASS_COLORS.items()]
    dist_handles = [Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=DISTRICT_COLORS[d],
                            markeredgecolor="white", markersize=8, label=d)
                     for d in DISTRICT_ORDER]
    misc_handles = [
        Line2D([0], [0], color="black", lw=1.1, linestyle="--", label="Study area AOI boundary"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#B8860B",
               markeredgecolor="black", markersize=12, label="Akosombo / Kpong dams"),
        Line2D([0], [0], color="#0D47A1", lw=1.2, label="Volta River / waterways"),
        Line2D([0], [0], color="#666666", lw=0.8, label="Roads (OSM)"),
    ]

    y = 0.98
    legend_ax.text(0.0, y, "Flood confidence (5-class)", fontsize=8.5, fontweight="bold",
                    transform=legend_ax.transAxes)
    leg1 = legend_ax.legend(handles=class_handles, loc="upper left",
                            bbox_to_anchor=(0.0, 0.96), fontsize=7.3,
                            frameon=False, handlelength=1.4)
    legend_ax.add_artist(leg1)

    leg2 = legend_ax.legend(handles=dist_handles, loc="upper left",
                            bbox_to_anchor=(0.0, 0.62), fontsize=7.3,
                            title="District", title_fontsize=8, frameon=False,
                            handlelength=1.4)
    legend_ax.add_artist(leg2)

    legend_ax.legend(handles=misc_handles, loc="upper left",
                     bbox_to_anchor=(0.0, 0.20), fontsize=7.0,
                     title="Reference features", title_fontsize=8, frameon=False,
                     handlelength=1.6)

    fig.text(0.06, 0.035,
              "CRS: WGS 84 / UTM Zone 30N (EPSG:32630)  |  Flood: Sentinel-1/2, Sep-Oct 2023  |  "
              "20-km bilateral AOI  |  All 24 communities designated by NADMO/VRA",
              fontsize=6.5, color="#444444")

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
