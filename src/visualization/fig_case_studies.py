from __future__ import annotations
"""
Case-study figure: four places, four chips each.

A  Mepe: what 10 m data cannot see (WorldView-2 flagged buildings vs Sentinel-2, Sentinel-1 change and our extent)
B  Ningo-Prampram: the seasonal scatter of the event-window definition (extent outlines and per-date flags)
C  Songor lagoon margin: remaining unconfirmed patches (Sentinel-2 true colour, UNOSAT outline)
D  River bars in the TerraSAR-X sector: agreement (Sentinel-2, change, agreement map)

Imagery: Sentinel-1 and Sentinel-2 (Copernicus); WorldView-2 view in A1 is cropped from the International Charter map
(activation 844, NASRDA, 21 Oct 2023).
"""
import logging
import textwrap

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from src.utils.paths import get_project_root
from src.validation.compare_unosat import UN
from src.visualization.palette import AGREE, CLASS, DEFINITION

Image.MAX_IMAGE_PIXELS = None
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
CHANGE_C, OLD_C, REF_C, FLAG_C = DEFINITION["change"], "#D55E00", "#7E1E9C", "#F0A030"


class Win:
    def __init__(self, ref, box):
        self.box = [b * 1000 for b in box]
        w = from_bounds(*self.box, transform=ref.transform).round_offsets().round_lengths()
        self.win, self.tr = w, rasterio.windows.transform(w, ref.transform)
        self.h, self.w = int(w.height), int(w.width)
        self.ext = [box[0], box[2], box[1], box[3]]

    def read(self, path, band=1):
        with rasterio.open(path) as s:
            return s.read(band, window=self.win, boundless=True, fill_value=np.nan if s.dtypes[0].startswith("float") else 0)

    def xy(self):
        xs = (self.tr.c + (np.arange(self.w) + 0.5) * self.tr.a) / 1000
        ys = (self.tr.f + (np.arange(self.h) + 0.5) * self.tr.e) / 1000
        return xs, ys


def outline(ax, wn, mask, color, lw=0.5):
    xs, ys = wn.xy()
    if mask.any():
        ax.contour(xs, ys, mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def base(ax, title, wn):
    ax.set_xlim(wn.ext[0], wn.ext[1]); ax.set_ylim(wn.ext[2], wn.ext[3]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(textwrap.fill(title, 34), fontsize=6.3, pad=1.5, linespacing=1.05)
    for s in ax.spines.values():
        s.set_edgecolor("#888"); s.set_linewidth(0.4)


def rgb(a):
    x = np.stack(a, -1) / 2600.0
    return np.clip(x, 0, 1) ** 0.75


def main() -> None:
    root = get_project_root()
    sat, fl, val = root / "data/processed/satellite", root / "data/processed/flood_extent", root / "data/processed/validation"
    ref = rasterio.open(fl / "fused_flood_confidence.tif")
    crs = ref.crs
    new_p, old_p = fl / "fused_flood_confidence.tif", root / "archive/v6_event_window_persistence/flood_extent/fused_flood_confidence.tif"
    bands = pd.read_csv(sat / "s1_change_by_date_bands.csv")
    Hs = ref.shape
    def band_idx(orbit, date):
        return int(bands[(bands.orbit == orbit) & (bands.date == date)].index[0]) + 1
    mb = gpd.read_file(val / "charter_mepe_flooded_buildings.gpkg").to_crs(crs)
    fig = plt.figure(figsize=(6.85, 6.25))
    gs = GridSpec(4, 5, width_ratios=[0.09, 1, 1, 1, 1], height_ratios=[0.73, 0.70, 0.70, 0.79], hspace=0.30, wspace=0.04, left=0.01, right=0.995, top=0.945, bottom=0.01, figure=fig)
    def label(row, letter, title):
        ax = fig.add_subplot(gs[row, 0]); ax.axis("off")
        ax.text(0.5, 0.98, letter.lower(), fontsize=11, fontweight="bold", va="top", ha="center")
    # ------------------------------------------------ A  Mepe
    # window = the Mepe map frame (lon/lat of its corners, shrunk 100 m) in UTM 30N
    tf_ = Transformer.from_crs(4326, crs, always_xy=True)
    fx0, fy0 = tf_.transform(0.4145, 6.0642); fx1, fy1 = tf_.transform(0.4444, 6.0862)
    wn = Win(ref, (round(fx0 / 1000 + 0.1, 2), round(fy0 / 1000 + 0.1, 2), round(fx1 / 1000 - 0.1, 2), round(fy1 / 1000 - 0.1, 2)))
    ax = [fig.add_subplot(gs[0, j]) for j in range(1, 5)]
    img = np.array(Image.open(root / "data/raw/validation/charter844/img_28582238.jpg").convert("RGB"))
    t = Transformer.from_crs(crs, 4326, always_xy=True)
    (lo0, la0), (lo1, la1) = t.transform(wn.box[0], wn.box[3]), t.transform(wn.box[2], wn.box[1])
    px = lambda lo, la: (581.5 + (lo - 25 / 60) * 2579 * 60, 771.5 + (6 + 5 / 60 - la) * 2576 * 60)
    (x0, y0), (x1, y1) = px(lo0, la0), px(lo1, la1)
    ax[0].imshow(img[int(y0):int(y1), int(x0):int(x1)], extent=wn.ext)
    base(ax[0], "WorldView-2 (Charter map); flagged: orange", wn)
    flag = mb[mb.charter_affected]
    flag = flag.set_geometry(flag.geometry.scale(1 / 1000, 1 / 1000, origin=(0, 0)))
    rgbE = rgb([wn.read(sat / "s2_rgb_event.tif", i) for i in (1, 2, 3)])
    fu = wn.read(new_p); newm = np.isin(fu, [1, 2, 3])
    for a, (arr, ttl, kw) in zip(ax[1:], ((rgbE, "Sentinel-2 true colour, event composite", {}),
                                        (wn.read(sat / "s1_scene_20231023_vv.tif") - wn.read(sat / "s1_pre_event_vv.tif"), "Sentinel-1 change, 23 Oct minus pre-event", {"cmap": "RdBu", "vmin": -6, "vmax": 6}),
                                        (wn.read(sat / "s2_mndwi_event.tif"), "Sentinel-2 MNDWI, event composite", {"cmap": "BrBG", "vmin": -0.6, "vmax": 0.6}))):
        a.imshow(arr, extent=wn.ext, interpolation="nearest", **kw)
        base(a, ttl, wn)
        flag.boundary.plot(ax=a, color=FLAG_C, linewidth=0.4)
        a.set_xlim(wn.ext[0], wn.ext[1]); a.set_ylim(wn.ext[2], wn.ext[3])
    outline(ax[3], wn, newm, CHANGE_C, 0.7)
    ax[3].text(0.5, 0.04, f"our extent in view: {newm.sum() / 1e4:.2f} km²", transform=ax[3].transAxes, ha="center", fontsize=5.3,
               bbox=dict(fc="white", ec="none", alpha=0.85))
    label(0, "A", "Mepe: what 10 m data cannot see")
    # ------------------------------------------------ B  Ningo-Prampram
    wn = Win(ref, (861.0, 640.5, 870.0, 646.5))
    ax = [fig.add_subplot(gs[1, j]) for j in range(1, 5)]
    rgbE = rgb([wn.read(sat / "s2_rgb_event.tif", i) for i in (1, 2, 3)])
    ax[0].imshow(rgbE, extent=wn.ext, interpolation="nearest"); base(ax[0], "Sentinel-2 true colour and the two extents", wn)
    oldm = np.isin(wn.read(old_p), [1, 2, 3]); newm = np.isin(wn.read(new_p), [1, 2, 3])
    outline(ax[0], wn, oldm, OLD_C, 0.5); outline(ax[0], wn, newm, CHANGE_C, 0.5)
    ax[0].legend(handles=[Line2D([0], [0], color=OLD_C, label=f"event-window, 2 dates: {oldm.sum() / 1e4:.1f} km²"),
                          Line2D([0], [0], color=CHANGE_C, label=f"release period, 2 dates: {newm.sum() / 1e4:.1f} km²")], loc="lower left", fontsize=5.2, framealpha=0.9)
    preVV = wn.read(sat / "s1_pre_event_vv.tif")
    with rasterio.open(sat / "s1_change_by_date.tif") as src:
        for a, (orb, d, lab) in zip(ax[1:], ((74, "2023-08-12", "12 Aug (reference period)"), (74, "2023-10-23", "23 Oct (release)"), (74, "2024-01-03", "3 Jan 2024 (dry season)"))):
            code = src.read(band_idx(orb, d), window=wn.win, boundless=True, fill_value=0)
            a.imshow(preVV, extent=wn.ext, cmap="gray", vmin=-22, vmax=0, interpolation="nearest")
            fl_ = np.where((code & 4) > 0, 1.0, np.nan)
            a.imshow(fl_, extent=wn.ext, cmap=ListedColormap(["#E41A1C"]), interpolation="nearest")
            base(a, f"Flagged (drop > 3 dB), {lab}", wn)
            a.text(0.03, 0.04, f"{np.nansum(fl_) / 1e4:.1f} km² flagged", transform=a.transAxes, fontsize=5.3, color="white", bbox=dict(fc="black", ec="none", alpha=0.6))
    label(1, "B", "Ningo-Prampram: seasonal change, not flood")
    # ------------------------------------------------ C  Lagoon margin
    wn = Win(ref, (868.0, 640.5, 880.0, 648.5))
    ax = [fig.add_subplot(gs[2, j]) for j in range(1, 5)]
    fu = wn.read(new_p); newm = np.isin(fu, [1, 2, 3]); pre = fu == 4
    dom = None
    g = gpd.read_file(root / UN.format("FloodExtent")).to_crs(crs)
    uf = rasterize(((geom, 1) for geom in g.geometry), out_shape=(wn.h, wn.w), transform=wn.tr, fill=0, dtype="uint8").astype(bool)
    ax[0].imshow(rgb([wn.read(sat / "s2_rgb_event.tif", i) for i in (1, 2, 3)]), extent=wn.ext, interpolation="nearest")
    outline(ax[0], wn, newm, CHANGE_C, 0.5); outline(ax[0], wn, uf, AGREE["unosat"], 0.5); base(ax[0], "Sentinel-2, event: our extent and UNOSAT", wn)
    ax[0].legend(handles=[Line2D([0], [0], color=CHANGE_C, label="our extent"), Line2D([0], [0], color=AGREE["unosat"], label="UNOSAT flood")], loc="lower left", fontsize=5.4, framealpha=0.9)
    ax[1].imshow(rgb([wn.read(sat / "s2_rgb_post.tif", i) for i in (1, 2, 3)]), extent=wn.ext, interpolation="nearest")
    outline(ax[1], wn, newm, CHANGE_C, 0.5); base(ax[1], "Sentinel-2, post-event (Nov–Jan)", wn)
    wcv = wn.read(sat / "worldcover_2021.tif").astype(int)
    wc_cols = {10: "#006400", 20: "#FFBB22", 30: "#FFFF4C", 40: "#F096FF", 50: "#FA0000", 60: "#B4B4B4", 80: "#0064C8", 90: "#0096A0", 95: "#00CF75"}
    lut = np.zeros(101, int); keys_ = list(wc_cols)
    for i_, k_ in enumerate(keys_, 1):
        lut[k_] = i_
    ax[2].imshow(lut[np.clip(wcv, 0, 100)], extent=wn.ext, cmap=ListedColormap(["#FFFFFF"] + [wc_cols[k_] for k_ in keys_]), vmin=0, vmax=len(keys_), interpolation="nearest")
    osm = gpd.read_file(root / "data/raw/landuse/osm_landuse_songor.gpkg").to_crs(crs)
    ponds = osm[osm.landuse == "salt_pond"]
    ponds = ponds.set_geometry(ponds.geometry.scale(1 / 1000, 1 / 1000, origin=(0, 0)))
    ponds.boundary.plot(ax=ax[2], color="black", linewidth=0.5)
    outline(ax[2], wn, newm, "#FF00FF", 0.5)
    base(ax[2], "ESA WorldCover 2021; OSM salt ponds in black", wn)
    names_ = {40: "Cropland", 30: "Grassland", 60: "Bare/sparse", 80: "Water", 90: "Herb. wetland", 10: "Trees"}
    ax[2].legend(handles=[mp.Patch(color=wc_cols[k_], label=v_) for k_, v_ in names_.items()] + [Line2D([0], [0], color="#FF00FF", label="our extent")], loc="lower left", fontsize=4.8, framealpha=0.85, ncol=2)
    from src.flood_mapping.change_counts import count_flags
    cnt, _, _ = count_flags(sat)
    c_ = cnt[wn.win.row_off:wn.win.row_off + wn.h, wn.win.col_off:wn.win.col_off + wn.w]
    ax[3].imshow(np.where(pre, np.nan, c_), extent=wn.ext, cmap="YlOrRd", vmin=0, vmax=7, interpolation="nearest")
    ax[3].imshow(np.where(pre, 1.0, np.nan), extent=wn.ext, cmap=ListedColormap([CLASS["pre"]]), interpolation="nearest")
    base(ax[3], "Dates flagged (0–7); pale blue: pre-existing", wn)
    label(2, "C", "Songor lagoon margin: wetland fringe, cause unconfirmed")
    # ------------------------------------------------ D  River bars (TerraSAR-X sector)
    wn = Win(ref, (854.0, 670.0, 866.0, 679.0))
    ax = [fig.add_subplot(gs[3, j]) for j in range(1, 5)]
    fu = wn.read(new_p); newm = np.isin(fu, [1, 2, 3]); pre = fu == 4
    tsx = wn.read(val / "charter_terrasarx_20231016.tif"); tf = tsx == 1
    ax[0].imshow(rgb([wn.read(sat / "s2_rgb_event.tif", i) for i in (1, 2, 3)]), extent=wn.ext, interpolation="nearest")
    outline(ax[0], wn, newm, CHANGE_C, 0.5); outline(ax[0], wn, tf, REF_C, 0.5); base(ax[0], "Sentinel-2: our extent and TerraSAR-X flood", wn)
    ax[0].legend(handles=[Line2D([0], [0], color=CHANGE_C, label="our extent"), Line2D([0], [0], color=REF_C, label="TerraSAR-X flooded (16 Oct)")], loc="lower left", fontsize=5.4, framealpha=0.9)
    dv = wn.read(sat / "s1_scene_20231023_vv.tif") - wn.read(sat / "s1_pre_event_vv.tif")
    ax[1].imshow(dv, extent=wn.ext, cmap="RdBu", vmin=-6, vmax=6, interpolation="nearest"); base(ax[1], "Sentinel-1 change, 23 Oct minus pre-event", wn)
    ax[2].imshow(wn.read(sat / "s2_ndwi_event.tif"), extent=wn.ext, cmap="BrBG", vmin=-0.6, vmax=0.6, interpolation="nearest"); base(ax[2], "Sentinel-2 NDWI, event composite", wn)
    ag = np.zeros((wn.h, wn.w), int); ag[pre] = 4; ag[newm & ~tf] = 1; ag[tf & ~newm] = 2; ag[newm & tf] = 3
    ax[3].imshow(ag, extent=wn.ext, cmap=ListedColormap([AGREE["none"], AGREE["ours"], AGREE["tsx"], AGREE["both"], AGREE["pre"]]), vmin=0, vmax=4, interpolation="nearest")
    base(ax[3], "Agreement with TerraSAR-X", wn)
    ax[3].legend(handles=[mp.Patch(color=AGREE["both"], label="Both"), mp.Patch(color=AGREE["ours"], label="Ours only"), mp.Patch(color=AGREE["tsx"], label="TerraSAR-X only"),
                          mp.Patch(color=CLASS["pre"], label="Pre-existing water")], loc="lower left", fontsize=5.2, framealpha=0.9)
    label(3, "D", "River bars: where it works")
    out = root / "outputs/figures/fig_case_studies.png"
    fig.savefig(out, dpi=600)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
