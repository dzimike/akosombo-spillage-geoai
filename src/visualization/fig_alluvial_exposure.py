from __future__ import annotations
"""
Alluvial figure.
A  Area flowing between the open-water definition (72.2 km²) and the change-based extent (44.2 km²).
B  Detection class of the fused extent (SAR change only / optical only / both) flowing through area, buildings, people
   and road length, each axis shown as a share of its own total.
Data: outputs/tables/exposure_by_detection_class.csv, definition_area_overlap.csv
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from src.utils.paths import get_project_root
from src.visualization.palette import CLASS, DEFINITION

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
BAR_W = 0.07


def ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.45):
    """Filled Bezier band from [y0a, y0b] at x0 to [y1a, y1b] at x1."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec="none", alpha=alpha, zorder=1))


def stack(vals, gap=0.012, top=1.0):
    """Return [(y_top, y_bottom)] for stacked segments from the top, heights proportional to vals."""
    tot = float(sum(vals)); h = top - gap * (len(vals) - 1)
    out, y = [], top
    for v in vals:
        out.append((y, y - h * v / tot)); y = y - h * v / tot - gap
    return out


def main() -> None:
    root = get_project_root()
    ex = pd.read_csv(root / "outputs/tables/exposure_by_detection_class.csv")
    ov = pd.read_csv(root / "outputs/tables/definition_area_overlap.csv").iloc[0]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [0.9, 2.0]})
    # ---- Panel A
    axA.set_xlim(-0.25, 1.25); axA.set_ylim(-0.12, 1.12); axA.axis("off")
    scale = 1.0 / ov.open_water_total_km2
    ow_both, ow_only, ch_only = ov.both_km2, ov.open_water_only_km2, ov.change_only_km2
    # left axis (open-water definition), stacked from top: only, both
    yL_only = (1.0, 1.0 - ow_only * scale); yL_both = (yL_only[1], yL_only[1] - ow_both * scale)
    # right axis (change extent), centred vertically against the left axis: both on top of change only
    r_tot = (ow_both + ch_only) * scale
    y_top = 1.0 - (1.0 - r_tot) / 2 - 0.0
    yR_both = (y_top, y_top - ow_both * scale); yR_only = (yR_both[1], yR_both[1] - ch_only * scale)
    ribbon(axA, BAR_W / 2 * 0 + 0.0, yL_both[0], yL_both[1], 1.0, yR_both[0], yR_both[1], "#888888", 0.5)
    for (ya, yb), col, lab, x in ((yL_only, DEFINITION["openwater"], f"Open-water only\n{ow_only:.1f} km²", 0.0),
                                   (yL_both, "#888888", f"In both\n{ow_both:.1f} km²", 0.0),
                                   (yR_both, "#888888", f"In both\n{ow_both:.1f} km²", 1.0), (yR_only, DEFINITION["change"], f"Change only\n{ch_only:.1f} km²", 1.0)):
        axA.add_patch(Rectangle((x - BAR_W / 2, yb), BAR_W, ya - yb, fc=col, ec="white", zorder=2))
        if lab:
            axA.text(x - BAR_W / 2 - 0.03 if x == 0 else x + BAR_W / 2 + 0.03, (ya + yb) / 2, lab, ha="right" if x == 0 else "left",
                     va="center", fontsize=9)
    axA.text(0, 1.06, f"Open-water definition\n{ov.open_water_total_km2:.1f} km²", ha="center", fontsize=10, fontweight="bold")
    axA.text(1.0, 1.06, f"Change-based extent\n{ov.change_total_km2:.1f} km²", ha="center", fontsize=10, fontweight="bold")
    axA.text(0.5, (yL_both[0] + yL_both[1]) / 2 + 0.0, "shared", ha="center", va="center", fontsize=8, color="#333")
    axA.text(0.0, -0.09, f"{ov.open_water_only_in_preexisting_km2:.0f} of the {ow_only:.0f} km² open-water-only area\nlies in pre-existing (pre-event SAR-dark) water",
             ha="center", fontsize=8, color="#555")
    axA.set_title("a  Area shared by the two definitions", loc="left", fontsize=11)
    # ---- Panel B
    axB.set_xlim(-0.45, 3.45); axB.set_ylim(-0.12, 1.12); axB.axis("off")
    metrics = [("area_km2", "Flooded area", "{:.1f} km²"), ("buildings", "Buildings exposed", "{:.0f}"),
               ("people", "People exposed", "{:,.0f}"), ("roads_km", "Flooded road", "{:.1f} km")]
    keys = ["sar", "optical", "both"]
    stacks = []
    for xi, (col, title, fmt) in enumerate(metrics):
        vals = ex[col].tolist(); tot = sum(vals)
        stacks.append(stack([max(v, tot * 0.004) for v in vals]))
        for k, (ya, yb), v in zip(keys, stacks[-1], vals):
            axB.add_patch(Rectangle((xi - BAR_W / 2, yb), BAR_W, ya - yb, fc=CLASS[k], ec="white", zorder=2))
            share = 100 * v / tot
            if ya - yb > 0.05:
                axB.text(xi + BAR_W / 2 + 0.03, (ya + yb) / 2, f"{share:.0f}%  ({fmt.format(v)})", va="center", fontsize=8.5)
            else:
                axB.text(xi + BAR_W / 2 + 0.03, (ya + yb) / 2, f"{share:.0f}%", va="center", fontsize=7.5, color="#444")
        axB.text(xi, 1.06, f"{title}\n{fmt.format(tot)}", ha="center", fontsize=10, fontweight="bold")
    for xi in range(3):
        for k, a, b in zip(keys, stacks[xi], stacks[xi + 1]):
            ribbon(axB, xi + BAR_W / 2, a[0], a[1], xi + 1 - BAR_W / 2, b[0], b[1], CLASS[k], 0.45)
    import matplotlib.patches as mp
    axB.legend(handles=[mp.Patch(color=CLASS[k], label=l) for k, l in zip(keys, ["SAR change only (Class 1)", "Optical water only (Class 2)", "SAR and optical (Class 3)"])],
               loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    axB.set_title("b  What each detector contributes to the exposure", loc="left", fontsize=11)
    fig.tight_layout()
    out = root / "outputs/figures/fig_alluvial_exposure.png"
    fig.savefig(out, dpi=600)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
