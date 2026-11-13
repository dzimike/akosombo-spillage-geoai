from __future__ import annotations
"""
Definition-by-reference agreement matrix (F1) with marginal bars.

Rows: flood definitions; columns: independent references (UNOSAT, TerraSAR-X, TanDEM-X, Mepe buildings).
Right bars: mean F1 over the three flood-area references; top bars: mean F1 of each reference over all definitions.
Data: outputs/tables/definition_reference_matrix.csv (src/validation/definition_reference_matrix.py).
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from src.utils.paths import get_project_root
from src.visualization.palette import DEFINITION, REFERENCE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
REFS = [("UNOSAT 23 Oct", "UNOSAT\n23 Oct", "unosat"), ("TerraSAR-X 16 Oct", "TerraSAR-X\n16 Oct\n(238 km²)", "tsx"),
        ("TanDEM-X 20 Oct", "TanDEM-X\n20 Oct\n(3.4 km² flood)", "tdx"), ("Mepe buildings", "Mepe flooded\nbuildings\n(179 flagged)", "mepe")]
SHORT = {
    "Open-water rule (event water outside static water)": "Open-water rule, event composite",
    "Optical water mask alone": "Optical water alone",
    "Open-water rule on the 23 Oct scene": "Open-water rule, 23 Oct scene",
    "Single scene 23 Oct, change rule (dVV, dVH < -3 dB)": "Change rule, 23 Oct scene only",
    "Event window 15 Sep-15 Nov, K=3 dB, >= 2 dates (fused)": "Change, event window, K=3, ≥2 dates",
    "SAR change mask alone (K=3 dB, >= 2 release dates)": "SAR change alone, K=3, ≥2 dates",
}


def main() -> None:
    root = get_project_root()
    d = pd.read_csv(root / "outputs/tables/definition_reference_matrix.csv")
    labels = []
    for n in d.definition:
        if n in SHORT:
            labels.append(SHORT[n])
        else:
            k = n.split("K=")[1].split(" dB")[0]; m = n.split(">= ")[1].split(" dates")[0]
            labels.append(("★ " if "[primary]" in n else "") + f"Change, release period, K={k}, ≥{m} date{'s' if m != '1' else ''}" + (" (primary)" if "[primary]" in n else ""))
    F = np.array([[d.loc[i, f"{rn} F1"] for rn, _, _ in REFS] for i in d.index])
    mean_rows = F[:, :3].mean(axis=1)
    n = len(d)
    fig = plt.figure(figsize=(11.5, 8.6))
    gs = GridSpec(2, 3, width_ratios=[3.6, 4.0, 1.6], height_ratios=[1.0, 7.0], wspace=0.03, hspace=0.05, figure=fig, bottom=0.17)
    axL, axH, axR, axT = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[0, 1])
    fig.canvas.draw()
    pos = axH.get_position()
    axC = fig.add_axes([pos.x0 + 0.08 * pos.width, 0.095, 0.84 * pos.width, 0.014])
    # heatmap
    im = axH.imshow(F, cmap="YlGnBu", vmin=0, vmax=0.65, aspect="auto")
    for i in range(n):
        for j in range(F.shape[1]):
            v = F[i, j]
            axH.text(j, i, f"{v:.2f}" if v >= 0.1 else f"{v:.3f}", ha="center", va="center", fontsize=8,
                     color="white" if v > 0.40 else "#222")
    best = F.argmax(axis=0)
    for j, i in enumerate(best):
        axH.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#C62828", lw=2))
    prim = [i for i, nme in enumerate(d.definition) if "[primary]" in nme][0]
    axH.add_patch(plt.Rectangle((-0.5, prim - 0.5), F.shape[1], 1, fill=False, ec="black", lw=1.6, ls="--"))
    axH.set_xticks(range(len(REFS))); axH.set_xticklabels([r[1] for r in REFS], fontsize=8)
    axH.xaxis.tick_bottom(); axH.set_yticks([])
    axH.set_xticks(np.arange(-.5, F.shape[1], 1), minor=True); axH.set_yticks(np.arange(-.5, n, 1), minor=True)
    axH.grid(which="minor", color="white", lw=1.2); axH.tick_params(which="minor", length=0)
    for s in axH.spines.values():
        s.set_visible(False)
    # group separators
    groups = list(d.group)
    for i in range(1, n):
        if groups[i] != groups[i - 1]:
            axH.axhline(i - 0.5, color="#333", lw=1.4)
    # left labels
    axL.set_xlim(0, 1); axL.set_ylim(n - 0.5, -0.5); axL.axis("off")
    for i in range(n):
        axL.text(1.0, i, labels[i], ha="right", va="center", fontsize=8.2, fontweight="bold" if i == prim else "normal")
        axL.text(-0.02, i, f"{d.area_km2[i]:.0f}", ha="left", va="center", fontsize=7.5, color="#555")
    axL.text(-0.02, -1.2, "Area\n(km²)", ha="left", va="bottom", fontsize=7.5, color="#555")
    # right marginal bars
    axR.barh(range(n), mean_rows, color=[DEFINITION[k] for k in d.key], height=0.7)
    for i, v in enumerate(mean_rows):
        axR.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7.5)
    axR.set_ylim(n - 0.5, -0.5); axR.set_xlim(0, 0.5); axR.set_yticks([])
    axR.set_xlabel("Mean F1, three\nflood-area references", fontsize=8); axR.tick_params(labelsize=7)
    axR.spines[["top", "right", "left"]].set_visible(False)
    # top marginal bars
    colmean = F.mean(axis=0)
    axT.bar(range(len(REFS)), colmean, color=[REFERENCE[r[2]] for r in REFS], width=0.62)
    for j, v in enumerate(colmean):
        axT.text(j, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    axT.set_xlim(-0.5, len(REFS) - 0.5); axT.set_ylim(0, 0.65); axT.set_xticks([])
    axT.set_ylabel("Mean F1\nover rows", fontsize=8); axT.tick_params(labelsize=7)
    axT.spines[["top", "right"]].set_visible(False)
    cb = fig.colorbar(im, cax=axC, orientation="horizontal")
    cb.set_label("F1 against reference", fontsize=8); cb.ax.tick_params(labelsize=7)
    out = root / "outputs/figures/fig_definition_reference_matrix.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
