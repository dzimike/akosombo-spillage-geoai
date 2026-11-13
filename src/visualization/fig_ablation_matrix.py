from __future__ import annotations
"""
Pairwise feature-group ablation (XGBoost), drawn at final size (174 mm wide).

a  Test-block F1 after removing one group (diagonal) or two groups (off-diagonal), mean over ten block splits; colour = change from the full model.
b  Leave-one-block-out mean F1, same layout (one model per held-out block, so independent of the train/test split).
c  Test F1 of each group alone, mean and standard deviation over ten seeds (a seed changes the block split and the model random state).
d  Change in test F1 from removing each group, one point per seed.
Data: outputs/tables/xgb_ablation_pairwise.csv (ablation_pairwise.py, seed 42), xgb_ablation_seeds.csv (ablation_seeds.py).
The matrices are symmetric, so only the lower triangle is shown.
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from src.models.ablation_pairwise import GROUPS
from src.utils.paths import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
G = list(GROUPS)
SHORT = {"Terrain": "Terrain", "Proximity": "Proximity", "Optical event": "Optical\nevent", "Optical pre": "Optical\npre", "SAR": "SAR", "Settlement": "Settlement"}
FS = 5.6                                     # base font size in points at final width


def mat(df, col):
    M = np.full((6, 6), np.nan)
    for i, a in enumerate(G):
        for j, b in enumerate(G):
            if j > i:
                continue                      # lower triangle only
            if i == j:
                v = df[df.variant == f"drop:{a}"][col]
            else:
                v = df[df.variant == f"drop:{G[j]}+{G[i]}"][col]
            M[i, j] = v.iloc[0]
    return M


def fmt(d):
    s = f"{d:+.2f}"
    return "0.00" if s in ("+0.00", "-0.00") else s


def heat(ax, M, full, title, vmin):
    D = M - full
    im = ax.imshow(np.ma.masked_invalid(D), cmap="Reds_r", vmin=vmin, vmax=0)
    for i in range(6):
        for j in range(i + 1):
            ax.text(j, i, f"{M[i, j]:.2f}\n({fmt(D[i, j])})", ha="center", va="center", fontsize=FS - 0.4,
                    color="white" if D[i, j] < vmin * 0.6 else "#222", fontweight="bold" if i == j else "normal", linespacing=1.0)
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=0.8))
    labs = [f"{SHORT[g]}\n({len(GROUPS[g])})" for g in G]
    ax.set_xticks(range(6)); ax.set_xticklabels(labs, fontsize=FS - 0.4, linespacing=0.95)
    ax.set_yticks(range(6)); ax.set_yticklabels(labs, fontsize=FS - 0.4, linespacing=0.95)
    ax.tick_params(length=2, pad=1.5)
    ax.set_title(title, fontsize=FS + 1, loc="left", pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def main() -> None:
    root = get_project_root()
    df = pd.read_csv(root / "outputs/tables/xgb_ablation_pairwise.csv")
    sd = pd.read_csv(root / "outputs/tables/xgb_ablation_seeds.csv")
    full = df[df.variant == "full"].iloc[0]
    W, H = 6.85, 6.3
    fig = plt.figure(figsize=(W, H))
    ax_ = lambda l, b_, w, h: fig.add_axes([l / W, b_ / H, w / W, h / H])          # inches from the lower-left corner
    axA, axB = ax_(0.80, 3.25, 2.80, 2.80), ax_(3.85, 3.25, 2.80, 2.80)
    axC, axD = ax_(1.00, 0.50, 2.15, 1.45), ax_(4.35, 0.50, 2.35, 1.45)
    agg = sd.groupby("variant").test_f1.mean().reset_index()
    im = heat(axA, mat(agg, "test_f1"), agg[agg.variant == "full"].test_f1.iloc[0], "a  Test-block F1, mean of 10 splits", -0.45)
    heat(axB, mat(df, "lobo_f1_mean"), full.lobo_f1_mean, "b  Leave-one-block-out mean F1", -0.45)
    axB.set_yticklabels([])
    cax = ax_(2.35, 2.60, 2.4, 0.09)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("Change in F1 from the full model", fontsize=FS, labelpad=1.5); cb.ax.tick_params(labelsize=FS - 0.4, length=2, pad=1)

    seeds = sorted(sd.seed.unique()); y = np.arange(6)
    fullf1 = sd[sd.variant == "full"].set_index("seed").test_f1
    # c: group alone, mean +/- sd over seeds
    mu = [sd[sd.variant == f"only:{g}"].test_f1.mean() for g in G]; s_ = [sd[sd.variant == f"only:{g}"].test_f1.std() for g in G]
    axC.barh(y, mu, xerr=s_, height=0.6, color="#555555", ecolor="black", error_kw=dict(lw=0.6, capsize=1.5, capthick=0.6))
    axC.axvline(fullf1.mean(), color="black", ls="--", lw=0.7)
    for i, (m_, e_) in enumerate(zip(mu, s_)):
        axC.text(min(m_ + e_ + 0.035, 1.0), i, f"{m_:.2f}", va="center", fontsize=FS - 0.4)
    axC.set_yticks(y); axC.set_yticklabels([SHORT[g].replace("\n", " ") for g in G], fontsize=FS - 0.2); axC.set_ylim(5.6, -1.0); axC.set_xlim(0, 1.12)
    axC.set_xlabel("Test-block F1 of the group alone", fontsize=FS); axC.set_title("c  Group alone (mean ± SD, 10 seeds)", fontsize=FS + 1, loc="left", pad=3)
    axC.text(fullf1.mean() - 0.012, -0.78, f"full model {fullf1.mean():.2f}", ha="right", va="center", fontsize=FS - 0.6)
    # d: change from removing each group, per seed
    rng = np.random.default_rng(1)
    for i, g in enumerate(G):
        dv = np.array([sd[(sd.variant == f"drop:{g}") & (sd.seed == s)].test_f1.iloc[0] - fullf1[s] for s in seeds])
        axD.scatter(dv, i + rng.uniform(-0.18, 0.18, len(dv)), s=5, color="#555555", alpha=0.75, lw=0)
        axD.plot([dv.mean() - dv.std(), dv.mean() + dv.std()], [i, i], color="black", lw=0.9)
        axD.plot([dv.mean()], [i], marker="|", color="black", ms=5, mew=0.9)
        b = sd[(sd.variant == f"drop:{g}") & (sd.seed == 42)].test_f1.iloc[0] - fullf1[42]
        axD.plot([b], [i], marker="D", ms=2.8, mfc="none", mec="#D55E00", mew=0.7)
    axD.axvline(0, color="black", lw=0.6)
    axD.set_yticks(y); axD.set_yticklabels([SHORT[g].replace("\n", " ") for g in G], fontsize=FS - 0.2); axD.set_ylim(5.6, -1.0)
    axD.set_xlabel("Change in test-block F1 when the group is removed", fontsize=FS); axD.set_title("d  Group removed, one point per seed", fontsize=FS + 1, loc="left", pad=3)
    for ax in (axC, axD):
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=FS - 0.4, length=2, pad=1.5)
    out = root / "outputs/figures/fig_ablation_matrix.png"
    fig.savefig(out, dpi=600)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
