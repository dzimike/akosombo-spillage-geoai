from __future__ import annotations
"""Recovery index v2: ranked index with Monte Carlo rank spread, and comparison of v1 and v2 ranks."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.paths import get_project_root
from src.visualization.palette import DISTRICT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
DIST_COL = dict(DISTRICT)


def main() -> None:
    root = get_project_root()
    d = pd.read_csv(root / "outputs/tables/community_recovery_index_v2.csv").dropna(subset=["rank_v2"]).sort_values("rank_v2")
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
    y = range(len(d))
    ax[0].barh(list(y), d.recovery_index_v2, color=[DIST_COL.get(x, "#999") for x in d.district], height=0.7)
    ax[0].errorbar(d.recovery_index_v2, list(y), xerr=None, fmt="none")
    for i, (v, sd, r) in enumerate(zip(d.recovery_index_v2, d.rank_std, d.rank_v2)):
        ax[0].text(v + 0.01, i, f"rank {int(r)} (±{sd:.1f})", va="center", fontsize=7)
    ax[0].set_yticks(list(y)); ax[0].set_yticklabels(d.community, fontsize=8); ax[0].invert_yaxis()
    ax[0].set_xlabel("Recovery index v2 (0 = slowest, 1 = fastest of the 22 assessed)"); ax[0].set_xlim(0, 1.15)
    ax[0].spines[["top", "right"]].set_visible(False); ax[0].grid(axis="x", ls=":", alpha=0.4)
    ax[0].set_title("A  Ranking on non-overlapping service areas (± Monte Carlo rank SD)", loc="left", fontsize=10)
    ax[1].plot([1, 22], [1, 22], color="#999", ls="--", lw=1)
    for _, r in d.iterrows():
        ax[1].scatter(r.rank_v1_assessed, r.rank_v2, color=DIST_COL.get(r.district, "#999"), s=40)
        ax[1].annotate(r.community, (r.rank_v1_assessed, r.rank_v2), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    ax[1].set_xlabel("Rank in v1 (overlapping 10 km buffers; same 22 communities)"); ax[1].set_ylabel("Rank in v2")
    ax[1].set_title("B  v1 versus v2 ranks", loc="left", fontsize=10)
    ax[1].spines[["top", "right"]].set_visible(False); ax[1].grid(ls=":", alpha=0.4)
    import matplotlib.patches as mp
    ax[0].legend(handles=[mp.Patch(color=c, label=k) for k, c in DIST_COL.items() if k in set(d.district)], fontsize=8, loc="lower right")
    fig.tight_layout()
    out = root / "outputs/figures/fig_recovery_index_v2.png"
    fig.savefig(out, dpi=200)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
