from __future__ import annotations
"""
Figure: per-date Sentinel-1 change time series (orbits 74 and 147, July 2023 - January 2024).

Panel A  share of the final flood extent flagged on each date (dVV and dVH < -3 dB against the orbit-matched pre-event median)
Panel B  the same for patch groups of the event-window definition (river/lagoon-adjacent >= 2 ha, isolated >= 2 ha,
         Ningo-Prampram isolated scatter, Ada West lagoon-adjacent): the scatter flags in Jul-Aug and from November on
Panel C  area flagged on land outside pre-existing water for the whole domain, on a logarithmic axis: dry-season change
         dominates from late November, which is why persistence is counted inside the release period only
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.paths import get_project_root
from src.visualization.palette import DEFINITION, PATCH_GROUP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    root = get_project_root()
    new = pd.read_csv(root / "outputs/tables/s1_change_timeseries.csv", parse_dates=["date"])
    old = pd.read_csv(root / "outputs/tables/s1_change_timeseries_eventwindow_definition.csv", parse_dates=["date"])
    fig, ax = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    mk = {74: "o", 147: "s"}

    def shade(a):
        a.axvspan(pd.Timestamp("2023-09-15"), pd.Timestamp("2023-10-30"), color="#BDBDBD", alpha=0.55, lw=0)
        a.axvline(pd.Timestamp("2023-10-09"), color="#555555", ls="--", lw=1)
        a.axvspan(pd.Timestamp("2023-08-01"), pd.Timestamp("2023-09-10"), color="#E4E4E4", alpha=0.9, lw=0)
        a.grid(ls=":", alpha=0.4); a.spines[["top", "right"]].set_visible(False)

    # A: final flood, release-period definition
    d = new[new.region == "final_flood"]
    for o, g in d.groupby("orbit"):
        ax[0].plot(g.date, g.flagged_share_of_valid, marker=mk[o], color=DEFINITION["change"], ls="-", lw=1, ms=5, label=f"orbit {o}")
    ax[0].set_ylabel("Share of flood extent flagged"); ax[0].set_title("a  Final (release-period) flood extent", loc="left", fontsize=10)
    ax[0].legend(fontsize=8); shade(ax[0])
    # B: groups of the event-window definition
    labs = {"patches_adjacent_ge2ha": ("river/lagoon-adjacent ≥ 2 ha", PATCH_GROUP["adjacent"]), "patches_isolated_ge2ha": ("isolated ≥ 2 ha", PATCH_GROUP["isolated"]),
            "patches_NingoPrampram_isolated": ("Ningo-Prampram isolated scatter", PATCH_GROUP["ningo"]), "patches_AdaWest_adjacent": ("Ada West lagoon-adjacent", PATCH_GROUP["adawest"])}
    for reg, (lab, col) in labs.items():
        g = old[old.region == reg].sort_values("date")
        ax[1].plot(g.date, g.flagged_share_of_valid, "-", color=col, lw=1, alpha=0.6)
        for o, gg in g.groupby("orbit"):
            ax[1].plot(gg.date, gg.flagged_share_of_valid, mk[o], color=col, ms=5, label=lab if o == 74 else None)
    ax[1].set_ylabel("Share of patch group flagged"); ax[1].set_title("b  Patch groups of the 15 Sep–15 Nov two-date definition", loc="left", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper left"); shade(ax[1])
    # C: AOI land flagged area
    d = new[new.region == "AOI_land"]
    for o, g in d.groupby("orbit"):
        ax[2].plot(g.date, g.flagged_km2.clip(lower=0.1), marker=mk[o], color=DEFINITION["change"], lw=1, ms=5, label=f"orbit {o} (valid {g.valid_share_of_region.mean():.0%} of domain)")
    ax[2].set_yscale("log"); ax[2].set_ylabel("Area flagged, whole domain (km², log)")
    ax[2].set_title("c  All land outside pre-existing water", loc="left", fontsize=10); ax[2].legend(fontsize=8); shade(ax[2])
    ax[2].set_xlim(pd.Timestamp("2023-07-01"), pd.Timestamp("2024-02-05"))
    fig.tight_layout()
    out = root / "outputs/figures/fig_change_timeseries.png"
    fig.savefig(out, dpi=600)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
