#!/usr/bin/env python3
"""
Regenerate all 24-community figures.

Steps:
  1. Compute Monte Carlo rank frequencies for all 24 communities and save
     outputs/tables/recovery_weight_sensitivity_24.csv
  2. Run publication_figures.py fig4 + fig7
  3. Run uncertainty_figures.py fig4 + fig5
  4. Regenerate community_recovery_index_map.png

Run from project root:
    python regenerate_24_figures.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path

ROOT   = Path(__file__).parent
TABLES = ROOT / "outputs" / "tables"
FIGS   = ROOT / "outputs" / "figures"
UFIGS  = FIGS / "uncertainty_paper"
MAPS   = ROOT / "outputs" / "maps"
FIGS.mkdir(exist_ok=True)
UFIGS.mkdir(exist_ok=True)

DPI = 300

import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.visualization.palette import DISTRICT as _PAL_DISTRICT, FLOOD_EXTENT as _FLOOD
DISTRICT_COLORS = dict(_PAL_DISTRICT)
DISTRICT_ORDER = ["North Tongu", "Shai-Osudoku", "Lower Manya Krobo",
                  "Ada East", "Central Tongu", "South Tongu"]


def load_recovery_index_csv(path: Path) -> pd.DataFrame:
    """Read community_recovery_index_24.csv, normalising the community-id column name."""
    df = pd.read_csv(path)
    if "community" in df.columns and "community_name" not in df.columns:
        df = df.rename(columns={"community": "community_name"})
    return df


def panel_tag(ax, letter, fontsize=15):
    """Draw a minimal bold panel-letter tag above an axes' top-left corner."""
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="bottom", ha="right")


# ============================================================
# Step 1: Monte Carlo rank frequencies
# ============================================================
def generate_mc_rank_frequencies() -> pd.DataFrame:
    print("Step 1: computing Monte Carlo rank frequencies…")
    ri_path = TABLES / "community_recovery_index_24.csv"
    df = load_recovery_index_csv(ri_path)

    comp_cols = ["ndvi_norm", "wrec_norm", "racc_norm", "setl_norm"]
    components = df[comp_cols].values.astype(float)
    communities = df["community_name"].values
    n = len(communities)

    from scipy.stats import dirichlet as _dirichlet
    np.random.seed(42)
    n_draws = 2000
    weight_draws = _dirichlet.rvs([2] * 4, size=n_draws, random_state=42)

    rank_counts = np.zeros((n, n), dtype=int)
    for w in weight_draws:
        scores = components @ w
        # rank 1 = highest score
        order  = np.argsort(scores)[::-1]
        ranks  = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        for i, r in enumerate(ranks):
            rank_counts[i, r - 1] += 1

    rank_pcts = rank_counts / n_draws * 100

    rows = []
    for i, comm in enumerate(communities):
        row = {"community_name": comm}
        for r in range(n):
            row[f"rank_{r + 1}_pct"] = round(rank_pcts[i, r], 2)
        row["modal_rank"] = int(np.argmax(rank_counts[i]) + 1)
        row["rank_std"] = float(
            df.loc[df["community_name"] == comm, "rank_std"].values[0])
        rows.append(row)

    result = pd.DataFrame(rows)
    out = TABLES / "recovery_weight_sensitivity_24.csv"
    result.to_csv(out, index=False)
    print(f"  Saved: {out}  ({n} communities × {n} ranks)")
    return result


# ============================================================
# Step 2: Fig 4 — community exposure hotspots (24-community)
# ============================================================
def make_fig4_exposure(out: Path) -> None:
    """Community flood exposure summary as a single large 4-panel figure
    (A-D), using the actual computed exposure fields (exposed_buildings,
    exposed_pop, flooded_road_km) rather than raw service-buffer totals.
    No embedded descriptive titles — panels carry only a small letter tag;
    full captions live in the manuscript text."""
    print("Step 2: fig4 exposure hotspots (4-panel)…")
    from src.visualization.narrow_paper_figures import exposure as _district_exposure
    _district_exposure(ROOT, out)   # district-level (non-overlapping) exposure figure
    return
    cs_path = TABLES / "community_exposure_summary_24.csv"
    if not cs_path.exists():
        print("  [skip] community_exposure_summary_24.csv not found")
        return

    try:
        from adjustText import adjust_text
        HAVE_ADJUST_TEXT = True
    except ImportError:
        HAVE_ADJUST_TEXT = False

    cs = pd.read_csv(cs_path)
    cs["pct_buildings_exposed"] = 100 * cs["exposed_buildings"] / cs["total_buildings"]

    dist_handles = [mpatches.Patch(facecolor=DISTRICT_COLORS[d], label=d)
                    for d in DISTRICT_ORDER]

    fig = plt.figure(figsize=(26, 22))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30,
                          left=0.06, right=0.98, top=0.96, bottom=0.08)

    # Panel A — exposed buildings
    ax1 = fig.add_subplot(gs[0, 0])
    cs_a = cs.sort_values("exposed_buildings", ascending=True).reset_index(drop=True)
    bc = [DISTRICT_COLORS.get(d, "#888888") for d in cs_a["district"]]
    bars = ax1.barh(cs_a["community_name"], cs_a["exposed_buildings"], color=bc, height=0.68)
    for bar, v in zip(bars, cs_a["exposed_buildings"]):
        ax1.text(bar.get_width() + cs_a["exposed_buildings"].max() * 0.015,
                 bar.get_y() + bar.get_height() / 2, f"{int(v):,}", va="center", fontsize=12)
    ax1.set_xlim(0, cs_a["exposed_buildings"].max() * 1.22)
    ax1.set_xlabel("Exposed buildings in service buffer (#)", fontsize=14)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", linestyle=":", alpha=0.4)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.tick_params(labelsize=13)
    panel_tag(ax1, "A", fontsize=17)

    # Panel B — exposed population
    ax2 = fig.add_subplot(gs[0, 1])
    cs_b = cs.sort_values("exposed_pop", ascending=True).reset_index(drop=True)
    bc = [DISTRICT_COLORS.get(d, "#888888") for d in cs_b["district"]]
    bars = ax2.barh(cs_b["community_name"], cs_b["exposed_pop"], color=bc, height=0.68)
    for bar, v in zip(bars, cs_b["exposed_pop"]):
        ax2.text(bar.get_width() + cs_b["exposed_pop"].max() * 0.015,
                 bar.get_y() + bar.get_height() / 2, f"{int(v):,}", va="center", fontsize=12)
    ax2.set_xlim(0, cs_b["exposed_pop"].max() * 1.22)
    ax2.set_xlabel("Estimated exposed population (WorldPop 2020)", fontsize=14)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", linestyle=":", alpha=0.4)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax2.tick_params(labelsize=13)
    panel_tag(ax2, "B", fontsize=17)

    # Panel C — flooded road length
    ax3 = fig.add_subplot(gs[1, 0])
    cs_c = cs.sort_values("flooded_road_km", ascending=True).reset_index(drop=True)
    bc = [DISTRICT_COLORS.get(d, "#888888") for d in cs_c["district"]]
    bars = ax3.barh(cs_c["community_name"], cs_c["flooded_road_km"], color=bc, height=0.68)
    for bar, v in zip(bars, cs_c["flooded_road_km"]):
        ax3.text(bar.get_width() + cs_c["flooded_road_km"].max() * 0.015,
                 bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center", fontsize=12)
    ax3.set_xlim(0, cs_c["flooded_road_km"].max() * 1.22)
    ax3.set_xlabel("Flooded road length in service buffer (km)", fontsize=14)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(axis="x", linestyle=":", alpha=0.4)
    ax3.tick_params(labelsize=13)
    panel_tag(ax3, "C", fontsize=17)

    # Panel D — flood footprint area vs. building exposure rate,
    # circle size proportional to exposed population; ALL 24 communities
    # labelled using automatic label placement to avoid overlap.
    ax4 = fig.add_subplot(gs[1, 1])
    x_max = cs["flood_area_km2"].max()
    y_max = cs["pct_buildings_exposed"].max()
    pop_min, pop_max = cs["exposed_pop"].min(), cs["exposed_pop"].max()
    sizes = 60 + 400 * (cs["exposed_pop"] - pop_min) / max(pop_max - pop_min, 1)

    for dist in DISTRICT_ORDER:
        sub = cs[cs["district"] == dist]
        if sub.empty:
            continue
        sub_sizes = sizes.loc[sub.index]
        ax4.scatter(sub["flood_area_km2"], sub["pct_buildings_exposed"],
                   s=sub_sizes, c=DISTRICT_COLORS[dist], alpha=0.85,
                   edgecolors="white", lw=0.7, zorder=3, label=dist)

    texts = []
    for idx, row in cs.iterrows():
        x, y = row["flood_area_km2"], row["pct_buildings_exposed"]
        texts.append(ax4.text(x, y, row["community_name"], fontsize=10.5, zorder=5))

    if HAVE_ADJUST_TEXT:
        adjust_text(texts, ax=ax4,
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6),
                    expand=(2.2, 2.6), force_text=(1.3, 1.7),
                    force_static=(0.6, 0.8), force_pull=(0.02, 0.02),
                    lim=3000)
    else:
        # fallback: simple vertical stagger if adjustText unavailable
        for i, t in enumerate(texts):
            t.set_position((t.get_position()[0] + 5, t.get_position()[1] + (i % 3 - 1) * 3))

    ax4.set_xlabel("Flood footprint area in service buffer (km²)", fontsize=14)
    ax4.set_ylabel("Building exposure rate (%)", fontsize=14)
    ax4.set_xlim(0, x_max * 1.12)
    ax4.set_ylim(-0.005 * max(y_max, 1e-9) * 10, y_max * 1.30)
    ax4.grid(linestyle=":", alpha=0.3)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.tick_params(labelsize=13)
    panel_tag(ax4, "D", fontsize=17)

    size_ref = [pop_min, (pop_min + pop_max) / 2, pop_max]
    size_handles = [Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="grey", markeredgecolor="grey",
                           markersize=(60 + 400 * (v - pop_min) / max(pop_max - pop_min, 1)) ** 0.5 / 2.6,
                           label=f"{int(v):,}")
                    for v in size_ref]
    leg_size = ax4.legend(handles=size_handles, title="Exposed population",
                          loc="upper left", fontsize=10, title_fontsize=10.5, framealpha=0.9)
    ax4.add_artist(leg_size)

    # Shared district legend across the whole figure
    fig.legend(handles=dist_handles, loc="lower center", ncol=6,
              fontsize=13, title="District", title_fontsize=13.5,
              framealpha=0.95, bbox_to_anchor=(0.5, 0.02))

    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Step 3: Fig 7 — recovery analysis (24-community)
# ============================================================
def make_fig7_recovery(out: Path) -> None:
    print("Step 3: fig7_recovery_analysis…")
    ri_path = TABLES / "community_recovery_index_24.csv"
    if not ri_path.exists():
        print("  [skip] community_recovery_index_24.csv not found"); return

    df_ri = load_recovery_index_csv(ri_path)

    exp_path = TABLES / "community_exposure_summary_24.csv"
    df_exp   = pd.read_csv(exp_path).set_index("community_name") \
               if exp_path.exists() else pd.DataFrame()

    df_bar   = df_ri.sort_values("recovery_index", ascending=True).reset_index(drop=True)
    df_heat  = df_ri.sort_values("recovery_index", ascending=False).reset_index(drop=True)
    n        = len(df_ri)
    cmap_ri  = plt.colormaps["RdYlGn"]

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.34)

    # Panel A — recovery index bars
    ax1 = fig.add_subplot(gs[0, 0])
    ri_bar_c = [cmap_ri(v) for v in df_bar["recovery_index"]]
    bars = ax1.barh(df_bar["community_name"], df_bar["recovery_index"],
                    color=ri_bar_c, edgecolor="grey", lw=0.3, height=0.72)
    # District stripe
    for i, (_, row) in enumerate(df_bar.iterrows()):
        dc = DISTRICT_COLORS.get(row["district"], "#888888")
        ax1.barh(row["community_name"], 0.015,
                 color=dc, height=0.72, left=0, zorder=4)
    ax1.axvline(0.5, color="grey", lw=1.0, linestyle=":", alpha=0.6)
    for bar, val in zip(bars, df_bar["recovery_index"]):
        ax1.text(bar.get_width() + 0.012,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=7.5, fontweight="bold")
    ax1.set_xlim(0, 1.17)
    ax1.set_xlabel("Recovery Index (0 = least  ·  1 = most)", fontsize=9)
    panel_tag(ax1, "A")
    ax1.grid(axis="x", linestyle=":", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap_ri, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax1, fraction=0.022, pad=0.03, shrink=0.55)
    cb.set_label("Index value", fontsize=7)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["Low", "Mid", "High"])
    cb.ax.tick_params(labelsize=6)

    # Panel B — component heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    comp_cols   = ["ndvi_norm", "wrec_norm", "racc_norm", "setl_norm"]
    comp_labels = ["Veg\nrecovery", "Water\nrecession", "Road\naccess", "Settlement\nproxy"]
    heat_data   = df_heat[comp_cols].values
    im = ax2.imshow(heat_data, aspect="auto", cmap="RdYlGn",
                    vmin=0, vmax=1, interpolation="nearest")
    ax2.set_xticks(range(4)); ax2.set_xticklabels(comp_labels, fontsize=8.5)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(
        [f"{c}  [{d[:2]}]" for c, d in
         zip(df_heat["community_name"], df_heat["district"])],
        fontsize=7.5)
    panel_tag(ax2, "B")
    for i in range(n):
        for j in range(4):
            v = heat_data[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=5.5,
                     color="white" if (v < 0.35 or v > 0.75) else "#222222")
    cb2 = fig.colorbar(im, ax=ax2, fraction=0.022, pad=0.03, shrink=0.55)
    cb2.set_label("Normalised score", fontsize=7)
    cb2.ax.tick_params(labelsize=6)

    # Panel C — flood area vs recovery index scatter
    ax3 = fig.add_subplot(gs[1, 0])
    if not df_exp.empty and "flood_area_km2" in df_exp.columns:
        joined = df_ri.set_index("community_name").join(
            df_exp[["flood_area_km2"]], how="left")
        for dist in DISTRICT_ORDER:
            sub = joined[joined["district"] == dist]
            if sub.empty: continue
            ax3.scatter(sub["flood_area_km2"], sub["recovery_index"],
                        c=DISTRICT_COLORS[dist], s=55, alpha=0.85,
                        edgecolors="white", lw=0.5, zorder=3, label=dist)
            for comm, row in sub.iterrows():
                ax3.annotate(comm,
                             (row["flood_area_km2"], row["recovery_index"]),
                             textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax3.axhline(0.5, color="grey", lw=0.8, linestyle=":", alpha=0.6)
    ax3.set_xlabel("Flood area in service buffer (km²)", fontsize=9)
    ax3.set_ylabel("Recovery Index", fontsize=9)
    panel_tag(ax3, "C")
    ax3.grid(linestyle=":", alpha=0.3)
    ax3.spines[["top", "right"]].set_visible(False)
    # (district legend is the shared figure-level legend below; a panel legend here
    # hid the two highest-flood-area communities)

    # Panel D — rank standard deviation
    ax4 = fig.add_subplot(gs[1, 1])
    df_std    = df_ri.sort_values("rank_std", ascending=True).reset_index(drop=True)
    std_c     = [DISTRICT_COLORS.get(d, "#888888") for d in df_std["district"]]
    bars4     = ax4.barh(df_std["community_name"], df_std["rank_std"],
                         color=std_c, edgecolor="grey", lw=0.3, height=0.72)
    std_max   = df_std["rank_std"].max()
    for bar, val in zip(bars4, df_std["rank_std"]):
        ax4.text(bar.get_width() + std_max * 0.015,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=7.5)
    ax4.set_xlabel("Rank standard deviation (2,000 MC draws)", fontsize=9)
    panel_tag(ax4, "D")
    ax4.grid(axis="x", linestyle=":", alpha=0.4)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.tick_params(labelsize=8)

    dist_handles = [mpatches.Patch(facecolor=DISTRICT_COLORS[d], label=d)
                    for d in DISTRICT_ORDER]
    fig.legend(handles=dist_handles, loc="lower center", ncol=5,
               fontsize=8, title="District", title_fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Step 4: Uncertainty Fig 4 — rank stability stacked bars
# ============================================================
def make_ufig4_rank_stability(out: Path) -> None:
    print("Step 4: uncertainty fig4_recovery_rank_stability…")
    mc_path = TABLES / "recovery_weight_sensitivity_24.csv"
    ri_path = TABLES / "community_recovery_index_24.csv"
    if not mc_path.exists() or not ri_path.exists():
        print("  [skip] CSV not found"); return

    df_ri = load_recovery_index_csv(ri_path)
    df_ri = df_ri.sort_values("modal_rank").reset_index(drop=True)

    df_mc = pd.read_csv(mc_path).set_index("community_name")
    df_mc = df_mc.reindex(df_ri["community_name"])

    n_comm = len(df_ri)
    rank_cols = sorted([c for c in df_mc.columns
                        if c.startswith("rank_") and c.endswith("_pct")],
                       key=lambda x: int(x.split("_")[1]))
    n_ranks = len(rank_cols)
    RANK_CMAP = plt.cm.RdYlGn_r

    fig_h = max(8.0, 0.42 * n_comm + 3.0)
    fig = plt.figure(figsize=(15, fig_h), dpi=DPI)
    gs  = fig.add_gridspec(1, 2, width_ratios=[4, 1],
                           left=0.20, right=0.96,
                           top=0.91, bottom=0.06, wspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_std  = fig.add_subplot(gs[1])

    y_pos = np.arange(n_comm)
    bar_h = 0.75

    ytick_labels = [
        f"{row['community_name']}  (RI={row['recovery_index']:.3f})  [{row.get('district','')[:3]}]"
        for _, row in df_ri.iterrows()
    ]

    bottoms = np.zeros(n_comm)
    for rank_idx, col in enumerate(rank_cols):
        if col not in df_mc.columns: continue
        vals  = df_mc[col].fillna(0).values
        frac  = rank_idx / max(n_ranks - 1, 1)
        color = RANK_CMAP(frac)
        ax_main.barh(y_pos, vals, left=bottoms, height=bar_h,
                     color=color, edgecolor="none")
        for i, val in enumerate(vals):
            if val >= 9:
                ax_main.text(bottoms[i] + val / 2, y_pos[i], f"{val:.0f}%",
                             ha="center", va="center", fontsize=5.0,
                             color="white", fontweight="bold")
        bottoms += vals

    ax_main.set_xlim(0, 100)
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(ytick_labels, fontsize=7)
    ax_main.set_xlabel("Percentage of 2,000 Monte Carlo draws (%)", fontsize=8)
    ax_main.set_title("(a)", fontsize=10, fontweight="bold", loc="left", pad=6)
    ax_main.grid(axis="x", lw=0.4, alpha=0.35)
    ax_main.spines[["top", "right"]].set_visible(False)
    ax_main.tick_params(axis="x", labelsize=7)
    ax_main.tick_params(axis="y", length=0)

    sm = plt.cm.ScalarMappable(cmap=RANK_CMAP,
                               norm=mcolors.Normalize(vmin=1, vmax=n_ranks))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_main, orientation="horizontal",
                      fraction=0.018, pad=0.05, shrink=0.38,
                      anchor=(0.0, 1.0))
    cb.set_label(f"Rank  (1 = fastest · {n_ranks} = slowest)", fontsize=7)
    cb.set_ticks([1, n_ranks // 4, n_ranks // 2, 3 * n_ranks // 4, n_ranks])
    cb.ax.tick_params(labelsize=6)

    # Right panel — rank std
    rank_stds  = df_mc["rank_std"].fillna(0).values
    bar_c_std  = [DISTRICT_COLORS.get(d, "#888888") for d in df_ri["district"]]
    ax_std.barh(y_pos, rank_stds, height=bar_h,
                color=bar_c_std, edgecolor="none")
    std_max = max(rank_stds.max(), 0.5)
    for i, val in enumerate(rank_stds):
        if val >= std_max * 0.22:
            ax_std.text(val - std_max * 0.05, y_pos[i], f"{val:.1f}",
                        va="center", ha="right", fontsize=6,
                        color="white", fontweight="bold")
        else:
            ax_std.text(val + std_max * 0.05, y_pos[i], f"{val:.1f}",
                        va="center", ha="left", fontsize=6, color="#333333")
    ax_std.set_xlim(0, std_max * 1.35)
    ax_std.set_yticks(y_pos); ax_std.set_yticklabels([])
    ax_std.tick_params(axis="y", left=False)
    ax_std.set_xlabel("Rank\nstd dev", fontsize=7.5)
    ax_std.set_title("(b)", fontsize=10, fontweight="bold", loc="left", pad=6)
    ax_std.grid(axis="x", lw=0.4, alpha=0.35)
    ax_std.spines[["top", "right"]].set_visible(False)
    ax_std.tick_params(labelsize=6)

    dist_handles = [mpatches.Patch(facecolor=DISTRICT_COLORS[d], label=d)
                    for d in DISTRICT_ORDER
                    if d in df_ri["district"].values]
    ax_std.legend(handles=dist_handles, fontsize=6, title="District",
                  title_fontsize=6, framealpha=0.85,
                  loc="lower right", bbox_to_anchor=(2.3, 0.0))

    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Step 5: Uncertainty Fig 5 — uncertainty decomposition
# ============================================================
def make_ufig5_decomposition(out: Path) -> None:
    print("Step 5: uncertainty fig5_uncertainty_decomposition…")
    from src.visualization.narrow_paper_figures import uncertainty_decomposition as _ud
    _ud(ROOT, out)
    return
    thr_path = TABLES / "threshold_sensitivity.csv"
    mc_path  = TABLES / "recovery_weight_sensitivity_24.csv"
    ri_path  = TABLES / "community_recovery_index_24.csv"

    if not thr_path.exists() or not mc_path.exists() or not ri_path.exists():
        print("  [skip] one or more required CSVs not found"); return

    df_thr = pd.read_csv(thr_path)
    df_mc  = pd.read_csv(mc_path)
    df_ri  = load_recovery_index_csv(ri_path)

    df_ri_sorted = df_ri.sort_values("modal_rank").reset_index(drop=True)
    comm_order   = df_ri_sorted["community_name"].tolist()
    df_mc_ord    = df_mc.set_index("community_name").reindex(comm_order).reset_index()
    rank_stds    = df_mc_ord["rank_std"].fillna(0).values

    dist_map = df_ri.set_index("community_name")["district"].to_dict()

    THR_REF = 0.50
    direct_path = TABLES / "threshold_exposure_direct.csv"
    if not direct_path.exists():
        print("  [skip] threshold_exposure_direct.csv not found"); return
    dd = pd.read_csv(direct_path).set_index("threshold")
    # Population on the paper's AOI scale (see uncertainty_figures.POP_SCALE)
    dd["population_exposed"] = dd["population_exposed"] * (366077 / 426499)

    # Direct full-scene sampling: percentage range (max - min) across
    # thresholds 0.50-0.95 relative to the value at threshold 0.50. Each
    # metric has its own range; they are no longer identical by construction.
    hp = dd.loc[THR_REF:0.95]
    cols = {"Buildings": "buildings_exposed", "Population": "population_exposed",
            "Roads (km)": "roads_exposed_km", "Flood area\n(km²)": "flood_area_km2"}
    exposure_cv = {k: 100 * (hp[c].max() - hp[c].min()) / dd.loc[THR_REF, c]
                   for k, c in cols.items()}
    abs_ranges = {k: (hp[c].min(), hp[c].max()) for k, c in cols.items()}

    n_comm = len(comm_order)
    fig = plt.figure(figsize=(15, max(6.0, 0.28 * n_comm + 3.5)), dpi=DPI)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 2.8], wspace=0.32,
                           left=0.07, right=0.97, top=0.88, bottom=0.12)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])

    # Left — two uncertainty sources on the same metrics (log axis):
    # (i) probability threshold 0.50-0.95; (ii) SAR water thresholds +/-2 dB
    ld_path = TABLES / "label_definition_exposure.csv"
    metric_names  = list(exposure_cv.keys())
    cv_vals       = list(exposure_cv.values())
    ld_vals = [np.nan] * len(metric_names)
    if ld_path.exists():
        ld = pd.read_csv(ld_path).set_index("definition")
        def span(col):
            v = [ld.loc[k, col] for k in ("primary", "sar_thr_plus2dB", "sar_thr_minus2dB")]
            return 100 * (max(v) - min(v)) / ld.loc["primary", col]
        ld_vals = [span("buildings"), span("population"), span("roads_km"), span("flood_km2")]
    metric_colors = ["#1565C0", "#C62828", "#2E7D32", "#6A1B9A"]
    x = np.arange(len(metric_names)); w = 0.38
    bars_t = ax_l.bar(x - w / 2, cv_vals, w, color=metric_colors, edgecolor="white", linewidth=0.5,
                      label="Probability threshold 0.50-0.95")
    bars_d = ax_l.bar(x + w / 2, ld_vals, w, color=metric_colors, alpha=0.45, hatch="//",
                      edgecolor="white", linewidth=0.5, label="SAR water thresholds \u00b12 dB")
    for bars, vals in ((bars_t, cv_vals), (bars_d, ld_vals)):
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax_l.text(bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:.1f}%",
                          ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax_l.set_yscale("log")
    ax_l.set_xticks(x); ax_l.set_xticklabels(metric_names, fontsize=8)
    ax_l.set_ylabel("Range (% of reference)", fontsize=8)
    ax_l.set_title("(a)", fontsize=10, fontweight="bold", loc="left", pad=6)
    ax_l.set_ylim(1, 400)
    ax_l.tick_params(labelsize=8)
    ax_l.grid(axis="y", lw=0.4, alpha=0.4, which="both")
    ax_l.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    ax_l.spines[["top", "right"]].set_visible(False)

    # Right — rank std horizontal bars
    y_r   = np.arange(n_comm)
    bar_cr = [DISTRICT_COLORS.get(dist_map.get(c, ""), "#888888")
              for c in comm_order]
    ax_r.barh(y_r, rank_stds, color=bar_cr, edgecolor="none", height=0.72)
    std_max = max(rank_stds.max(), 0.1)
    for i, val in enumerate(rank_stds):
        ax_r.text(val + std_max * 0.02, y_r[i], f"{val:.2f}",
                  va="center", ha="left", fontsize=6.5)

    most_stable_idx   = int(np.argmin(rank_stds))
    most_unstable_idx = int(np.argmax(rank_stds))
    ax_r.annotate("Most stable",
                  xy=(rank_stds[most_stable_idx], y_r[most_stable_idx]),
                  xytext=(std_max * 0.5, y_r[most_stable_idx]),
                  fontsize=6.5, color="#2E7D32",
                  arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=0.8))
    ax_r.annotate("Most uncertain",
                  xy=(rank_stds[most_unstable_idx], y_r[most_unstable_idx]),
                  xytext=(std_max * 0.50, y_r[most_unstable_idx] - 2),
                  fontsize=6.5, color="#C62828",
                  arrowprops=dict(arrowstyle="->", color="#C62828", lw=0.8))

    ax_r.set_yticks(y_r)
    ax_r.set_yticklabels(comm_order, fontsize=7.5)
    ax_r.set_xlim(0, std_max * 1.32)
    ax_r.set_xlabel("Rank standard deviation  (Monte Carlo, n = 2,000)", fontsize=8)
    ax_r.set_title("(b)", fontsize=10, fontweight="bold", loc="left", pad=6)
    ax_r.grid(axis="x", lw=0.4, alpha=0.4)
    ax_r.spines[["top", "right"]].set_visible(False)
    ax_r.tick_params(labelsize=7)

    dist_handles = [mpatches.Patch(facecolor=DISTRICT_COLORS[d], label=d)
                    for d in DISTRICT_ORDER
                    if d in (dist_map.get(c) for c in comm_order)]
    ax_r.legend(handles=dist_handles, loc="lower right",
                fontsize=6.5, title="District", title_fontsize=6.5,
                framealpha=0.85)

    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Step 6: Community recovery index map
# ============================================================
def make_recovery_map(out: Path) -> None:
    print("Step 6: community_recovery_index_map…")
    try:
        import geopandas as gpd
    except ImportError:
        print("  [skip] geopandas not available"); return

    ri_path = TABLES / "community_recovery_index_24.csv"
    gpkg    = ROOT / "data/processed/exposure/community_exposure_24.gpkg"
    if not ri_path.exists() or not gpkg.exists():
        print("  [skip] required files not found"); return

    df_ri  = load_recovery_index_csv(ri_path)

    gdf = gpd.read_file(gpkg)
    gdf = gdf.merge(df_ri[["community_name", "recovery_index", "rank",
                             "rank_std", "district"]],
                    on="community_name", how="left")

    fig, ax = plt.subplots(figsize=(9, 11), dpi=DPI)
    fig.patch.set_facecolor("#f5f5f5")
    ax.set_facecolor("#dde8f0")

    cmap_ri = plt.colormaps["RdYlGn"]
    norm_ri = mcolors.Normalize(vmin=0, vmax=1)

    for _, row in gdf.iterrows():
        if pd.isna(row.get("recovery_index")) or pd.isna(row.get("rank")):
            continue
        x, y = row.geometry.x, row.geometry.y
        ri    = float(row["recovery_index"])
        color = cmap_ri(norm_ri(ri))
        ax.plot(x, y, "o", color=color, ms=9, mec="white", mew=0.8, zorder=5)
        ax.annotate(
            f"{row['community_name']}\n(RI={ri:.3f}, R{int(row['rank'])})",
            (x, y), textcoords="offset points", xytext=(7, 4), fontsize=5.5,
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))

    xmin, xmax = gdf.geometry.x.min(), gdf.geometry.x.max()
    ymin, ymax = gdf.geometry.y.min(), gdf.geometry.y.max()
    pad_x = (xmax - xmin) * 0.08
    pad_y = (ymax - ymin) * 0.08
    ax.set_xlim(xmin - pad_x, xmax + pad_x + (xmax - xmin) * 0.20)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_xlabel("Easting (m, UTM 30N)", fontsize=8)
    ax.set_ylabel("Northing (m, UTM 30N)", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.35, alpha=0.35, color="white")

    sm = plt.cm.ScalarMappable(cmap=cmap_ri, norm=norm_ri)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, shrink=0.55)
    cb.set_label("Recovery Index", fontsize=9)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=7)

    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Main
# ============================================================
# ============================================================
# Step 7: DL paper fig1 — study area + spatial block partition
# ============================================================
def make_dl_fig1(out: Path) -> None:
    print("Step 7: dl_paper/fig1_study_area_blocks…")
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.enums import Resampling
    except ImportError:
        print("  [skip] geopandas or rasterio not available"); return
    try:
        from adjustText import adjust_text
        HAVE_ADJUST_TEXT = True
    except ImportError:
        HAVE_ADJUST_TEXT = False

    from src.visualization.dl_figures import (
        get_spatial_splits, block_to_utm, SPLIT_COLORS,
    )

    parquet_path = ROOT / "data/processed/features/model_training_samples.parquet"
    label_path   = ROOT / "data/processed/flood_extent/fused_flood_confidence.tif"
    comm_path    = ROOT / "data/processed/aoi/affected_communities.gpkg"
    aoi_path     = ROOT / "data/processed/aoi/lower_volta_aoi.gpkg"
    gadm0_path   = ROOT / "data/raw/admin/gadm41_GHA.gpkg"

    if not parquet_path.exists() or not label_path.exists():
        print("  [skip] parquet or label raster not found"); return

    # Reconstruct spatial splits using the same shared logic as train_unet.py
    # and dl_figures.py, so this figure's partition can never silently drift
    # from the one actually used for model training/evaluation.
    train_b, val_b, test_b = get_spatial_splits(parquet_path)

    with rasterio.open(label_path) as src:
        H, W  = src.height, src.width
        tfm   = src.transform
        scale = 10
        lbl   = src.read(1, out_shape=(H // scale, W // scale),
                         resampling=Resampling.nearest).astype(float)
    lbl[lbl == 255] = np.nan
    flood_bg = np.where((lbl >= 1) & (lbl <= 3), 1.0, 0.0)   # flood classes only (not permanent water)
    left   = tfm.c
    top    = tfm.f
    right  = tfm.c + W * tfm.a
    bottom = tfm.f + H * tfm.e

    comm_gdf = gpd.read_file(comm_path).to_crs("EPSG:32630")
    try:
        aoi_gdf  = gpd.read_file(aoi_path).to_crs("EPSG:32630")
        has_aoi  = True
    except Exception:
        has_aoi = False

    fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
    ax.imshow(flood_bg, extent=[left, right, bottom, top],
              cmap="Blues", vmin=0, vmax=1, alpha=0.4, origin="upper",
              aspect="auto", interpolation="none")

    for bid in train_b | val_b | test_b:
        br, bc = map(int, bid.split("_"))
        x0, y0, x1, y1 = block_to_utm(br, bc, tfm)
        split = "train" if bid in train_b else ("val" if bid in val_b else "test")
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=0.5, edgecolor="white",
            facecolor=mcolors.to_rgba(SPLIT_COLORS[split], 0.35)))

    # Community markers — colour by district, label all 24
    texts = []
    for _, row in comm_gdf.iterrows():
        x, y  = row.geometry.x, row.geometry.y
        color = DISTRICT_COLORS.get(row.get("district", ""), "#FFD54F")
        ax.plot(x, y, "o", color=color, ms=8, mec="white", mew=0.7, zorder=5)
        texts.append(ax.text(x, y, row.get("community_name", ""),
                     fontsize=9, color="black", zorder=6))

    if HAVE_ADJUST_TEXT:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6),
                    expand=(1.8, 2.0), force_text=(0.8, 1.0),
                    force_static=(0.4, 0.5), lim=2000)

    if has_aoi:
        aoi_gdf.boundary.plot(ax=ax, color="black", linewidth=1.1, linestyle="--", zorder=4)

    legend_handles = [
        mpatches.Patch(facecolor=mcolors.to_rgba(SPLIT_COLORS["train"], 0.6),
                       edgecolor="white", lw=0.5, label=f"Training ({len(train_b)} blocks)"),
        mpatches.Patch(facecolor=mcolors.to_rgba(SPLIT_COLORS["val"], 0.6),
                       edgecolor="white", lw=0.5, label=f"Validation ({len(val_b)} blocks)"),
        mpatches.Patch(facecolor=mcolors.to_rgba(SPLIT_COLORS["test"], 0.6),
                       edgecolor="white", lw=0.5, label=f"Test ({len(test_b)} blocks)"),
        Line2D([0], [0], color="black", lw=1.1, linestyle="--", label="Study area AOI boundary"),
        mpatches.Patch(facecolor=mcolors.to_rgba(_FLOOD, 0.5),
                       edgecolor="none", label="Flood extent (labels)"),
    ] + [mpatches.Patch(facecolor=DISTRICT_COLORS[d], label=d)
         for d in DISTRICT_ORDER if d in comm_gdf["district"].values]

    ax.legend(handles=legend_handles, loc="lower left", fontsize=8.5,
              framealpha=0.9, frameon=True)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.set_xlabel("Easting (km, UTM 30N)", fontsize=11)
    ax.set_ylabel("Northing (km, UTM 30N)", fontsize=11)
    ax.tick_params(labelsize=9)
    panel_b_x = 0.02
    ax.text(panel_b_x, 0.98, "(b)", transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # Ghana context inset — Panel (a)
    if gadm0_path.exists():
        inset_ax = fig.add_axes([0.70, 0.70, 0.22, 0.22])
        gadm0 = gpd.read_file(gadm0_path, layer="ADM_ADM_0").to_crs("EPSG:32630")
        gadm0.plot(ax=inset_ax, facecolor="#EDEDED", edgecolor="black", linewidth=0.6)
        aoi_b2 = (aoi_gdf if has_aoi else comm_gdf).total_bounds
        box_w = (aoi_b2[2] - aoi_b2[0]) * 3
        box_h = (aoi_b2[3] - aoi_b2[1]) * 3
        cx = (aoi_b2[0] + aoi_b2[2]) / 2
        cy = (aoi_b2[1] + aoi_b2[3]) / 2
        inset_ax.add_patch(mpatches.Rectangle(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            facecolor="none", edgecolor="#C62828", linewidth=1.5, zorder=5))
        inset_ax.set_xticks([]); inset_ax.set_yticks([])
        inset_ax.set_title("(a) Ghana", fontsize=8.5, fontweight="bold")
        for spine in inset_ax.spines.values():
            spine.set_edgecolor("#888888")

    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Step 8: Uncertainty paper fig1 — flood extent + communities
# ============================================================
def make_uncertainty_fig1(out: Path) -> None:
    print("Step 8: uncertainty_paper/fig1_study_area_flood_extent…")
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.enums import Resampling
    except ImportError:
        print("  [skip] geopandas or rasterio not available"); return
    try:
        from adjustText import adjust_text
        HAVE_ADJUST_TEXT = True
    except ImportError:
        HAVE_ADJUST_TEXT = False

    THR_REF   = 0.50
    REF_AREA  = float(pd.read_csv(ROOT / "data/processed/flood_extent/fused_flood_stats.csv").set_index("class").loc["total_flooded_all_sources", "area_km2"])
    tif_path  = ROOT / "data/processed/flood_extent/fused_flood_confidence.tif"
    comm_path = ROOT / "data/processed/aoi/affected_communities.gpkg"
    aoi_path  = ROOT / "data/processed/aoi/lower_volta_aoi.gpkg"
    gadm0_path = ROOT / "data/raw/admin/gadm41_GHA.gpkg"

    if not tif_path.exists():
        print("  [skip] fused flood confidence raster not found"); return

    comm_gdf = gpd.read_file(comm_path).to_crs("EPSG:32630")
    aoi_gdf  = gpd.read_file(aoi_path).to_crs("EPSG:32630") if aoi_path.exists() else None

    # The map shows the fused change-based flood extent (classes 1-3, pre-existing water excluded), the label used in the paper,
    # not the raw classifier score (which maps 109 km2 at 0.50). Down-sampled by block maximum so small patches stay visible.
    scale = 8
    with rasterio.open(tif_path) as fsrc:
        cls = fsrc.read(1); left, _, _, top = fsrc.bounds; res = fsrc.res[0]
    H, W = cls.shape; hh, ww = (H // scale) * scale, (W // scale) * scale
    fl = np.isin(cls, [1, 2, 3])[:hh, :ww].reshape(hh // scale, scale, ww // scale, scale).any(axis=(1, 3))
    flood = np.where(fl, 1.0, np.nan)
    right, bottom = left + ww * res, top - hh * res

    fig, ax = plt.subplots(figsize=(11, 13), dpi=DPI)

    flood_rgba = np.zeros((*flood.shape, 4), dtype=np.float32)
    _fr, _fg, _fb = mcolors.to_rgb(_FLOOD)
    flood_rgba[..., 0] = _fr
    flood_rgba[..., 1] = _fg
    flood_rgba[..., 2] = _fb
    flood_rgba[..., 3] = np.where(np.isnan(flood), 0.0, 0.72)
    ax.imshow(flood_rgba, extent=[left, right, bottom, top],
              origin="upper", aspect="auto", interpolation="none")

    if aoi_gdf is not None:
        aoi_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, linestyle="--", zorder=4)

    texts = []
    for _, row in comm_gdf.iterrows():
        x, y  = row.geometry.x, row.geometry.y
        color = DISTRICT_COLORS.get(row.get("district", ""), "#FF6F00")
        ax.plot(x, y, "o", color=color, ms=9, mec="white", mew=0.8, zorder=6)
        texts.append(ax.text(x, y, row.get("community_name", ""),
                     fontsize=10, color="black", zorder=7))

    if HAVE_ADJUST_TEXT:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6),
                    expand=(1.8, 2.0), force_text=(0.8, 1.0),
                    force_static=(0.4, 0.5), lim=2000)

    # View extent: cover full AOI + all communities, not just the raster's
    # own (partial) coverage.
    if aoi_gdf is not None:
        aoi_b = aoi_gdf.total_bounds
        comm_b = comm_gdf.total_bounds
        view_left   = min(left, aoi_b[0], comm_b[0])
        view_bottom = min(bottom, aoi_b[1], comm_b[1])
        view_right  = max(right, aoi_b[2], comm_b[2])
        view_top    = max(top, aoi_b[3], comm_b[3])
        pad_x = (view_right - view_left) * 0.03
        pad_y = (view_top - view_bottom) * 0.03
        view_left -= pad_x; view_right += pad_x
        view_bottom -= pad_y; view_top += pad_y
    else:
        view_left, view_bottom, view_right, view_top = left, bottom, right, top

    ax.set_xlim(view_left, view_right)
    ax.set_ylim(view_bottom, view_top)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}"))
    ax.set_xlabel("Easting (km, UTM 30N)", fontsize=11)
    ax.set_ylabel("Northing (km, UTM 30N)", fontsize=11)
    ax.tick_params(labelsize=9)

    # Scale bar
    bar_len_m = 20000
    bar_x0 = view_left + (view_right - view_left) * 0.04
    bar_y0 = view_bottom + (view_top - view_bottom) * 0.03
    ax.plot([bar_x0, bar_x0 + bar_len_m], [bar_y0, bar_y0], color="black", lw=3, zorder=9)
    ax.text(bar_x0 + bar_len_m / 2, bar_y0 + (view_top - view_bottom) * 0.012, "20 km",
            ha="center", fontsize=9, zorder=9)

    # North arrow
    na_x = view_right - (view_right - view_left) * 0.05
    na_y0 = view_bottom + (view_top - view_bottom) * 0.03
    na_y1 = na_y0 + (view_top - view_bottom) * 0.05
    ax.annotate("", xy=(na_x, na_y1), xytext=(na_x, na_y0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2), zorder=9)
    ax.text(na_x, na_y1 + (view_top - view_bottom) * 0.008, "N", ha="center", fontsize=10,
            fontweight="bold", zorder=9)

    legend_handles = [
        mpatches.Patch(facecolor=_FLOOD, alpha=0.72, edgecolor="none",
                       label=f"Fused flood extent ({REF_AREA:.1f} km²)"),
    ]
    if aoi_gdf is not None:
        legend_handles.append(Line2D([0], [0], color="black", lw=1.0, linestyle="--",
                                     label="Study area AOI boundary"))
    legend_handles += [mpatches.Patch(facecolor=DISTRICT_COLORS[d], label=d)
         for d in DISTRICT_ORDER if d in comm_gdf["district"].values]

    ax.legend(handles=legend_handles, loc="upper left", fontsize=8.5,
              framealpha=0.9, frameon=True)

    # Ghana context inset
    if gadm0_path.exists():
        inset_ax = fig.add_axes([0.68, 0.68, 0.22, 0.22])
        gadm0 = gpd.read_file(gadm0_path, layer="ADM_ADM_0").to_crs("EPSG:32630")
        gadm0.plot(ax=inset_ax, facecolor="#EDEDED", edgecolor="black", linewidth=0.6)
        aoi_b2 = (aoi_gdf if aoi_gdf is not None else comm_gdf).total_bounds
        box_w = (aoi_b2[2] - aoi_b2[0]) * 3
        box_h = (aoi_b2[3] - aoi_b2[1]) * 3
        cx = (aoi_b2[0] + aoi_b2[2]) / 2
        cy = (aoi_b2[1] + aoi_b2[3]) / 2
        inset_ax.add_patch(mpatches.Rectangle(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            facecolor="none", edgecolor="#C62828", linewidth=1.5, zorder=5))
        inset_ax.set_xticks([]); inset_ax.set_yticks([])
        inset_ax.set_title("Ghana", fontsize=8, fontweight="bold")
        for spine in inset_ax.spines.values():
            spine.set_edgecolor("#888888")

    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Step 9: pub_recovery_dashboard — 24-community version
# ============================================================
def make_pub_recovery_dashboard(out: Path) -> None:
    print("Step 9: pub_recovery_dashboard…")
    ri_path   = TABLES / "community_recovery_index_24.csv"
    ndvi_p    = ROOT / "data/processed/recovery/ndvi_recovery.tif"
    water_p   = ROOT / "data/processed/recovery/water_recession.tif"
    comm_path = ROOT / "data/processed/aoi/affected_communities.gpkg"

    if not ri_path.exists():
        print("  [skip] community_recovery_index_24.csv not found"); return

    df_ri = load_recovery_index_csv(ri_path)
    df_ri_sorted = df_ri.sort_values("recovery_index", ascending=True).reset_index(drop=True)

    cmap_ri   = plt.colormaps["RdYlGn"]
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32)

    # Panel A — NDVI recovery raster
    ax1 = fig.add_subplot(gs[0, 0])
    if ndvi_p.exists():
        try:
            import rasterio
            with rasterio.open(ndvi_p) as src:
                arr = src.read(1).astype(float)
                if src.nodata is not None and not np.isnan(src.nodata):
                    arr[arr <= src.nodata] = np.nan
                else:
                    arr[arr < -9000] = np.nan
                bounds = src.bounds
            masked = np.ma.masked_invalid(arr)
            cmap_n = plt.colormaps["RdYlGn"].copy()
            cmap_n.set_bad("lightgrey", 0.3)
            ax1.imshow(masked, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                       cmap=cmap_n, vmin=0, vmax=1, alpha=0.85, origin="upper",
                       aspect="auto", interpolation="bilinear")
            cb1 = fig.colorbar(plt.cm.ScalarMappable(
                cmap=cmap_n, norm=mcolors.Normalize(0, 1)), ax=ax1,
                fraction=0.03, pad=0.02, shrink=0.75)
            cb1.set_label("Recovery ratio", fontsize=7)
            cb1.ax.tick_params(labelsize=6)
        except Exception as e:
            ax1.text(0.5, 0.5, f"Raster unavailable\n{e}", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=8)
    else:
        ax1.text(0.5, 0.5, "NDVI raster\nnot found", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=9)
    panel_tag(ax1, "A")
    ax1.tick_params(labelsize=7)

    # Panel B — Water recession raster
    ax2 = fig.add_subplot(gs[0, 1])
    if water_p.exists():
        try:
            import rasterio
            with rasterio.open(water_p) as src:
                arr2   = src.read(1).astype(float)
                bounds2 = src.bounds
            arr2[~np.isfinite(arr2)] = np.nan
            masked2 = np.ma.masked_invalid(arr2)
            cmap_w  = plt.colormaps["Blues"].copy()
            cmap_w.set_bad("lightgrey", 0.3)
            vmax = float(np.nanpercentile(arr2[np.isfinite(arr2)], 97)) if np.isfinite(arr2).any() else 1.0
            ax2.imshow(masked2, extent=[bounds2.left, bounds2.right, bounds2.bottom, bounds2.top],
                       cmap=cmap_w, vmin=0, vmax=vmax, alpha=0.85, origin="upper",
                       aspect="auto", interpolation="bilinear")
            cb2 = fig.colorbar(plt.cm.ScalarMappable(
                cmap=cmap_w, norm=mcolors.Normalize(0, vmax)), ax=ax2,
                fraction=0.03, pad=0.02, shrink=0.75)
            cb2.set_label("MNDWI recession score", fontsize=7)
            cb2.ax.tick_params(labelsize=6)
        except Exception as e:
            ax2.text(0.5, 0.5, f"Raster unavailable\n{e}", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Water recession raster\nnot found", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=9)
    panel_tag(ax2, "B")
    ax2.tick_params(labelsize=7)

    # Overlay community dots on both raster panels
    try:
        import geopandas as gpd
        comm_gdf = gpd.read_file(comm_path).to_crs("EPSG:32630")
        for ax_r in [ax1, ax2]:
            for _, row in comm_gdf.iterrows():
                x, y = row.geometry.x, row.geometry.y
                ax_r.plot(x, y, "*", color="white", ms=5, mec="black", mew=0.4, zorder=5)
    except Exception:
        pass

    # Panel C — Recovery index bars (24 communities)
    ax3 = fig.add_subplot(gs[1, 0])
    bar_colors = [cmap_ri(v) for v in df_ri_sorted["recovery_index"]]
    dist_colors = [DISTRICT_COLORS.get(d, "#888") for d in df_ri_sorted["district"]]
    bars = ax3.barh(df_ri_sorted["community_name"], df_ri_sorted["recovery_index"],
                    color=bar_colors, edgecolor="grey", lw=0.3, height=0.72)
    for i, (bar, dc) in enumerate(zip(bars, dist_colors)):
        ax3.barh(df_ri_sorted["community_name"].iloc[i], 0.018,
                 color=dc, height=0.72, left=0, zorder=4)
    for bar, val in zip(bars, df_ri_sorted["recovery_index"]):
        ax3.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=7, fontweight="bold")
    ax3.axvline(0.5, color="grey", lw=0.8, linestyle=":", alpha=0.6)
    ax3.set_xlim(0, 1.18)
    ax3.set_xlabel("Recovery Index (0 = least  ·  1 = most)", fontsize=9)
    panel_tag(ax3, "C")
    ax3.grid(axis="x", linestyle=":", alpha=0.4)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(labelsize=7.5)
    sm = plt.cm.ScalarMappable(cmap=cmap_ri, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cb3 = fig.colorbar(sm, ax=ax3, fraction=0.020, pad=0.03, shrink=0.5)
    cb3.set_label("Index value", fontsize=7)
    cb3.set_ticks([0, 0.5, 1]); cb3.set_ticklabels(["Low", "Mid", "High"])
    cb3.ax.tick_params(labelsize=6)

    # Panel D — 24×4 normalised component heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    df_heat   = df_ri.sort_values("recovery_index", ascending=False).reset_index(drop=True)
    comp_cols = ["ndvi_norm", "wrec_norm", "racc_norm", "setl_norm"]
    comp_lbls = ["Veg\nrecovery", "Water\nrecession", "Road\naccess", "Settlement\nproxy"]
    heat_data = df_heat[comp_cols].values
    im = ax4.imshow(heat_data, aspect="auto", cmap="RdYlGn",
                    vmin=0, vmax=1, interpolation="nearest")
    ax4.set_xticks(range(4)); ax4.set_xticklabels(comp_lbls, fontsize=8)
    ax4.set_yticks(range(len(df_heat)))
    ax4.set_yticklabels(
        [f"{c}  [{d[:2]}]" for c, d in zip(df_heat["community_name"], df_heat["district"])],
        fontsize=7)
    panel_tag(ax4, "D")
    for i in range(len(df_heat)):
        for j in range(4):
            v = heat_data[i, j]
            ax4.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                     color="white" if (v < 0.35 or v > 0.75) else "#222222")
    cb4 = fig.colorbar(im, ax=ax4, fraction=0.020, pad=0.03, shrink=0.5)
    cb4.set_label("Normalised score", fontsize=7)
    cb4.ax.tick_params(labelsize=6)

    dist_handles = [mpatches.Patch(facecolor=DISTRICT_COLORS[d], label=d)
                    for d in DISTRICT_ORDER]
    fig.legend(handles=dist_handles, loc="lower center", ncol=6,
               fontsize=7.5, title="District", title_fontsize=7.5,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Step 10: pub_exposure_summary — 24-community version
# ============================================================
def make_pub_exposure_summary(out: Path) -> None:
    print("Step 10: pub_exposure_summary…")
    cs_path = TABLES / "community_exposure_summary_24.csv"
    if not cs_path.exists():
        print("  [skip] community_exposure_summary_24.csv not found"); return

    cs = pd.read_csv(cs_path)
    cs_fa  = cs.sort_values("flood_area_km2",   ascending=False).reset_index(drop=True)
    cs_pop = cs.sort_values("total_pop",         ascending=False).reset_index(drop=True)
    cs_bld = cs.sort_values("total_buildings",   ascending=False).reset_index(drop=True)
    cs_rd  = cs.sort_values("flooded_road_km",   ascending=False).reset_index(drop=True)

    def dc(df): return [DISTRICT_COLORS.get(d, "#888") for d in df["district"]]

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.36)

    def hbar(ax, df, col, xlabel, letter, fmt="{:.1f}"):
        colors = dc(df)
        vals   = df[col].fillna(0)
        bars   = ax.barh(df["community_name"], vals, color=colors, height=0.68)
        vmax   = vals.max()
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + vmax * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va="center", fontsize=7.5)
        ax.set_xlim(0, vmax * 1.22)
        ax.set_xlabel(xlabel, fontsize=9)
        panel_tag(ax, letter)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    hbar(ax=fig.add_subplot(gs[0, 0]), df=cs_fa,
         col="flood_area_km2",
         xlabel="Flood area in service buffer (km²)",
         letter="A")

    hbar(ax=fig.add_subplot(gs[0, 1]), df=cs_pop,
         col="total_pop",
         xlabel="Buffer population (WorldPop 2020)",
         letter="B",
         fmt="{:,.0f}")

    hbar(ax=fig.add_subplot(gs[1, 0]), df=cs_bld,
         col="total_buildings",
         xlabel="OSM buildings in service buffer (#)",
         letter="C",
         fmt="{:,.0f}")

    ax_rd = fig.add_subplot(gs[1, 1])
    cs_rdf = cs.dropna(subset=["flooded_road_km"]).sort_values(
        "flooded_road_km", ascending=False).reset_index(drop=True)
    if len(cs_rdf) > 0:
        hbar(ax=ax_rd, df=cs_rdf,
             col="flooded_road_km",
             xlabel="Flooded road length in buffer (km)",
             letter="D")
    else:
        cs_rd2 = cs.sort_values("total_road_km", ascending=False).reset_index(drop=True)
        hbar(ax=ax_rd, df=cs_rd2,
             col="total_road_km",
             xlabel="Total road length in service buffer (km)",
             letter="D")

    dist_handles = [mpatches.Patch(facecolor=DISTRICT_COLORS[d], label=d)
                    for d in DISTRICT_ORDER]
    fig.legend(handles=dist_handles, loc="lower center", ncol=6,
               fontsize=8, title="District", title_fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    generate_mc_rank_frequencies()
    make_fig4_exposure(FIGS / "fig4_exposure_hotspots.png")
    make_fig7_recovery(FIGS / "fig7_recovery_analysis.png")
    make_ufig4_rank_stability(UFIGS / "fig4_recovery_rank_stability.png")
    make_ufig5_decomposition(UFIGS / "fig5_uncertainty_decomposition.png")
    make_recovery_map(MAPS / "community_recovery_index_map.png")
    make_dl_fig1(FIGS / "dl_paper" / "fig1_study_area_blocks.png")
    make_uncertainty_fig1(UFIGS / "fig1_study_area_flood_extent.png")
    make_pub_recovery_dashboard(MAPS / "pub_recovery_dashboard.png")
    make_pub_exposure_summary(MAPS / "pub_exposure_summary.png")
    print("\nDone — all 24-community figures regenerated.")
