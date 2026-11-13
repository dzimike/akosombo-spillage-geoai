from __future__ import annotations
"""Overlay of the fused change-based extent on the Charter TerraSAR-X (16 Oct 2023) flooded area, map frame on our grid."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from matplotlib.colors import to_rgb

from src.utils.paths import get_project_root
from src.visualization.palette import AGREE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    root = get_project_root()
    t = rasterio.open(root / "data/processed/validation/charter_terrasarx_20231016.tif").read(1)
    fu = rasterio.open(root / "data/processed/flood_extent/fused_flood_confidence.tif").read(1)
    ins = t != 255
    rows = np.where(ins.any(1))[0]; cols = np.where(ins.any(0))[0]
    sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    k = 3
    tt, ff = t[sl], fu[sl]
    H, W = (tt.shape[0] // k) * k, (tt.shape[1] // k) * k
    red = lambda m: m[:H, :W].reshape(H // k, k, W // k, k).any(axis=(1, 3))
    m, o, w, p4, v = red(tt == 1), red(np.isin(ff, [1, 2, 3])), red(tt == 2), red(ff == 4), red(tt != 255)
    img = np.ones((H // k, W // k, 3))
    img[v] = to_rgb(AGREE["none"]); img[p4 & v] = to_rgb(AGREE["pre"]); img[w] = (0.72, 0.78, 0.85)
    img[o & ~m] = to_rgb(AGREE["ours"]); img[m & ~o] = to_rgb(AGREE["tsx"]); img[m & o] = to_rgb(AGREE["both"])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(img, extent=[0, W * 10 / 1000, 0, H * 10 / 1000], origin="upper"); ax.set_xlabel("km"); ax.set_ylabel("km")
    ax.legend(handles=[mp.Patch(color=AGREE["both"], label="Both"), mp.Patch(color=AGREE["ours"], label="Our extent only"),
                       mp.Patch(color=AGREE["tsx"], label="TerraSAR-X only"), mp.Patch(color=(0.72, 0.78, 0.85), label="Map water body (JRC)"),
                       mp.Patch(color=AGREE["pre"], label="Our pre-existing water")], loc="lower left", fontsize=8)
    fig.tight_layout()
    out = root / "outputs/figures/fig_terrasarx_agreement.png"
    fig.savefig(out, dpi=600)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
