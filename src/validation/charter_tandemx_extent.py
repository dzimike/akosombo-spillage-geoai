from __future__ import annotations
"""
Independent flood extent from International Charter activation 844, University of Santa Maria (UFSM) map
'Flooded areas in the Lower Volta River, Ghana - situation on 20 October 2023' (TanDEM-X 20 Oct 2023, change
against Sentinel-1 4 and 11 Oct; analyst product, not field-validated).

The JPEG (data/raw/validation/charter844/img_28561528.jpg, 1323 x 930 px, ~65 m/px) carries a lon/lat graticule of
'+' crosses every 5 arc-minutes. The crosses are located automatically, an affine lon/lat -> pixel transform is
fitted by least squares, the red 'Flooded areas' and blue 'Permanent water bodies' fills are classified from the
RGB, and the result is resampled (nearest) onto the project 10 m grid inside the map's area-of-interest polygon.

Outputs: data/processed/validation/charter_tandemx_20231020.tif   (1 flooded, 2 permanent water, 0 other, 255 outside AOI)
         outputs/tables/charter_tandemx_agreement.csv             (agreement of the change-based and previous extents)
"""
import logging

import numpy as np
import pandas as pd
import rasterio
from matplotlib.path import Path as MPath
from PIL import Image
from pyproj import Transformer

from src.utils.paths import get_project_root, get_output_path

Image.MAX_IMAGE_PIXELS = None
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCALE = 1323 / 2000.0                       # coordinates below were read off a 2000-px-wide rendering
LON = {0.1667: 204, 0.25: 415, 0.3333: 627, 0.4167: 838, 0.5: 1049, 0.5833: 1260, 0.6667: 1472, 0.75: 1683, 0.8333: 1894}
LAT = {6.0833: 299, 6.0: 511, 5.9167: 722, 5.8333: 933}
# area-of-interest polygon drawn on the map (same rendering)
AOI = [(75, 165), (700, 165), (1853, 589), (1908, 1077), (1240, 1084), (1046, 743), (583, 478), (75, 350)]
LEGEND = (75, 668, 735, 1100)               # x0, y0, x1, y1 of the legend panel overlapping the frame (rendering px)


def find_crosses(gray: np.ndarray):
    pts = []
    for lon, xd in LON.items():
        for lat, yd in LAT.items():
            x, y = int(round(xd * SCALE)), int(round(yd * SCALE))
            r = 14
            win = gray[y - r:y + r + 1, x - r:x + r + 1] < 110
            if win.sum() < 25:
                continue
            ry = int(np.argmax(win.sum(axis=1))); cx = int(np.argmax(win.sum(axis=0)))
            if win.sum(axis=1).max() < 15 or win.sum(axis=0).max() < 15:
                continue
            pts.append((lon, lat, x - r + cx, y - r + ry))
    return np.array(pts)


def main() -> None:
    root = get_project_root()
    im = np.array(Image.open(root / "data/raw/validation/charter844/img_28561528.jpg").convert("RGB")).astype(int)
    gray = im.mean(axis=2)
    pts = find_crosses(gray)
    log.info("graticule crosses located: %d", len(pts))
    A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    cx, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    cy, *_ = np.linalg.lstsq(A, pts[:, 3], rcond=None)
    res = np.hypot(A @ cx - pts[:, 2], A @ cy - pts[:, 3])
    m_per_px = 111320 * np.cos(np.radians(6.0)) / abs(cx[0]) * 1.0
    log.info("affine fit residual: mean %.2f px, max %.2f px (~%.0f m/px)", res.mean(), res.max(), m_per_px)

    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    red = (r > 200) & (g < 70) & (b < 70)
    blue = (b > 200) & (r > 120) & (r < 200) & (g > 170) & (g < 225)
    cls = np.zeros(red.shape, np.uint8); cls[blue] = 2; cls[red] = 1
    ys, xs = np.mgrid[0:cls.shape[0], 0:cls.shape[1]]
    poly = MPath(np.array(AOI) * SCALE)
    inside = poly.contains_points(np.c_[xs.ravel(), ys.ravel()]).reshape(xs.shape)
    lx0, ly0, lx1, ly1 = [v * SCALE for v in LEGEND]
    inside &= ~((xs >= lx0) & (xs <= lx1) & (ys >= ly0) & (ys <= ly1) & ~poly.contains_points(
        np.c_[xs.ravel(), ys.ravel()]).reshape(xs.shape))
    cls[~inside] = 255
    log.info("AOI pixels %d; flooded %d px (%.1f km2); permanent water %d px", inside.sum(), (cls == 1).sum(),
             (cls == 1).sum() * (m_per_px / 1000) ** 2, (cls == 2).sum())

    # inverse: for every 10 m grid cell find the map pixel
    with rasterio.open(root / "data/processed/satellite/s1_pre_event_vv.tif") as s:
        prof, tr, crs = s.profile.copy(), s.transform, s.crs
        H, W = s.height, s.width
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    # px = cx.[lon, lat, 1]; invert the 2x2 part
    M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]]); Mi = np.linalg.inv(M); off = np.array([cx[2], cy[2]])
    out = np.full((H, W), 255, np.uint8)
    cols = np.arange(W)
    for r0 in range(0, H, 500):
        rows = np.arange(r0, min(r0 + 500, H))
        X = tr.c + (cols[None, :] + 0.5) * tr.a + (rows[:, None] + 0.5) * tr.b
        Y = tr.f + (cols[None, :] + 0.5) * tr.d + (rows[:, None] + 0.5) * tr.e
        lon, lat = inv.transform(X, Y)
        px = M[0, 0] * lon + M[0, 1] * lat + off[0]
        py = M[1, 0] * lon + M[1, 1] * lat + off[1]
        ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
        ok = (ix >= 0) & (ix < cls.shape[1]) & (iy >= 0) & (iy < cls.shape[0])
        blk = np.full(ix.shape, 255, np.uint8)
        blk[ok] = cls[iy[ok], ix[ok]]
        out[rows] = blk
    prof.update(dtype="uint8", count=1, nodata=255, compress="lzw")
    p = root / "data/processed/validation/charter_tandemx_20231020.tif"
    p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(p, "w", **prof) as dst:
        dst.write(out, 1)
    log.info("written %s: flooded %.1f km2 on our grid; AOI %.0f km2", p, (out == 1).sum() / 1e4, (out != 255).sum() / 1e4)


if __name__ == "__main__":
    main()
