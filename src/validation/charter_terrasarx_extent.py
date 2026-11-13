from __future__ import annotations
"""
Independent flood extents from two International Charter (activation 844) maps based on the TerraSAR-X acquisition of
16 October 2023, digitised from the published JPEGs:

A  UN-SPIDER 'Flooded Area: Ghana, Upper Section of Lower Volta River. Situation on 16 October 2023'
   (img_28547888.jpg, 10,952 x 6,612 px, ~3 m/px; legend 'Potentially Flooded Area' dark blue and JRC 'Water Body'
   light blue; TerraSAR-X, DLR/Airbus; analyst product, not field-validated; UTM 31N frame).
B  UFSM 'Flooded areas in the Lower Volta River, Situation on 16 October 2023' (img_28547948.jpg, 1,164 x 820 px,
   TerraSAR-X 16 Oct with Sentinel-1 4 Oct; red flooded areas, light-blue permanent water), a sub-area of A used to
   cross-check A.

Georeferencing of A. The frame is a UTM 31N grid-aligned rectangle with thick tick marks every 6 arc-minutes of
longitude (top and bottom edges) and one latitude tick (6 deg 6' N) on the left and right edges. Tick positions are
detected automatically; a UTM-31N axis-aligned transform (E0, N0, metres per pixel) is fitted by least squares to
the 3 + 3 longitude and 2 latitude tick constraints (pyproj). B is registered to A, not read from its
own graticule: its crosses are 2.2 km apart (1.2 arc-minutes) and its longitude labels (0 deg 8', 10', 11', 12' E) appear
to be rounded, so they cannot be used. B is a sub-area of the same analysis, so a similarity transform (isotropic
scale, translation) from B pixels to A pixels was found by maximising the agreement of the flooded and water classes
(Dice 0.85; 10.05 m per B pixel), then carried to UTM 31N through A's tick fit.

Outputs: data/processed/validation/charter_terrasarx_20231016.tif  (A on the project grid: 1 flooded, 2 water, 0 other,
         255 outside the map frame), charter_terrasarx_ufsm_20231016.tif (B, same coding)
"""
import logging

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from scipy.optimize import least_squares

from src.utils.paths import get_project_root

Image.MAX_IMAGE_PIXELS = None
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# frame and tick positions of map A in original pixels (found by scanning the JPEG: frame lines and 26-px tick blocks)
A_LEFT, A_RIGHT, A_TOP, A_BOTTOM = 241.5, 8803.0, 722.0, 6035.0
A_TOP_TICKS = {0.1: 833.0, 0.2: 4466.0, 0.3: 8098.5}          # lon (deg) -> x
A_BOTTOM_TICKS = {0.1: 804.5, 0.2: 4438.5, 0.3: 8072.0}
A_LAT_TICK = 6.1                                                # 6 deg 6' N
A_LEFT_Y, A_RIGHT_Y = 2656.5, 2701.0


def classify_a(rgb: np.ndarray) -> np.ndarray:
    """1 flooded (dark blue 90,122,183), 2 JRC water (79,175,226), 0 other."""
    d_f = np.linalg.norm(rgb - np.array([90, 122, 183]), axis=-1)
    d_w = np.linalg.norm(rgb - np.array([79, 175, 226]), axis=-1)
    cls = np.zeros(rgb.shape[:2], np.uint8)
    cls[(d_w < 28) & (d_w <= d_f)] = 2
    cls[(d_f < 32) & (d_f < d_w)] = 1
    return cls


def fit_a():
    t = Transformer.from_crs(4326, 32631, always_xy=True)
    inv = Transformer.from_crs(32631, 4326, always_xy=True)

    def resid(p):
        e0, n0, s = p
        r = []
        for edge_y, ticks in ((A_TOP, A_TOP_TICKS), (A_BOTTOM, A_BOTTOM_TICKS)):
            for lon, x in ticks.items():
                lo, _ = inv.transform(e0 + s * x, n0 - s * edge_y)
                r.append((lo - lon) * 111320 * np.cos(np.radians(6.1)))        # metres
        for x, y in ((A_LEFT, A_LEFT_Y), (A_RIGHT, A_RIGHT_Y)):
            _, la = inv.transform(e0 + s * x, n0 - s * y)
            r.append((la - A_LAT_TICK) * 110600)
        return r
    e_guess, n_guess = t.transform(0.1, 6.1)
    p0 = [e_guess - 3.04 * A_TOP_TICKS[0.1], n_guess + 3.04 * A_LEFT_Y, 3.04]
    sol = least_squares(resid, p0)
    log.info("A fit: E0 %.1f N0 %.1f pixel %.3f m; residuals (m) %s", *sol.x, np.round(sol.fun, 1))
    return sol.x


def to_grid(cls: np.ndarray, forward, ref_path, frame) -> np.ndarray:
    """Resample (nearest) the map classes onto the project grid; forward(X, Y in UTM30N) -> map px (x, y)."""
    with rasterio.open(ref_path) as s:
        tr, H, W = s.transform, s.height, s.width
    out = np.full((H, W), 255, np.uint8)
    x0, x1, y0, y1 = frame
    cols = np.arange(W)
    for r0 in range(0, H, 500):
        rows = np.arange(r0, min(r0 + 500, H))
        X = tr.c + (cols[None, :] + 0.5) * tr.a + (rows[:, None] + 0.5) * tr.b
        Y = tr.f + (cols[None, :] + 0.5) * tr.d + (rows[:, None] + 0.5) * tr.e
        px, py = forward(X, Y)
        ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
        ok = (ix > x0) & (ix < x1) & (iy > y0) & (iy < y1)
        blk = np.full(ix.shape, 255, np.uint8)
        blk[ok] = cls[iy[ok], ix[ok]]
        out[rows] = blk
    return out


def digitise_a(root):
    e0, n0, s = fit_a()
    im = np.array(Image.open(root / "data/raw/validation/charter844/img_28547888.jpg").convert("RGB")).astype(int)
    cls = classify_a(im)
    log.info("A: flooded %d px (%.1f km2), water %d px (%.1f km2)", (cls == 1).sum(), (cls == 1).sum() * s * s / 1e6,
             (cls == 2).sum(), (cls == 2).sum() * s * s / 1e6)
    tf = Transformer.from_crs(32630, 32631, always_xy=True)

    def forward(X, Y):
        E, N = tf.transform(X, Y)
        return (E - e0) / s, (n0 - N) / s
    frame = (A_LEFT + 10, A_RIGHT - 10, A_TOP + 10, A_BOTTOM - 10)
    return cls, forward, frame, s


B_FRAME = (45, 768, 100, 790)           # x0, x1, y0, y1 of the map interior of B (pixels)


def classify_b(rgb: np.ndarray) -> np.ndarray:
    cls = np.zeros(rgb.shape[:2], np.uint8)
    cls[(rgb[..., 0] > 100) & (rgb[..., 0] < 175) & (rgb[..., 1] > 170) & (rgb[..., 1] < 225) & (rgb[..., 2] > 225)] = 2
    cls[(rgb[..., 0] > 200) & (rgb[..., 1] < 70) & (rgb[..., 2] < 70)] = 1
    return cls


def register_b_to_a(a_cls: np.ndarray, b_cls: np.ndarray, s_a: float):
    """Similarity transform B px -> A px: A = (ax + (x - x0) k, ay + (y - y0) k), k = mpp / s_a. Returns mpp, ax, ay, dice."""
    from scipy.optimize import minimize
    x0, x1, y0, y1 = B_FRAME
    Y, X = np.mgrid[y0:y1:2, x0:x1:2]
    bb = b_cls[Y, X]
    H, W = a_cls.shape

    def neg_dice(p):
        mpp, ax, ay = p
        k = mpp / s_a
        xa = np.rint(ax + (X - x0) * k).astype(int); ya = np.rint(ay + (Y - y0) * k).astype(int)
        ok = (xa >= 0) & (xa < W) & (ya >= 0) & (ya < H)
        c = a_cls[ya[ok], xa[ok]]; b = bb[ok]
        tp = ((c == b) & (b > 0)).sum()
        return -2 * tp / ((b > 0).sum() + (c > 0).sum() + 1e-9)
    # coarse seed from the scene geometry (B covers the western part of A; ~10 m per pixel)
    best = (0, None)
    for mpp in np.arange(9.6, 10.8, 0.1):
        for ax in range(1900, 2500, 24):
            for ay in range(1200, 1700, 24):
                d = -neg_dice((mpp, ax, ay))
                if d > best[0]:
                    best = (d, (mpp, ax, ay))
    p0 = np.array(best[1], float)
    r = minimize(neg_dice, p0, method="Nelder-Mead",
                 options={"xatol": 0.5, "fatol": 1e-4, "initial_simplex": p0 + np.array([[0, 0, 0], [0.15, 0, 0], [0, 12, 0], [0, 0, 12]])})
    return r.x[0], r.x[1], r.x[2], -r.fun


def write(out, ref, path):
    with rasterio.open(ref) as src:
        prof = src.profile.copy()
    prof.update(dtype="uint8", count=1, nodata=255, compress="lzw")
    with rasterio.open(path, "w", **prof) as d:
        d.write(out, 1)
    log.info("written %s: frame area on our grid %.0f km2, flooded %.1f km2, water %.1f km2", path,
             (out != 255).sum() / 1e4, (out == 1).sum() / 1e4, (out == 2).sum() / 1e4)


def main() -> None:
    root = get_project_root()
    ref = root / "data/processed/satellite/s1_pre_event_vv.tif"
    cls, fwd, frame, s = digitise_a(root)
    out = to_grid(cls, fwd, ref, frame)
    write(out, ref, root / "data/processed/validation/charter_terrasarx_20231016.tif")

    # B (UFSM sub-map): register to A, then carry to the project grid
    b = np.array(Image.open(root / "data/raw/validation/charter844/img_28547948.jpg").convert("RGB")).astype(int)
    bcls = classify_b(b)
    mpp, ax, ay, dice = register_b_to_a(cls, bcls, s)
    log.info("B registered to A: %.2f m per pixel, A offset (%.0f, %.0f) px, Dice %.3f (flooded+water)", mpp, ax, ay, dice)
    k = mpp / s
    x0, x1, y0, y1 = B_FRAME

    def forward_b(X, Y):
        xa, ya = fwd(X, Y)                       # A pixels
        return x0 + (xa - ax) / k, y0 + (ya - ay) / k
    outb = to_grid(bcls, forward_b, ref, (x0, x1, y0, y1))
    write(outb, ref, root / "data/processed/validation/charter_terrasarx_ufsm_20231016.tif")
    # consistency of B with A inside B's frame (on A's own pixels)
    Y, X = np.mgrid[y0:y1:1, x0:x1:1]
    xa = np.rint(ax + (X - x0) * k).astype(int); ya = np.rint(ay + (Y - y0) * k).astype(int)
    ca = cls[ya, xa]; cb = bcls[Y, X]
    for c_, n in ((1, "flooded"), (2, "water")):
        tp = ((ca == c_) & (cb == c_)).sum()
        log.info("B vs A (%s): B %.2f km2, A %.2f km2, Dice %.3f", n, (cb == c_).sum() * (mpp / 1000) ** 2,
                 (ca == c_).sum() * (mpp / 1000) ** 2, 2 * tp / ((ca == c_).sum() + (cb == c_).sum()))


if __name__ == "__main__":
    main()
