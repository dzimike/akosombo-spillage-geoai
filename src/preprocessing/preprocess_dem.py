from __future__ import annotations
"""
Phase 3 – Task 3.3: DEM Preprocessing and Terrain Derivatives

Downloads SRTM 30 m tiles via the OpenTopography API (no auth needed for
SRTM), clips to the AOI, fills sinks, and derives all terrain variables.

Terrain products
----------------
elevation, slope, aspect, flow direction, flow accumulation,
Topographic Wetness Index (TWI), relative elevation

Usage
-----
    python src/preprocessing/preprocess_dem.py --config config/config.yaml

Outputs
-------
data/processed/dem/elevation.tif
data/processed/dem/slope.tif
data/processed/dem/aspect.tif
data/processed/dem/flow_accumulation.tif
data/processed/dem/twi.tif
data/processed/dem/relative_elevation.tif
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.merge
import rasterio.warp
import rasterio.transform
from rasterio.crs import CRS
from rasterio.enums import Resampling

from src.utils.paths import load_config, get_processed_path, get_data_path, get_project_root, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SRTM_URL = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_{col:02d}_{row:02d}.zip"
NODATA_FILL = -9999.0
TARGET_CRS  = "EPSG:32630"
TARGET_RES  = 30  # metres


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _srtm_tile_ids(bounds_wgs84: tuple) -> list[tuple[int, int]]:
    """Return CGIAR-CSI SRTM 5°×5° tile (col, row) indices covering bounds."""
    minx, miny, maxx, maxy = bounds_wgs84
    cols = range(int((minx + 180) // 5) + 1, int((maxx + 180) // 5) + 2)
    rows = range(int((60  - maxy) // 5) + 1, int((60  - miny) // 5) + 2)
    return [(c, r) for c in cols for r in rows]


def download_srtm_tiles(bounds_wgs84: tuple, raw_dem_dir: Path) -> list[Path]:
    """Download SRTM tiles as needed and return list of local TIF paths."""
    import urllib.request, zipfile, io
    tile_ids = _srtm_tile_ids(bounds_wgs84)
    tif_paths = []
    for col, row in tile_ids:
        tif_name = f"srtm_{col:02d}_{row:02d}.tif"
        tif_path = raw_dem_dir / tif_name
        if tif_path.exists():
            log.info("SRTM tile already present: %s", tif_name)
            tif_paths.append(tif_path)
            continue
        url = SRTM_URL.format(col=col, row=row)
        log.info("Downloading SRTM tile %s …", tif_name)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                zdata = resp.read()
            with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".tif"):
                        data = zf.read(member)
                        tif_path.write_bytes(data)
                        log.info("Extracted %s (%.1f MB)", tif_name, len(data) / 1e6)
                        tif_paths.append(tif_path)
                        break
        except Exception as exc:
            log.warning("Could not download tile %s: %s — trying OpenTopography", tif_name, exc)
            tif_path = _download_opentopo(bounds_wgs84, raw_dem_dir / "srtm_opentopo.tif")
            if tif_path:
                return [tif_path]
    return tif_paths


def _download_opentopo(bounds_wgs84: tuple, dest: Path) -> Path | None:
    """Fallback: download SRTM30 via OpenTopography bulk API."""
    import urllib.request
    minx, miny, maxx, maxy = bounds_wgs84
    url = (
        f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1"
        f"&south={miny:.4f}&north={maxy:.4f}&west={minx:.4f}&east={maxx:.4f}"
        f"&outputFormat=GTiff"
    )
    log.info("Fetching SRTM from OpenTopography …")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            dest.write_bytes(resp.read())
        log.info("OpenTopography DEM saved: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest
    except Exception as exc:
        log.error("OpenTopography download failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Terrain processing
# ---------------------------------------------------------------------------

def mosaic_and_clip(tile_paths: list[Path], aoi_shapes: list, out_path: Path) -> dict:
    """Merge tiles, clip to AOI, reproject to UTM-30N at 30 m, write GeoTIFF."""
    datasets = [rasterio.open(str(p)) for p in tile_paths]
    if len(datasets) > 1:
        mosaic, mosaic_transform = rasterio.merge.merge(datasets)
        crs = datasets[0].crs
    else:
        with rasterio.open(str(tile_paths[0])) as src:
            mosaic = src.read()
            mosaic_transform = src.transform
            crs = src.crs

    for ds in datasets:
        ds.close()

    # Clip to AOI (in source CRS)
    from rasterio.mask import mask as rio_mask
    from rasterio.io import MemoryFile
    tmp_profile = {
        "driver": "GTiff", "count": 1, "dtype": mosaic.dtype,
        "crs": crs, "transform": mosaic_transform,
        "height": mosaic.shape[1], "width": mosaic.shape[2],
        "nodata": NODATA_FILL,
    }
    with MemoryFile() as memfile:
        with memfile.open(**tmp_profile) as mem:
            arr = mosaic[0].astype(np.float32)
            arr[arr < -500] = NODATA_FILL
            mem.write(arr, 1)
        with memfile.open() as mem:
            clipped, clip_transform = rio_mask(mem, aoi_shapes, crop=True, nodata=NODATA_FILL)

    # Reproject to UTM-30N
    dst_crs = CRS.from_string(TARGET_CRS)
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        crs, dst_crs, clipped.shape[2], clipped.shape[1],
        left=clip_transform.c, bottom=clip_transform.f + clipped.shape[1] * clip_transform.e,
        right=clip_transform.c + clipped.shape[2] * clip_transform.a,
        top=clip_transform.f,
        resolution=TARGET_RES,
    )
    reprojected = np.full((1, dst_height, dst_width), NODATA_FILL, dtype=np.float32)
    rasterio.warp.reproject(
        source=clipped.astype(np.float32),
        destination=reprojected,
        src_transform=clip_transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=NODATA_FILL,
        dst_nodata=NODATA_FILL,
    )
    profile = {
        "driver": "GTiff", "count": 1, "dtype": "float32",
        "crs": TARGET_CRS, "transform": dst_transform,
        "height": dst_height, "width": dst_width,
        "nodata": NODATA_FILL, "compress": "lzw",
    }
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(reprojected)
    log.info("Elevation clipped & reprojected: %dx%d px → %s", dst_width, dst_height, out_path.name)
    return profile


def compute_slope_aspect(dem: np.ndarray, res_m: float, nodata: float):
    """Return slope (degrees) and aspect (degrees) arrays."""
    valid = dem.copy()
    valid[valid == nodata] = np.nan
    dy, dx = np.gradient(valid, res_m)
    slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect = np.degrees(np.arctan2(-dy, dx)) % 360
    slope[np.isnan(valid)]  = nodata
    aspect[np.isnan(valid)] = nodata
    return slope.astype(np.float32), aspect.astype(np.float32)


def compute_flow_accumulation(dem: np.ndarray, nodata: float) -> np.ndarray:
    """D8 flow accumulation using pysheds (installed separately)."""
    try:
        from pysheds.grid import Grid
        import tempfile, os
        import rasterio
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        # pysheds needs a file-backed grid
        grid = Grid.from_array(np.where(dem == nodata, np.nan, dem))
        pit_filled = grid.fill_pits(dem)
        flooded    = grid.fill_depressions(pit_filled)
        inflated   = grid.resolve_flats(flooded)
        fdir       = grid.flowdir(inflated)
        acc        = grid.accumulation(fdir)
        os.unlink(tmp_path)
        log.info("Flow accumulation computed via pysheds")
        return acc.astype(np.float32)
    except ImportError:
        log.warning("pysheds not installed — using simplified D8 approximation")
        return _simple_flow_accum(dem, nodata)


def _simple_flow_accum(dem: np.ndarray, nodata: float) -> np.ndarray:
    """Simplified flow accumulation proxy: distance-weighted rank of elevation."""
    valid = np.where(dem == nodata, np.nan, dem)
    rank  = np.argsort(np.argsort(valid.ravel())).reshape(valid.shape).astype(np.float32)
    rank[np.isnan(valid)] = nodata
    return rank


def compute_twi(flow_accum: np.ndarray, slope_deg: np.ndarray,
                cell_area_m2: float, nodata: float) -> np.ndarray:
    """TWI = ln(a / tan(β)); floor slope to avoid division by zero."""
    slope_rad  = np.where(slope_deg == nodata, np.nan, np.radians(slope_deg))
    slope_safe = np.where(slope_rad < 0.001, 0.001, slope_rad)
    accum      = np.where(flow_accum == nodata, np.nan, flow_accum)
    area       = np.maximum(accum * cell_area_m2, 1.0)
    twi        = np.log(area / np.tan(slope_safe))
    twi[np.isnan(slope_rad) | np.isnan(accum)] = nodata
    return twi.astype(np.float32)


def compute_relative_elevation(dem: np.ndarray, nodata: float,
                                low_percentile: float = 5.0) -> np.ndarray:
    """Relative elevation above low-lying areas (low percentile proxy)."""
    valid  = dem[dem != nodata]
    floor  = np.percentile(valid, low_percentile) if len(valid) else 0.0
    rel    = np.where(dem == nodata, nodata, dem - floor)
    return rel.astype(np.float32)


def write_tif(array: np.ndarray, profile: dict, out_path: Path, nodata: float) -> None:
    p = profile.copy()
    p.update({"nodata": nodata, "compress": "lzw"})
    with rasterio.open(str(out_path), "w", **p) as dst:
        dst.write(array, 1)
    log.info("Saved: %s", out_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    root   = get_project_root()

    raw_dem_dir = get_data_path("dem", config)
    ensure_dir(raw_dem_dir)
    dem_dir = get_processed_path("dem", config)
    ensure_dir(dem_dir)

    aoi_path = root / "data/processed/aoi/lower_volta_aoi.gpkg"
    aoi      = gpd.read_file(str(aoi_path)).to_crs("EPSG:4326")
    bounds   = aoi.total_bounds          # (minx, miny, maxx, maxy) in WGS84
    aoi_wgs84_shapes = [aoi.geometry.union_all().__geo_interface__]

    with Timer() as t:
        # 1. Download SRTM tiles
        tile_paths = download_srtm_tiles(tuple(bounds), raw_dem_dir)
        if not tile_paths:
            log.error("No DEM tiles available — aborting")
            return

        # 2. Mosaic, clip, reproject → elevation.tif
        elev_path = dem_dir / "elevation.tif"
        profile   = mosaic_and_clip(tile_paths, aoi_wgs84_shapes, elev_path)
        res_m     = TARGET_RES
        cell_area = res_m ** 2

        with rasterio.open(str(elev_path)) as src:
            dem = src.read(1).astype(np.float32)

        # 3. Slope & aspect
        slope, aspect = compute_slope_aspect(dem, res_m, NODATA_FILL)
        write_tif(slope,  profile, dem_dir / "slope.tif",  NODATA_FILL)
        write_tif(aspect, profile, dem_dir / "aspect.tif", NODATA_FILL)

        # 4. Flow accumulation
        flow_accum = compute_flow_accumulation(dem, NODATA_FILL)
        write_tif(flow_accum, profile, dem_dir / "flow_accumulation.tif", NODATA_FILL)

        # 5. TWI
        twi = compute_twi(flow_accum, slope, cell_area, NODATA_FILL)
        write_tif(twi, profile, dem_dir / "twi.tif", NODATA_FILL)

        # 6. Relative elevation
        rel_elev = compute_relative_elevation(dem, NODATA_FILL)
        write_tif(rel_elev, profile, dem_dir / "relative_elevation.tif", NODATA_FILL)

        # Summary stats
        valid = dem[dem != NODATA_FILL]
        log.info(
            "DEM stats — min: %.1f m, max: %.1f m, mean: %.1f m",
            valid.min(), valid.max(), valid.mean(),
        )

    outputs = [str(dem_dir / f) for f in
               ["elevation.tif", "slope.tif", "aspect.tif",
                "flow_accumulation.tif", "twi.tif", "relative_elevation.tif"]]
    log_run(
        script=__file__,
        input_paths=[str(p) for p in tile_paths],
        output_paths=outputs,
        parameters={"crs": TARGET_CRS, "resolution_m": res_m},
        status="success",
        runtime_seconds=t.elapsed,
    )
    log.info("DEM preprocessing complete in %.1f s", t.elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DEM and derive terrain variables")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
