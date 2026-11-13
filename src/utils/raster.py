"""
Raster I/O and alignment utilities.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


def read_band(path: str, band: int = 1) -> tuple[np.ndarray, dict]:
    """Read a single raster band and return (array, profile)."""
    with rasterio.open(path) as src:
        data = src.read(band)
        profile = src.profile.copy()
    return data, profile


def reproject_raster(src_path: str, dst_path: str, dst_crs: str = "EPSG:32630") -> None:
    """Reproject raster to dst_crs and write to dst_path."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )


def read_aligned(src_path: str, template_path: str, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """
    Read src_path raster reprojected and resampled in-memory to match template_path.

    Returns a 2-D numpy array (single band) aligned to the template grid.
    NaN is used for nodata.
    """
    with rasterio.open(template_path) as tmpl:
        dst_crs = tmpl.crs
        dst_transform = tmpl.transform
        dst_width = tmpl.width
        dst_height = tmpl.height

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        dst_arr = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_width=dst_width,
            dst_height=dst_height,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
    return dst_arr


def align_raster(src_path: str, template_path: str, dst_path: str) -> None:
    """Align src raster to the grid, resolution, and CRS of a template raster."""
    with rasterio.open(template_path) as template:
        profile = template.profile.copy()
        dst_transform = template.transform
        dst_crs = template.crs
        dst_width = template.width
        dst_height = template.height

    with rasterio.open(src_path) as src:
        profile.update(count=src.count, dtype=src.dtypes[0])
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_width=dst_width,
                    dst_height=dst_height,
                    resampling=Resampling.bilinear,
                )
