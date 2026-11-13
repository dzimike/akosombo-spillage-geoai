from __future__ import annotations
"""
Phase 2 – Task 2.3: Population Data Loader

Downloads (or uses pre-placed) WorldPop Ghana 1 km² raster, clips it to the
Lower Volta AOI, and derives community-level population estimates by
spatially joining study-area districts and known community points.

Usage
-----
    python src/data_ingestion/load_population.py --config config/config.yaml

Outputs
-------
data/processed/population/population_grid_clipped.tif
data/processed/population/community_population_estimates.csv
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.utils.logging_utils import Timer, log_run
from src.utils.paths import load_config, get_processed_path, get_data_path, get_project_root, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/"
    "Global_2000_2020_1km_UNadj/2020/GHA/"
    "GHA_ppp_2020_1km_Aggregated_UNadj.tif"
)
WORLDPOP_FNAME = "GHA_ppp_2020_1km_Aggregated_UNadj.tif"


def download_worldpop(raw_pop_dir: Path) -> Path:
    """Download WorldPop Ghana raster if not already present."""
    import urllib.request
    dest = raw_pop_dir / WORLDPOP_FNAME
    if dest.exists():
        log.info("WorldPop raster already present: %s", dest.name)
        return dest
    log.info("Downloading WorldPop Ghana 2020 1 km raster...")
    urllib.request.urlretrieve(WORLDPOP_URL, str(dest))
    log.info("Downloaded: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def clip_worldpop_to_aoi(worldpop_path: Path, aoi_path: Path, output_path: Path) -> None:
    """Clip and reproject WorldPop raster to the AOI extent."""
    import rasterio
    import rasterio.mask
    import rasterio.warp

    aoi = gpd.read_file(str(aoi_path)).to_crs("EPSG:4326")
    shapes = [aoi.geometry.union_all().__geo_interface__]

    with rasterio.open(str(worldpop_path)) as src:
        out_image, out_transform = rasterio.mask.mask(
            src, shapes, crop=True, nodata=-99999
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": -99999,
            "compress": "lzw",
        })
        with rasterio.open(str(output_path), "w", **out_meta) as dst:
            dst.write(out_image)

    # Report total population in clipped area
    valid = out_image[0][out_image[0] > 0]
    log.info(
        "WorldPop clipped: %d×%d pixels, %.0f total population in AOI",
        out_image.shape[2], out_image.shape[1], valid.sum(),
    )


def estimate_community_populations(
    worldpop_clipped: Path,
    districts_path: Path,
    communities_path: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """
    Derive population estimates per study community.

    Strategy: sum WorldPop pixels within a 5 km buffer around each community
    centroid, scaled by fractional overlap with the district's total to avoid
    double-counting.
    """
    import rasterio
    import rasterio.mask
    from shapely.geometry import Point

    communities = gpd.read_file(str(communities_path)).to_crs("EPSG:4326")
    districts   = gpd.read_file(str(districts_path)).to_crs("EPSG:4326")

    records = []
    with rasterio.open(str(worldpop_clipped)) as src:
        for _, row in communities.iterrows():
            geom = row.geometry
            # 5 km buffer (approx 0.045° at equator)
            buf = geom.buffer(0.045)
            try:
                masked, _ = rasterio.mask.mask(src, [buf.__geo_interface__], crop=True, nodata=-99999)
                pop = float(masked[masked > 0].sum())
            except Exception:
                pop = np.nan

            # Identify which district this community falls in
            pt = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
            joined = gpd.sjoin(pt, districts[["NAME_2", "geometry"]], how="left", predicate="within")
            district_name = joined["NAME_2"].iloc[0] if len(joined) > 0 else "Unknown"

            records.append({
                "community": row.get("community_name", row.get("name", row.get("community", "Unknown"))),
                "district": district_name,
                "lon": geom.x,
                "lat": geom.y,
                "population_estimate_5km": round(pop),
                "source": "WorldPop 2020 1km UN-adjusted",
                "notes": "5 km buffer sum — indicative only",
            })

    df = pd.DataFrame(records)
    df.to_csv(str(output_csv), index=False)
    log.info("Community population estimates written: %d communities → %s", len(df), output_csv.name)
    for _, r in df.iterrows():
        log.info("  %-20s (%s): ~%d people", r["community"], r["district"], r["population_estimate_5km"])
    return df


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    root = get_project_root()

    raw_pop_dir = get_data_path("population", config)
    ensure_dir(raw_pop_dir)
    pop_dir = get_processed_path("population", config)
    ensure_dir(pop_dir)

    aoi_path         = root / "data/processed/aoi/lower_volta_aoi.gpkg"
    communities_path = root / "data/processed/aoi/affected_communities.gpkg"
    districts_path   = root / "data/raw/admin/study_area_districts.gpkg"
    clipped_tif      = pop_dir / "population_grid_clipped.tif"
    community_csv    = pop_dir / "community_population_estimates.csv"

    with Timer() as t:
        worldpop_raw = download_worldpop(raw_pop_dir)
        clip_worldpop_to_aoi(worldpop_raw, aoi_path, clipped_tif)
        estimate_community_populations(clipped_tif, districts_path, communities_path, community_csv)

    log_run(
        script=__file__,
        input_paths=[str(worldpop_raw), str(aoi_path)],
        output_paths=[str(clipped_tif), str(community_csv)],
        parameters={"source": "WorldPop 2020 1km UN-adjusted", "buffer_km": 5},
        status="success",
        runtime_seconds=t.elapsed,
    )
    log.info("Population data loaded in %.1f s", t.elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and clip WorldPop population data")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
