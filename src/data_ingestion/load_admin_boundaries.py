from __future__ import annotations
"""
Phase 2: Administrative Boundary Loader

Loads Ghana national, regional, and district boundaries from a GADM 4.1
single-file GeoPackage (gadm41_GHA.gpkg) containing layers:
  ADM_ADM_0  – national boundary (1 feature)
  ADM_ADM_1  – regions (16 features)
  ADM_ADM_2  – districts (260 features)

Place gadm41_GHA.gpkg in data/raw/admin/ before running.

Outputs
-------
data/raw/admin/ghana_national.gpkg
data/raw/admin/ghana_regions.gpkg
data/raw/admin/ghana_districts.gpkg
data/raw/admin/study_area_districts.gpkg   ← flood-affected districts only
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd

from src.utils.paths import load_config, get_project_root, get_data_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Districts spanning the flood-affected Lower Volta study area
STUDY_DISTRICTS = [
    # Volta Region – Tongu districts
    "North Tongu",
    "Central Tongu",
    "South Tongu",
    # Greater Accra Region – Ada districts (near the river mouth)
    "Ada East",
    "Ada West",
    # Volta Region coastal / lower Volta
    "Keta Municipal",
    "Ketu South",
    "Ketu North",
    "Anloga",
]

GADM_GPKG = "gadm41_GHA.gpkg"
LAYER_NATIONAL = "ADM_ADM_0"
LAYER_REGIONS  = "ADM_ADM_1"
LAYER_DISTRICTS = "ADM_ADM_2"


def load_gadm_layer(gpkg_path: Path, layer: str) -> gpd.GeoDataFrame:
    """Load a layer from the GADM single-file GeoPackage."""
    gdf = gpd.read_file(str(gpkg_path), layer=layer)
    log.info("Loaded layer '%s': %d features", layer, len(gdf))
    return gdf


def save_layer(gdf: gpd.GeoDataFrame, path: Path) -> None:
    gdf.to_file(str(path), driver="GPKG")
    log.info("Saved: %s (%d features)", path.name, len(gdf))


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    admin_dir = Path(get_data_path("admin", config))
    ensure_dir(admin_dir)

    gpkg_path = admin_dir / GADM_GPKG
    if not gpkg_path.exists():
        log.error(
            "GADM file not found: %s\n"
            "Download from https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_GHA.gpkg\n"
            "and place it in data/raw/admin/",
            gpkg_path,
        )
        return

    with Timer() as t:
        # Load all three admin levels
        national  = load_gadm_layer(gpkg_path, LAYER_NATIONAL)
        regions   = load_gadm_layer(gpkg_path, LAYER_REGIONS)
        districts = load_gadm_layer(gpkg_path, LAYER_DISTRICTS)

        # Save full Ghana layers
        save_layer(national,  admin_dir / "ghana_national.gpkg")
        save_layer(regions,   admin_dir / "ghana_regions.gpkg")
        save_layer(districts, admin_dir / "ghana_districts.gpkg")

        # Extract study-area districts
        study = districts[districts["NAME_2"].isin(STUDY_DISTRICTS)].copy()
        if len(study) == 0:
            log.warning("No study districts matched — check STUDY_DISTRICTS names")
        else:
            save_layer(study, admin_dir / "study_area_districts.gpkg")
            log.info(
                "Study districts extracted: %s",
                ", ".join(sorted(study["NAME_2"].tolist())),
            )

        # Log region coverage
        for region in sorted(study["NAME_1"].unique()):
            n = (study["NAME_1"] == region).sum()
            log.info("  %s: %d districts", region, n)

    outputs = [
        str(admin_dir / "ghana_national.gpkg"),
        str(admin_dir / "ghana_regions.gpkg"),
        str(admin_dir / "ghana_districts.gpkg"),
        str(admin_dir / "study_area_districts.gpkg"),
    ]
    log_run(
        script=__file__,
        input_paths=[str(gpkg_path)],
        output_paths=outputs,
        parameters={"study_districts": STUDY_DISTRICTS},
        status="success",
        runtime_seconds=t.elapsed,
    )
    log.info("Admin boundaries loaded in %.1f s", t.elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Ghana admin boundaries from GADM 4.1")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
