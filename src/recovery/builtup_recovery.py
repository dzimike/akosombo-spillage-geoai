from __future__ import annotations
"""
Phase 8 – Task 8.3: Built-Up and Settlement Recovery Proxy

Estimates recovery of built-up areas using NDBI trajectory,
water persistence around settlements, and night-time light (NTL) proxies.

Outputs
-------
data/processed/recovery/settlement_recovery_proxy.csv
"""

import argparse
import logging
import geopandas as gpd
import pandas as pd
import numpy as np

from src.utils.paths import load_config, get_processed_path, ensure_dir
from src.utils.logging_utils import log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def compute_settlement_water_persistence(
    community_buffers: gpd.GeoDataFrame,
    water_persistence_raster: str,
) -> pd.DataFrame:
    """
    For each community buffer, compute mean water persistence score
    from the water persistence raster.
    """
    # TODO: implement zonal statistics using rasterstats or rioxarray
    raise NotImplementedError("Zonal statistics not yet implemented")


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    ensure_dir(get_processed_path("recovery", config))
    log.warning("Settlement recovery proxy requires water persistence raster. Complete ndwi_recovery.py first.")
    # TODO: compute NDBI recovery, NTL recovery if available, water persistence per community


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Built-up and settlement recovery proxy")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
