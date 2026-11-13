"""
Time-series plotting utilities for NDVI, water extent, and recovery trajectories.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def plot_ndvi_timeseries(
    dates: list,
    ndvi_values: list[float],
    community_name: str,
    event_start: str,
    event_end: str,
    output_path: str,
) -> None:
    """Plot NDVI time series with event window shaded."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, ndvi_values, marker="o", color="#2E7D32", linewidth=1.5, markersize=4)
    ax.axvspan(
        pd.to_datetime(event_start), pd.to_datetime(event_end),
        alpha=0.2, color="red", label="Spillage event window"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean NDVI")
    ax.set_title(f"NDVI Time Series – {community_name}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=8)
    plt.xticks(rotation=45)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("NDVI time series saved: %s", output_path)


def plot_flood_area_timeseries(
    dates: list,
    flood_areas_km2: list[float],
    output_path: str,
) -> None:
    """Plot detected flood area over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dates, flood_areas_km2, color="#1565C0", alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Detected flood area (km²)")
    ax.set_title("Temporal Flood Extent – Lower Volta Basin")
    plt.xticks(rotation=45)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Flood area time series saved: %s", output_path)
