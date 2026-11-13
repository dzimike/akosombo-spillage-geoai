from __future__ import annotations
"""
Sidecar metadata writer for raster outputs.

Every raster output should have an accompanying .metadata.yaml file
documenting provenance, CRS, resolution, and limitations.
"""

from datetime import datetime
from pathlib import Path

import yaml


def write_raster_metadata(
    output_path: str | Path,
    title: str,
    description: str,
    source_data: list[str],
    processing_script: str,
    crs: str,
    resolution: str,
    units: str,
    method: str,
    limitations: str,
) -> Path:
    """Write a .metadata.yaml sidecar next to the raster file."""
    meta_path = Path(str(output_path) + ".metadata.yaml")
    record = {
        "title": title,
        "description": description,
        "date_created": datetime.utcnow().isoformat(),
        "source_data": source_data,
        "processing_script": processing_script,
        "crs": crs,
        "resolution": resolution,
        "units": units,
        "method": method,
        "limitations": limitations,
    }
    with open(meta_path, "w") as f:
        yaml.dump(record, f, default_flow_style=False, sort_keys=False)
    return meta_path
