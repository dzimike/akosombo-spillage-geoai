from __future__ import annotations
"""
Path resolution utilities for the akosombo-spillage-geoai project.

All paths are resolved relative to the project root so scripts work
regardless of where they are invoked.
"""

from pathlib import Path
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | None = None) -> dict:
    """Load and return the main project configuration."""
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_data_path(subpath: str, config: dict | None = None) -> Path:
    """Resolve a path under data/ using the config or project root."""
    root = get_project_root()
    if config:
        return root / config["paths"]["raw"] / subpath
    return root / "data" / subpath


def get_processed_path(subpath: str, config: dict | None = None) -> Path:
    """Resolve a path under data/processed/."""
    root = get_project_root()
    base = config["paths"]["processed"] if config else "data/processed"
    return root / base / subpath


def get_output_path(subpath: str, config: dict | None = None) -> Path:
    """Resolve a path under outputs/."""
    root = get_project_root()
    base = config["paths"]["outputs"] if config else "outputs"
    return root / base / subpath


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist, then return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
