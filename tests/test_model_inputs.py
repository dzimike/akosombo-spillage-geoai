"""
Model input validation tests.

Checks the training sample table for data quality issues that would
compromise model training or create spatial leakage.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.paths import get_project_root

ROOT = get_project_root()
SAMPLE_PATH = ROOT / "data/processed/features/model_training_samples.parquet"
REQUIRED_FEATURES = [
    "elevation", "slope", "twi", "dist_to_volta",
    "ndvi_pre", "sar_vv_change", "population_density",
]
TARGET_COL = "flood_label"
VALID_LABELS = {0, 1}


@pytest.fixture
def sample_df():
    if not SAMPLE_PATH.exists():
        pytest.skip("Training samples not yet built")
    return pd.read_parquet(SAMPLE_PATH)


def test_target_label_present(sample_df):
    assert TARGET_COL in sample_df.columns, f"Target column '{TARGET_COL}' missing"


def test_valid_class_labels(sample_df):
    unique = set(sample_df[TARGET_COL].dropna().unique())
    unexpected = unique - VALID_LABELS
    assert not unexpected, f"Unexpected class labels: {unexpected}"


def test_no_missing_target(sample_df):
    null_count = sample_df[TARGET_COL].isna().sum()
    assert null_count == 0, f"{null_count} missing values in '{TARGET_COL}'"


def test_required_features_present(sample_df):
    missing = [f for f in REQUIRED_FEATURES if f not in sample_df.columns]
    assert not missing, f"Missing required features: {missing}"


def test_no_infinite_values(sample_df):
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(sample_df[col]).sum()
        assert inf_count == 0, f"Infinite values found in column '{col}'"


def test_spatial_block_ids_present(sample_df):
    """Spatial block IDs (added from pixel coordinates at training time) are required for leakage-free cross-validation."""
    from src.models.train_random_forest import add_spatial_blocks
    blocks = add_spatial_blocks(sample_df, 1000)
    assert blocks["block_id"].notna().all(), "block_id must not contain null values"
    assert blocks["block_id"].nunique() > 1, "samples must span more than one spatial block"


def test_minimum_sample_size(sample_df):
    """At least 300 samples required per CLAUDE.md validation protocol."""
    assert len(sample_df) >= 300, f"Only {len(sample_df)} samples; minimum 300 required"


def test_class_balance_not_extreme(sample_df):
    """Flood class should represent at least 5% of samples to ensure training signal."""
    flood_frac = (sample_df[TARGET_COL] == 1).mean()
    assert flood_frac >= 0.05, f"Flood class is only {100*flood_frac:.1f}% of samples – extreme imbalance"
