"""
Pytest configuration for akosombo-spillage-geoai test suite.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so src.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
