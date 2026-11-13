from __future__ import annotations
"""
Phase 7 – Task 7.5: Community Accessibility Graph Model (Optional)

Constructs a settlement-road-river network graph to estimate isolation risk
and accessibility disruption during and after the flood event.

Only implement if constructing a community network analysis is justified
by the research question and data quality.

Outputs
-------
outputs/models/community_graph_model.pt (if GNN used)
outputs/tables/accessibility_disruption_metrics.csv
outputs/maps/community_isolation_risk.gpkg
"""

import argparse
import logging

import geopandas as gpd
import networkx as nx
import pandas as pd

from src.utils.paths import load_config, get_output_path, ensure_dir
from src.utils.logging_utils import log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_community_graph(
    roads: gpd.GeoDataFrame,
    communities: gpd.GeoDataFrame,
    bridges: gpd.GeoDataFrame | None = None,
    health_facilities: gpd.GeoDataFrame | None = None,
) -> nx.Graph:
    """
    Build a road network graph with community and facility nodes.

    TODO: implement using OSMnx graph with community nodes added.
    """
    raise NotImplementedError("Graph model not yet implemented")


def identify_isolated_communities(
    graph: nx.Graph,
    flooded_road_ids: list,
    community_nodes: list,
) -> list[str]:
    """
    Return list of community names disconnected from the main road network
    after removing flooded road edges.
    """
    G = graph.copy()
    G.remove_edges_from(flooded_road_ids)
    # TODO: find communities with no path to a major hub node
    return []


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    ensure_dir(get_output_path("tables", config))
    log.warning("Graph model is optional. Implement after baseline pipeline is complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Community accessibility graph model")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
