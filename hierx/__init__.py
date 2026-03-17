"""
hierx: Hierarchical cost and interaction matrices for spatial networks.

This package provides efficient methods for computing and storing cost matrices
and interaction matrices for large-scale spatial networks using hierarchical
grouping strategies.

Main classes:
- Hierarchy: Hierarchical cost matrix representation
- InteractionHierarchy: Hierarchical interaction matrix as linear operator

Utility functions:
- power_law_interaction, exponential_interaction: Distance-decay functions
- generate_grid_network, generate_random_spatial_network: Network generation
- compute_dense_cost_matrix, compute_dense_interaction_matrix: Dense baselines

For details, see HIERARCHIES_CONCEPTUAL_DOCUMENTATION.md
"""

from hierx.backends import HAS_GRAPH_TOOL
from hierx.hierarchy import Hierarchy
from hierx.interaction import InteractionHierarchy
from hierx.storage import (
    HAS_H5PY,
    StorageError,
    load_hierarchy,
    load_interaction,
    save_hierarchy,
    save_interaction,
)
from hierx.utils import (
    compute_dense_cost_matrix,
    compute_dense_interaction_matrix,
    exponential_interaction,
    generate_grid_network,
    generate_large_spatial_network,
    generate_random_spatial_network,
    generate_transport_network,
    power_law_interaction,
)

__version__ = "0.1.1"

__all__ = [
    "HAS_GRAPH_TOOL",
    "HAS_H5PY",
    "Hierarchy",
    "InteractionHierarchy",
    "StorageError",
    "save_hierarchy",
    "load_hierarchy",
    "save_interaction",
    "load_interaction",
    "power_law_interaction",
    "exponential_interaction",
    "generate_grid_network",
    "generate_random_spatial_network",
    "generate_large_spatial_network",
    "generate_transport_network",
    "compute_dense_cost_matrix",
    "compute_dense_interaction_matrix",
]
