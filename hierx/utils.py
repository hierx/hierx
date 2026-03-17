"""
Utility functions for hierarchical interaction matrices.

This module provides helper functions for:
- Distance-to-interaction transformation (power law, exponential)
- Network generation (grid, random spatial)
- Dense matrix computation (for comparison and validation)
"""

from typing import Callable

import networkx as nx
import numpy as np


def power_law_interaction(
    cost: float, a: float = 1.0, b: float = -2.0, offset: float = 5000.0
) -> float:
    """
    Power law distance-decay function.

    Interaction decreases as a power of cost plus offset.

    Parameters
    ----------
    cost : float
        Travel cost (time in minutes or distance in meters)
    a : float, default=1.0
        Scaling factor
    b : float, default=-2.0
        Power exponent (should be negative for decay)
    offset : float, default=5000.0
        Offset to avoid singularity at zero cost

    Returns
    -------
    float
        Interaction value (dimensionless)

    Examples
    --------
    >>> power_law_interaction(10, a=1.0, b=-1.0, offset=5)
    0.0667  # ≈ (5 + 10)^(-1)
    """
    if np.isinf(cost):
        return 0.0
    return a * (offset + cost) ** b


def exponential_interaction(cost: float, a: float = 1e-3) -> float:
    """
    Exponential distance-decay function.

    Interaction decreases exponentially with cost.

    Parameters
    ----------
    cost : float
        Travel cost (time in minutes or distance in meters)
    a : float, default=1e-3
        Decay rate parameter

    Returns
    -------
    float
        Interaction value (dimensionless)

    Examples
    --------
    >>> exponential_interaction(1000, a=0.001)
    0.368  # ≈ exp(-0.001 × 1000)
    """
    if a < 0:
        raise ValueError(f"decay rate a must be non-negative, got {a!r}")
    if np.isinf(cost):
        return 0.0
    return np.exp(-a * cost)


def generate_grid_network(n_x: int, n_y: int, spacing: float = 1000.0) -> nx.Graph:
    """
    Generate a regular grid network with spatial coordinates.

    Creates a grid graph where nodes have 'x' and 'y' attributes for position,
    and edges have 'cost' attributes representing Euclidean distance.

    Parameters
    ----------
    n_x : int
        Number of nodes in x direction
    n_y : int
        Number of nodes in y direction
    spacing : float, default=1000.0
        Distance between adjacent nodes (meters)

    Returns
    -------
    nx.Graph
        Grid network with n_x × n_y nodes
        Node attributes: 'x', 'y' (coordinates)
        Edge attributes: 'cost' (travel time/distance)

    Examples
    --------
    >>> G = generate_grid_network(10, 10, spacing=1000)
    >>> len(G.nodes())
    100
    >>> len(G.edges())
    180  # (10-1)*10*2
    """
    if not isinstance(n_x, int) or n_x < 1:
        raise ValueError(f"n_x must be a positive integer, got {n_x!r}")
    if not isinstance(n_y, int) or n_y < 1:
        raise ValueError(f"n_y must be a positive integer, got {n_y!r}")
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing!r}")
    G = nx.grid_2d_graph(n_x, n_y)

    # Relabel nodes to integers
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # Add spatial coordinates
    node_list = list(mapping.keys())
    for node_id, (i, j) in zip(mapping.values(), node_list):
        G.nodes[node_id]["x"] = i * spacing
        G.nodes[node_id]["y"] = j * spacing

    # Add edge costs (Euclidean distance)
    for u, v in G.edges():
        x_u, y_u = G.nodes[u]["x"], G.nodes[u]["y"]
        x_v, y_v = G.nodes[v]["x"], G.nodes[v]["y"]
        cost = np.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2)
        G[u][v]["cost"] = cost

    return G


def generate_random_spatial_network(
    n_zones: int, area_size: float = 100000.0, connection_radius: float = 15000.0
) -> nx.Graph:
    """
    Generate a random spatial network with zones distributed uniformly.

    Nodes are placed randomly in a square area and connected if within
    a threshold distance (geometric random graph).

    Parameters
    ----------
    n_zones : int
        Number of zones (nodes)
    area_size : float, default=100000.0
        Side length of square area (meters)
    connection_radius : float, default=15000.0
        Maximum distance for edge creation (meters)

    Returns
    -------
    nx.Graph
        Random spatial network
        Node attributes: 'x', 'y' (coordinates)
        Edge attributes: 'cost' (Euclidean distance)

    Examples
    --------
    >>> G = generate_random_spatial_network(100, area_size=50000)
    >>> len(G.nodes())
    100
    """
    if not isinstance(n_zones, int) or n_zones < 1:
        raise ValueError(f"n_zones must be a positive integer, got {n_zones!r}")
    if area_size <= 0:
        raise ValueError(f"area_size must be positive, got {area_size!r}")
    if connection_radius <= 0:
        raise ValueError(f"connection_radius must be positive, got {connection_radius!r}")
    # Generate random positions
    rng = np.random.RandomState(42)
    positions = rng.uniform(0, area_size, size=(n_zones, 2))

    # Create graph
    G = nx.Graph()
    for i in range(n_zones):
        G.add_node(i, x=positions[i, 0], y=positions[i, 1])

    # Add edges within connection radius
    for i in range(n_zones):
        for j in range(i + 1, n_zones):
            x_i, y_i = positions[i]
            x_j, y_j = positions[j]
            cost = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
            if cost <= connection_radius:
                G.add_edge(i, j, cost=cost)

    # Ensure connectivity by adding edges to largest component
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        # Connect components to largest one
        largest = max(components, key=len)
        for component in components:
            if component != largest:
                # Connect closest nodes
                node_comp = next(iter(component))
                node_largest = min(
                    largest,
                    key=lambda n: np.sqrt(
                        (G.nodes[n]["x"] - G.nodes[node_comp]["x"]) ** 2
                        + (G.nodes[n]["y"] - G.nodes[node_comp]["y"]) ** 2
                    ),
                )
                x_c = G.nodes[node_comp]["x"]
                y_c = G.nodes[node_comp]["y"]
                x_l = G.nodes[node_largest]["x"]
                y_l = G.nodes[node_largest]["y"]
                cost = np.sqrt((x_c - x_l) ** 2 + (y_c - y_l) ** 2)
                G.add_edge(node_comp, node_largest, cost=cost)

    return G


def compute_dense_cost_matrix(
    network: nx.Graph,
    zones: list[int] | None = None,
    cutoff: float | None = None,
) -> np.ndarray:
    """
    Compute the full dense matrix of all-pairs shortest path costs.

    This is the baseline O(n²) approach that the hierarchical method approximates.
    Uses scipy.sparse.csgraph.dijkstra (C implementation) for performance.

    Parameters
    ----------
    network : nx.Graph
        Network with 'cost' edge attribute
    zones : list[int], optional
        List of zone IDs to include. If None, uses all nodes.
    cutoff : float, optional
        Maximum shortest-path distance. Entries beyond this are np.inf.

    Returns
    -------
    np.ndarray
        Dense cost matrix of shape (n, n)
        Entry [i, j] is shortest path cost from zone i to zone j

    Notes
    -----
    Computational complexity: O(n² log n) using Dijkstra
    Memory: O(n²)
    """
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

    from hierx.backends import convert_nx_to_csr

    if zones is None:
        zones = list(network.nodes())

    csr, _idx_to_node, node_to_idx = convert_nx_to_csr(network)
    zone_indices = [node_to_idx[z] for z in zones]

    kwargs: dict = {"directed": False, "indices": zone_indices}
    if cutoff is not None:
        kwargs["limit"] = cutoff
    dist = scipy_dijkstra(csr, **kwargs)

    return dist[:, zone_indices]


def compute_dense_interaction_matrix(
    cost_matrix: np.ndarray, interaction_fn: Callable[[float], float]
) -> np.ndarray:
    """
    Transform cost matrix to interaction matrix using distance-decay function.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Dense cost matrix (n × n)
    interaction_fn : Callable[[float], float]
        Function mapping cost to interaction value

    Returns
    -------
    np.ndarray
        Dense interaction matrix (n × n)
        Entry [i, j] = interaction_fn(cost_matrix[i, j])

    Examples
    --------
    >>> costs = np.array([[0, 10, 20], [10, 0, 15], [20, 15, 0]])
    >>> interaction_fn = lambda c: (c + 5)**(-1)
    >>> interactions = compute_dense_interaction_matrix(costs, interaction_fn)
    """
    vectorized_fn = np.vectorize(interaction_fn)
    return vectorized_fn(cost_matrix)


def generate_large_spatial_network(
    n_zones: int,
    area_size: float = 100000.0,
    connection_radius: float | None = None,
    seed: int = 42,
) -> nx.Graph:
    """
    Generate a large random spatial network using KDTree for O(n log n) edge creation.

    Unlike ``generate_random_spatial_network``, this avoids an O(n^2) pairwise
    distance check by using ``scipy.spatial.KDTree`` for neighbour queries.

    Parameters
    ----------
    n_zones : int
        Number of zones (nodes).
    area_size : float, default=100000.0
        Side length of the square area (meters).
    connection_radius : float | None
        Maximum distance for edge creation.  If *None*, defaults to
        ``area_size / sqrt(n_zones) * 3`` which keeps average degree roughly
        constant as the network grows.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        Connected spatial network.
        Node attributes: 'x', 'y' (coordinates).
        Edge attributes: 'cost' (Euclidean distance).
    """
    if not isinstance(n_zones, int) or n_zones < 1:
        raise ValueError(f"n_zones must be a positive integer, got {n_zones!r}")
    if area_size <= 0:
        raise ValueError(f"area_size must be positive, got {area_size!r}")

    from scipy.spatial import KDTree

    rng = np.random.RandomState(seed)
    positions = rng.uniform(0, area_size, size=(n_zones, 2))

    if connection_radius is None:
        connection_radius = area_size / np.sqrt(n_zones) * 3

    tree = KDTree(positions)
    pairs = tree.query_pairs(r=connection_radius, output_type="ndarray")

    G = nx.Graph()
    for i in range(n_zones):
        G.add_node(i, x=float(positions[i, 0]), y=float(positions[i, 1]))

    # Vectorised distance computation for all pairs at once
    if len(pairs) > 0:
        diffs = positions[pairs[:, 0]] - positions[pairs[:, 1]]
        dists = np.sqrt((diffs**2).sum(axis=1))
        for idx in range(len(pairs)):
            G.add_edge(int(pairs[idx, 0]), int(pairs[idx, 1]), cost=float(dists[idx]))

    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        largest_indices = np.array(list(largest))
        largest_positions = positions[largest_indices]
        largest_tree = KDTree(largest_positions)

        for component in components:
            if component == largest:
                continue
            comp_indices = np.array(list(component))
            comp_positions = positions[comp_indices]
            # Find closest node in component to any node in largest
            dists_to_largest, idxs_in_largest = largest_tree.query(comp_positions)
            best_comp_local = int(np.argmin(dists_to_largest))
            node_comp = int(comp_indices[best_comp_local])
            node_largest = int(largest_indices[idxs_in_largest[best_comp_local]])
            cost = float(dists_to_largest[best_comp_local])
            G.add_edge(node_comp, node_largest, cost=cost)

    return G


def generate_transport_network(
    n_zones: int,
    area_size: float = 100000.0,
    seed: int = 42,
) -> nx.Graph:
    """
    Generate a transport-like network via Delaunay triangulation.

    Produces a planar network that mimics road connectivity: every zone is
    connected to its Delaunay neighbours with edge costs equal to the
    Euclidean distance.

    Parameters
    ----------
    n_zones : int
        Number of zones (nodes).
    area_size : float, default=100000.0
        Side length of the square area (meters).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        Connected planar spatial network.
        Node attributes: 'x', 'y' (coordinates).
        Edge attributes: 'cost' (Euclidean distance).
    """
    if not isinstance(n_zones, int) or n_zones < 3:
        raise ValueError(f"n_zones must be an integer >= 3 (Delaunay requirement), got {n_zones!r}")
    if area_size <= 0:
        raise ValueError(f"area_size must be positive, got {area_size!r}")

    from scipy.spatial import Delaunay

    rng = np.random.RandomState(seed)
    positions = rng.uniform(0, area_size, size=(n_zones, 2))

    tri = Delaunay(positions)

    G = nx.Graph()
    for i in range(n_zones):
        G.add_node(i, x=float(positions[i, 0]), y=float(positions[i, 1]))

    for simplex in tri.simplices:
        for k in range(3):
            u, v = int(simplex[k]), int(simplex[(k + 1) % 3])
            if not G.has_edge(u, v):
                dx = positions[u, 0] - positions[v, 0]
                dy = positions[u, 1] - positions[v, 1]
                cost = float(np.sqrt(dx * dx + dy * dy))
                G.add_edge(u, v, cost=cost)

    return G
