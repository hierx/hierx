"""Shared test fixtures for hierx tests."""

import networkx as nx
import pytest

from hierx import generate_grid_network


@pytest.fixture
def toy_network():
    """4-zone linear graph A-B-C-D from documentation."""
    G = nx.Graph()
    G.add_node(0, label="A", x=0, y=0)
    G.add_node(1, label="B", x=20, y=0)
    G.add_node(2, label="C", x=40, y=0)
    G.add_node(3, label="D", x=60, y=0)
    G.add_edge(0, 1, cost=10)
    G.add_edge(1, 2, cost=15)
    G.add_edge(2, 3, cost=20)
    return G


@pytest.fixture
def grid_5x5():
    """5x5 grid network with 1000m spacing."""
    return generate_grid_network(5, 5, spacing=1000)


@pytest.fixture
def grid_10x10():
    """10x10 grid network with 1000m spacing."""
    return generate_grid_network(10, 10, spacing=1000)


@pytest.fixture
def default_interaction_fn():
    """Standard interaction function: (c + 1000)^{-2}."""
    return lambda c: (c + 1000) ** (-2)
