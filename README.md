# hierx

Hierarchical cost and interaction matrices for large-scale spatial networks.

## Overview

`hierx` provides an efficient implementation of hierarchical interaction matrices for spatial accessibility analysis. The hierarchical approach reduces computational complexity from O(n^2) to approximately O(n log n) while maintaining good approximation quality.

**Key features:**
- **Fast construction**: Build interaction matrices 5-30x faster than dense approach
- **Memory efficient**: Store only 10-30% of dense matrix entries
- **Scalable**: Handle continent-scale networks with hundreds of thousands of zones
- **Accurate**: Typical approximation error < 5-10% with default parameters

**Use cases:**
- Urban accessibility analysis
- Transportation network modeling
- Spatial interaction modeling
- Gravity model applications

## Method

The hierarchical approach organizes zones into multiple layers with exponentially increasing radii. At each layer:
1. **Representatives** are selected using a greedy spatial clustering algorithm
2. **Costs** are computed only between nearby representatives (within cutoff distance)
3. **Interactions** are derived by applying a distance-decay function to costs
4. **Corrections** prevent double-counting when aggregating across layers

The result is a sparse, multi-resolution representation that can be efficiently multiplied with activity vectors to compute accessibility.

**For detailed conceptual documentation**, see `HIERARCHIES_CONCEPTUAL_DOCUMENTATION.md`.

## Installation

### From source

```bash
git clone https://github.com/hierx/hierx.git
cd hierx
pip install -e .
```

### Requirements

- Python 3.10+
- networkx >= 2.6
- numpy >= 1.21
- scipy >= 1.7
- matplotlib >= 3.5 (optional, for plotting — install via `pip install "hierx[plot]"`)
- graph-tool (optional, alternative shortest-path backend — requires conda or manual install)

## Quick Start

### Basic usage

```python
import networkx as nx
from hierx import Hierarchy, InteractionHierarchy, power_law_interaction

# Create or load a spatial network
G = nx.Graph()
G.add_node(0, x=0, y=0)
G.add_node(1, x=1000, y=0)
G.add_node(2, x=2000, y=0)
G.add_edge(0, 1, cost=10)
G.add_edge(1, 2, cost=15)

# Build hierarchical cost matrix
hierarchy = Hierarchy(G, base_radius=5000, increase_factor=2, overlap_factor=1.5)

# Look up approximate cost between zones (hierarchical approximation)
cost = hierarchy.get_cost(0, 2)
print(f"Approx. cost from zone 0 to zone 2: {cost:.1f}")

# Build interaction hierarchy
def interaction_fn(c):
    return (c + 5000)**(-2)  # Power law decay
ih = InteractionHierarchy(hierarchy, interaction_fn)

# Compute accessibility
import numpy as np
activity = np.array([100, 200, 150])  # Population or opportunities
accessibility = ih.matvec(activity)
print(f"Accessibility of each zone: {accessibility}")
```

### Running examples

The package includes several examples demonstrating different use cases:

```bash
# Toy 4-zone network from documentation
python examples/toy_example.py

# 100-zone grid with visualizations (requires matplotlib: pip install "hierx[plot]")
python examples/small_network.py
```

## API Reference

### Core Classes

#### `Hierarchy`

Hierarchical cost matrix representation.

```python
Hierarchy(network, base_radius=5000, increase_factor=2, overlap_factor=1.5)
```

**Parameters:**
- `network`: Simple undirected `nx.Graph` with integer node IDs, 'cost' edge attribute, and 'x', 'y' node attributes
- `base_radius`: Radius of finest layer (meters or cost units)
- `increase_factor`: Multiplier for exponential layer growth (must be > 1)
- `overlap_factor`: Cutoff distance relative to radius (controls sparsity vs accuracy trade-off)

**Methods:**
- `get_cost(source_zone, dest_zone)`: Look up approximate cost via hierarchical lookup
- `get_density()`: Fraction of costs stored vs dense matrix
- `get_layer_info()`: Information about each hierarchical layer

#### `InteractionHierarchy`

Hierarchical interaction matrix as linear operator.

```python
InteractionHierarchy(hierarchy, interaction_function)
```

**Parameters:**
- `hierarchy`: Hierarchy instance
- `interaction_function`: Function mapping cost to interaction, e.g., `lambda c: (c + offset)**(-2)`

**Methods:**
- `matvec(activity_vector)`: Compute accessibility = InteractionMatrix x activity
- `get_row(source_zone)`: Get interaction row for a single zone
- `get_density()`: Fraction of interactions stored vs dense matrix
- `get_layer_info()`: Information about interaction matrices at each layer

### Utility Functions

```python
# Interaction functions
power_law_interaction(cost, a=1.0, b=-2.0, offset=5000)
exponential_interaction(cost, a=1e-3)

# Network generation
generate_grid_network(n_x, n_y, spacing=1000)
generate_random_spatial_network(n_zones, area_size=100000)
generate_large_spatial_network(n_zones, area_size=100000)
generate_transport_network(n_zones, area_size=100000)

# Dense baselines (for validation)
compute_dense_cost_matrix(network, zones)
compute_dense_interaction_matrix(cost_matrix, interaction_fn)
```

## Testing

```bash
# Run all fast tests
pytest tests/ -m "not slow" -v

# Run all tests including slow ones
pytest tests/ -v
```

## Reproducibility

To reproduce the figures and tables from the paper, see the companion
repository: [hierx-paper](https://github.com/hierx/hierx-paper).

## Parameter Tuning

The hierarchical approach has three key parameters:

### `base_radius`
- **Controls**: Size of finest layer groups
- **Smaller values**: Better local accuracy, more computation
- **Larger values**: Faster, but less accurate for short distances
- **Typical range**: 2000-10000 meters for urban networks

### `increase_factor`
- **Controls**: How quickly layers grow
- **Default**: 2 (each layer has 2x the radius of previous)
- **Higher values**: Fewer layers, faster, less accurate
- **Lower values**: More layers, slower, more accurate

### `overlap_factor`
- **Controls**: Cutoff distance relative to radius
- **Default**: 1.5 (compute costs up to 1.5x layer radius)
- **Higher values**: Better accuracy, more computation and memory
- **Lower values**: Faster and sparser, but less accurate

### `n_workers`
- **Controls**: Parallel construction of cost matrices
- **Default**: 1 (serial)
- **Parallel SciPy** (`n_workers > 1`): Uses Linux shared memory or fork-based pools. On Windows and macOS, the package falls back to serial construction automatically with a warning.
- **For guaranteed cross-platform parallel support**, use `backend="networkx"`.

### Tuning guidelines

- **For better accuracy**: Increase `overlap_factor` to 2.0 or decrease `base_radius`
- **For speed**: Increase `base_radius` or `increase_factor`
- **For large networks** (>10,000 zones): Increase `base_radius` to 10000+

## Citation

If you use this software in academic work, please cite using the metadata in
[CITATION.cff](CITATION.cff):

```bibtex
@software{hellervik2026hierarchies,
  author = {Hellervik, Alexander and Bohlin, Joakim and Andersson, Claes},
  title = {hierx: Hierarchical cost and interaction matrices for spatial networks},
  version = {0.1.1},
  license = {MIT},
  url = {https://github.com/hierx/hierx}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on bug reports and pull requests.

## Authors

Alexander Hellervik (alexander.hellervik@chalmers.se), Chalmers University of Technology
Joakim Bohlin (joakim.bohlin@chalmers.se), Chalmers University of Technology
Claes Andersson (claeand@chalmers.se), Chalmers University of Technology
