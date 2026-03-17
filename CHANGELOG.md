# Changelog

## [0.1.1] - 2026-03-17

Initial public release.

- `Hierarchy` class: hierarchical cost matrix with greedy spatial clustering,
  takeover mechanism, and multi-layer representative propagation
- `InteractionHierarchy` class: sparse linear operator with correction matrices
  that prevent double-counting across layers
- Three shortest-path backends: scipy (default, with shared-memory
  parallelism on Linux), networkx, and graph-tool (optional)
- Save/load to NPZ (zero dependencies) or HDF5 (optional `h5py`)
- Input validation on all public entry points with clear error messages
- Distance-decay functions (`power_law_interaction`, `exponential_interaction`)
  and network generators for testing
- Dense baseline utilities for validation (`compute_dense_cost_matrix`,
  `compute_dense_interaction_matrix`)
- Python 3.10+ required; depends on networkx, numpy, scipy
- CI: GitHub Actions across Linux/macOS/Windows, Python 3.10-3.12
