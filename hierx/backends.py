"""
Backend abstraction for shortest-path computation.

Backends (in order of preference for ``'auto'``):

1. **scipy** — C-optimised Dijkstra on CSR matrices (always available).
   Supports shared-memory parallelization via ``multiprocessing``.
2. **graph-tool** — C++/Boost backend (optional, must be installed separately).
3. **networkx** — Pure-Python reference implementation.
"""

from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

try:
    import graph_tool as gt
    from graph_tool import topology

    HAS_GRAPH_TOOL = True
except ImportError:
    HAS_GRAPH_TOOL = False
    gt = None
    topology = None


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def select_backend(preference: str = "auto") -> str:
    """Resolve backend choice.

    Parameters
    ----------
    preference : str
        ``'auto'`` (default), ``'scipy'``, ``'networkx'``, or ``'graph-tool'``.

    Returns
    -------
    str
        Resolved backend name.
    """
    if preference == "auto":
        return "scipy"
    if preference in ("scipy", "networkx"):
        return preference
    if preference == "graph-tool":
        if not HAS_GRAPH_TOOL:
            raise ValueError("Backend 'graph-tool' requested but graph-tool is not installed.")
        return "graph-tool"
    raise ValueError(
        f"Unknown backend '{preference}'. Choose 'auto', 'scipy', 'networkx', or 'graph-tool'."
    )


# ---------------------------------------------------------------------------
# Edge cost validation
# ---------------------------------------------------------------------------


def _validate_edge_costs(nx_graph: nx.Graph) -> None:
    """Check that all edges have a finite, non-negative numeric ``cost``.

    Raises
    ------
    ValueError
        If any edge is missing the ``cost`` attribute, has a ``None`` value,
        or has a non-numeric, negative, NaN, or infinite cost.
    """
    import math

    missing: list[tuple] = []
    invalid: list[tuple] = []
    for u, v, d in nx_graph.edges(data=True):
        if "cost" not in d or d["cost"] is None:
            missing.append((u, v))
            continue
        c = d["cost"]
        # Accept int, float, and numpy numeric scalars (np.int64, np.float32, etc.)
        if not isinstance(c, (int, float)) and not (
            hasattr(c, "dtype") and np.issubdtype(type(c), np.number)
        ):
            invalid.append((u, v, c))
        else:
            cf = float(c)
            if math.isnan(cf) or math.isinf(cf) or cf < 0:
                invalid.append((u, v, c))
    if missing:
        sample = missing[:5]
        raise ValueError(
            f"{len(missing)} edge(s) missing a 'cost' attribute "
            f"(sample: {sample}). All edges must have a numeric 'cost'."
        )
    if invalid:
        sample = invalid[:5]
        raise ValueError(
            f"{len(invalid)} edge(s) have invalid cost values "
            f"(sample: {sample}). All edge costs must be finite, "
            f"non-negative numbers (int or float)."
        )


# ---------------------------------------------------------------------------
# NetworkX → CSR conversion
# ---------------------------------------------------------------------------


def convert_nx_to_csr(
    nx_graph: nx.Graph,
) -> tuple[sp.csr_matrix, list[Any], dict[Any, int]]:
    """Convert a NetworkX graph to a scipy CSR matrix.

    Parameters
    ----------
    nx_graph : nx.Graph
        Undirected graph with ``'cost'`` edge attribute.

    Returns
    -------
    csr : scipy.sparse.csr_matrix
        Symmetric weight matrix (N × N).
    idx_to_node : list
        Maps CSR row/col index → NetworkX node ID.
    node_to_idx : dict
        Maps NetworkX node ID → CSR index.
    """
    nodes = sorted(nx_graph.nodes())
    node_to_idx: dict[Any, int] = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for u, v, d in nx_graph.edges(data=True):
        cost = float(d["cost"])
        ui, vi = node_to_idx[u], node_to_idx[v]
        rows.extend([ui, vi])
        cols.extend([vi, ui])
        data.extend([cost, cost])

    csr = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return csr, nodes, node_to_idx


def shortest_paths_scipy(
    csr: sp.csr_matrix,
    source_idx: int,
    idx_to_node: list[Any],
    cutoff: float | None = None,
) -> tuple[dict[Any, float], int]:
    """Single-source shortest paths via scipy's C Dijkstra.

    Returns a (dict, nodes_explored) tuple where dict is keyed by
    *NetworkX* node IDs and nodes_explored is the count of finite
    entries in the Dijkstra distance vector.
    """
    limit = cutoff if cutoff is not None else np.inf
    dist_row = csg.dijkstra(
        csr,
        directed=False,
        indices=[source_idx],
        limit=limit,
    ).ravel()
    result: dict[Any, float] = {}
    finite_mask = np.isfinite(dist_row)
    nodes_explored = int(np.count_nonzero(finite_mask))
    for idx in np.where(finite_mask)[0]:
        result[idx_to_node[idx]] = float(dist_row[idx])
    return result, nodes_explored


def _dists_to_dicts(
    dist: np.ndarray,
    chunk: list[int],
    idx_to_node: list,
    col_node_arr: np.ndarray,
) -> tuple[dict[Any, dict[Any, float]], int]:
    """Convert dense Dijkstra output chunk to nested node-keyed dicts.

    Iterates rows and extracts finite values per row.  This is a shared
    helper used by both the serial ``_chunk_dijkstra_scipy`` path and
    the threaded ``_calculate_costs_threaded_scipy`` path.

    Returns (result_dict, total_explored) where total_explored counts
    finite entries across all rows in this chunk.
    """
    result: dict[Any, dict[Any, float]] = {}
    total_explored = 0
    for row_pos, src_idx in enumerate(chunk):
        row = dist[row_pos]
        finite_idx = np.flatnonzero(np.isfinite(row))
        total_explored += len(finite_idx)
        result[idx_to_node[src_idx]] = dict(
            zip(
                col_node_arr[finite_idx].tolist(),
                row[finite_idx].tolist(),
            )
        )
    return result, total_explored


def _chunk_dijkstra_scipy(
    csr: sp.csr_matrix,
    source_indices: list[int],
    idx_to_node: list[Any],
    rep_indices: np.ndarray | None = None,
    cutoff: float | None = None,
    chunk_size: int = 512,
) -> dict[Any, dict[Any, float]]:
    """Compute shortest paths for many sources in memory-efficient chunks.

    Processes *chunk_size* sources at a time so the intermediate dense
    matrix stays small (``chunk_size × N`` instead of ``N × N``).

    Parameters
    ----------
    csr : scipy.sparse.csr_matrix
        Weight matrix.
    source_indices : list of int
        CSR indices of source nodes.
    idx_to_node : list
        Index → node-ID mapping.
    rep_indices : ndarray, optional
        Only keep distances to these column indices.
    cutoff : float, optional
        Distance limit.
    chunk_size : int
        Number of sources per Dijkstra batch.

    Returns
    -------
    dict[node_id, dict[node_id, float]]
    """
    limit = cutoff if cutoff is not None else np.inf

    # Adaptive chunk size: cap at chunk_size but reduce for large graphs
    # to avoid memory blowup (each chunk allocates chunk_size * N floats)
    n_nodes = csr.shape[0]
    max_chunk = max(16, int(500_000_000 / (n_nodes * 8)))
    chunk_size = min(chunk_size, max_chunk)

    # Column lookup
    n_cols = csr.shape[1]
    if rep_indices is not None and len(rep_indices) < n_cols:
        col_node_arr = np.array([idx_to_node[int(c)] for c in rep_indices])
        do_slice = True
    else:
        col_node_arr = np.array(idx_to_node)
        do_slice = False

    costs: dict[Any, dict[Any, float]] = {}
    total_explored = 0
    for start in range(0, len(source_indices), chunk_size):
        chunk = source_indices[start : start + chunk_size]
        dist = csg.dijkstra(csr, directed=False, indices=chunk, limit=limit)
        # Count finite entries from the FULL dist matrix before column slicing
        # since Dijkstra explored all those nodes regardless of which columns
        # we keep for the result.
        total_explored += int(np.count_nonzero(np.isfinite(dist)))
        if do_slice:
            dist = dist[:, rep_indices]
        chunk_costs, _ = _dists_to_dicts(dist, chunk, idx_to_node, col_node_arr)
        costs.update(chunk_costs)
    return costs, total_explored


# ---------------------------------------------------------------------------
# Shared-memory multiprocessing for scipy backend
# ---------------------------------------------------------------------------

_shm_worker_state: dict[str, Any] = {}


def _shm_pool_initializer(
    shm_data_name: str,
    shm_indices_name: str,
    shm_indptr_name: str,
    data_dtype: np.dtype,
    indices_dtype: np.dtype,
    indptr_dtype: np.dtype,
    data_size: int,
    indices_size: int,
    indptr_size: int,
    matrix_shape: tuple[int, int],
    idx_to_node: list,
) -> None:
    """Attach to shared memory and reconstruct CSR matrix in each worker."""
    import atexit
    from multiprocessing.shared_memory import SharedMemory

    shm_data = SharedMemory(name=shm_data_name, create=False)
    shm_indices = SharedMemory(name=shm_indices_name, create=False)
    shm_indptr = SharedMemory(name=shm_indptr_name, create=False)

    data = np.ndarray(data_size, dtype=data_dtype, buffer=shm_data.buf)
    indices = np.ndarray(indices_size, dtype=indices_dtype, buffer=shm_indices.buf)
    indptr = np.ndarray(indptr_size, dtype=indptr_dtype, buffer=shm_indptr.buf)

    _shm_worker_state["csr"] = sp.csr_matrix(
        (data, indices, indptr), shape=matrix_shape, copy=False
    )
    _shm_worker_state["idx_to_node"] = idx_to_node
    _shm_worker_state["shm_handles"] = [shm_data, shm_indices, shm_indptr]

    def _cleanup():
        for shm in _shm_worker_state.get("shm_handles", []):
            try:
                shm.close()
            except Exception:
                pass

    atexit.register(_cleanup)


def _shm_dijkstra_worker(
    args: tuple,
) -> tuple[dict[Any, dict[Any, float]], int]:
    """Worker: run chunked Dijkstra + dict construction in subprocess."""
    chunk, rep_indices, col_node_arr, do_slice, limit = args
    csr = _shm_worker_state["csr"]
    idx_to_node = _shm_worker_state["idx_to_node"]

    dist = csg.dijkstra(csr, directed=False, indices=chunk, limit=limit)
    # Count finite entries from full dist before column slicing
    nodes_explored = int(np.count_nonzero(np.isfinite(dist)))
    if do_slice:
        dist = dist[:, rep_indices]
    result, _ = _dists_to_dicts(dist, chunk, idx_to_node, col_node_arr)
    return result, nodes_explored


def create_csr_shared_memory(
    csr: sp.csr_matrix,
) -> tuple[list, dict]:
    """Copy CSR arrays into named shared memory segments.

    Returns (shm_handles, shm_info). Caller must close()+unlink() handles.
    """
    from multiprocessing.shared_memory import SharedMemory

    handles = []
    for arr in (csr.data, csr.indices, csr.indptr):
        shm = SharedMemory(create=True, size=arr.nbytes)
        shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np.copyto(shared_arr, arr)
        handles.append(shm)

    shm_info = {
        "shm_data_name": handles[0].name,
        "shm_indices_name": handles[1].name,
        "shm_indptr_name": handles[2].name,
        "data_dtype": csr.data.dtype,
        "indices_dtype": csr.indices.dtype,
        "indptr_dtype": csr.indptr.dtype,
        "data_size": len(csr.data),
        "indices_size": len(csr.indices),
        "indptr_size": len(csr.indptr),
        "matrix_shape": csr.shape,
    }
    return handles, shm_info


# ---------------------------------------------------------------------------
# Fork-based process-parallel scipy backend (fallback when /dev/shm is small)
# ---------------------------------------------------------------------------

_fork_worker_state: dict[str, Any] = {}


def _fork_scipy_pool_initializer(
    csr: sp.csr_matrix,
    idx_to_node: list,
) -> None:
    """Store CSR matrix inherited via fork() COW in each worker."""
    _fork_worker_state["csr"] = csr
    _fork_worker_state["idx_to_node"] = idx_to_node


def _fork_scipy_dijkstra_worker(
    args: tuple,
) -> tuple[dict[Any, dict[Any, float]], int]:
    """Worker: run chunked Dijkstra using fork-inherited CSR matrix."""
    chunk, rep_indices, col_node_arr, do_slice, limit = args
    csr = _fork_worker_state["csr"]
    idx_to_node = _fork_worker_state["idx_to_node"]

    dist = csg.dijkstra(csr, directed=False, indices=chunk, limit=limit)
    nodes_explored = int(np.count_nonzero(np.isfinite(dist)))
    if do_slice:
        dist = dist[:, rep_indices]
    result, _ = _dists_to_dicts(dist, chunk, idx_to_node, col_node_arr)
    return result, nodes_explored


# ---------------------------------------------------------------------------
# Chunked process-parallel graph-tool backend
# ---------------------------------------------------------------------------

_gt_worker_state: dict[str, Any] = {}


def _gt_pool_initializer(
    gt_graph: Any,
    node_to_vertex: dict[Any, int],
    weight_prop: Any,
) -> None:
    """Store graph-tool graph and build reverse vertex mapping in each worker."""
    _gt_worker_state["gt_graph"] = gt_graph
    _gt_worker_state["node_to_vertex"] = node_to_vertex
    _gt_worker_state["weight_prop"] = weight_prop


def _gt_dijkstra_worker(
    args: tuple,
) -> tuple[dict[Any, dict[Any, float]], int]:
    """Worker: run Dijkstra for a chunk of sources via graph-tool."""
    chunk_sources, rep_vertex_indices, rep_node_arr, cutoff = args
    gt_graph = _gt_worker_state["gt_graph"]
    node_to_vertex = _gt_worker_state["node_to_vertex"]
    weight_prop = _gt_worker_state["weight_prop"]

    result: dict[Any, dict[Any, float]] = {}
    total_explored = 0
    for source in chunk_sources:
        if source not in node_to_vertex:
            result[source] = {}
            continue

        source_vertex = gt_graph.vertex(node_to_vertex[source])
        if cutoff is not None:
            dist_map = topology.shortest_distance(
                gt_graph,
                source=source_vertex,
                weights=weight_prop,
                max_dist=cutoff,
            )
        else:
            dist_map = topology.shortest_distance(
                gt_graph,
                source=source_vertex,
                weights=weight_prop,
            )

        # Count finite entries from the full distance map (actual Dijkstra work)
        dist_np = dist_map.a
        total_explored += int(np.count_nonzero(np.isfinite(dist_np)))

        # Vectorized extraction: numpy fancy indexing instead of Python loop
        dist_reps = dist_np[rep_vertex_indices]
        finite_mask = np.isfinite(dist_reps)
        finite_idx = np.flatnonzero(finite_mask)

        result[source] = dict(
            zip(
                rep_node_arr[finite_idx].tolist(),
                dist_reps[finite_idx].tolist(),
            )
        )

    return result, total_explored


# ---------------------------------------------------------------------------
# NetworkX backend
# ---------------------------------------------------------------------------


def shortest_paths_nx(
    network: nx.Graph, source: Any, cutoff: float | None = None
) -> tuple[dict[Any, float], int]:
    """Single-source shortest paths via NetworkX Dijkstra.

    Returns (result_dict, nodes_explored) where nodes_explored is
    the number of nodes reached (length of result dict).
    """
    result = dict(
        nx.single_source_dijkstra_path_length(network, source, cutoff=cutoff, weight="cost")
    )
    return result, len(result)


# ---------------------------------------------------------------------------
# Graph-tool backend
# ---------------------------------------------------------------------------


def convert_nx_to_gt(
    nx_graph: nx.Graph,
) -> tuple[Any, dict[Any, int], Any]:
    """Convert a NetworkX graph to graph-tool format."""
    gt_graph = gt.Graph(directed=False)

    node_to_vertex: dict[Any, int] = {}
    for node in nx_graph.nodes():
        v = gt_graph.add_vertex()
        node_to_vertex[node] = int(v)

    weight_prop = gt_graph.new_edge_property("double")
    for u, v, data in nx_graph.edges(data=True):
        e = gt_graph.add_edge(node_to_vertex[u], node_to_vertex[v])
        weight_prop[e] = float(data["cost"])

    return gt_graph, node_to_vertex, weight_prop


def shortest_paths_gt(
    source_id: Any,
    gt_graph: Any,
    node_to_vertex: dict[Any, int],
    weight_prop: Any,
    cutoff: float | None = None,
) -> tuple[dict[Any, float], int]:
    """Single-source shortest paths via graph-tool.

    Returns (result_dict, nodes_explored) where nodes_explored is
    the count of finite entries in the distance array.
    """
    if source_id not in node_to_vertex:
        return {}, 0

    source_vertex = gt_graph.vertex(node_to_vertex[source_id])

    if cutoff is not None:
        dist_array = topology.shortest_distance(
            gt_graph, source=source_vertex, weights=weight_prop, max_dist=cutoff
        )
    else:
        dist_array = topology.shortest_distance(gt_graph, source=source_vertex, weights=weight_prop)

    vertex_to_node = {v_idx: node_id for node_id, v_idx in node_to_vertex.items()}
    result: dict[Any, float] = {}
    nodes_explored = 0
    for v in gt_graph.vertices():
        dist = dist_array[v]
        if dist < float("inf"):
            result[vertex_to_node[int(v)]] = float(dist)
            nodes_explored += 1
    return result, nodes_explored
