"""
Hierarchical cost matrix implementation.

This module implements the Hierarchy class which provides an efficient
representation of cost matrices for large spatial networks using a
multi-level hierarchical grouping strategy.

See HIERARCHIES_CONCEPTUAL_DOCUMENTATION.md for conceptual details.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np

from hierx.backends import (
    _chunk_dijkstra_scipy,
    _fork_scipy_dijkstra_worker,
    _fork_scipy_pool_initializer,
    _gt_dijkstra_worker,
    _gt_pool_initializer,
    _shm_dijkstra_worker,
    _shm_pool_initializer,
    _validate_edge_costs,
    convert_nx_to_csr,
    convert_nx_to_gt,
    create_csr_shared_memory,
    select_backend,
    shortest_paths_gt,
    shortest_paths_nx,
    shortest_paths_scipy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state for networkx pool workers
# ---------------------------------------------------------------------------
_nx_worker_state: dict[str, Any] = {}


def _nx_pool_initializer(network: nx.Graph) -> None:
    """Called once per worker process to store the NetworkX graph."""
    _nx_worker_state["network"] = network


def _nx_dijkstra_worker(
    args: tuple[Any, float | None, set],
) -> tuple[dict[Any, float], int]:
    """Worker for networkx backend (one source at a time)."""
    source, cutoff, rep_set = args
    lengths, explored = shortest_paths_nx(_nx_worker_state["network"], source, cutoff=cutoff)
    return {d: lengths[d] for d in rep_set if d in lengths}, explored


class Hierarchy:
    """
    Hierarchical cost matrix for efficient shortest path storage and lookup.

    The hierarchy organizes zones into multiple layers with exponentially
    increasing radii. At each layer, representative zones store costs to
    nearby representatives, creating a sparse multi-resolution cost structure.

    Attributes
    ----------
    network : nx.Graph
        The spatial network with 'cost' edge attribute
    zones : List[int]
        All zone IDs in the network
    base_radius : float
        Radius of the finest hierarchical layer
    increase_factor : float
        Multiplier for layer radii: r_{k+1} = increase_factor × r_k
    overlap_factor : float
        Controls cutoff size relative to layer radius: cutoff = overlap_factor × r
    repr_zones : Dict[float, List[int]]
        Representative zones at each layer radius
    costs : Dict[float, Dict[int, Dict[int, float]]]
        Sparse cost storage: costs[radius][source][dest] = cost
    cutoffs : Dict[float, Optional[float]]
        Cutoff distance used at each layer
    groups : Dict[float, Dict[int, int]]
        Group membership: groups[radius][zone] = representative
    radii : List[float]
        All layer radii from finest to coarsest

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edge(0, 1, cost=10)
    >>> G.add_edge(1, 2, cost=15)
    >>> h = Hierarchy(G, base_radius=20, increase_factor=2)
    >>> cost = h.get_cost(0, 2)
    >>> print(cost)
    25.0
    """

    def __init__(
        self,
        network: nx.Graph,
        base_radius: float = 5000,
        increase_factor: float = 2,
        overlap_factor: float = 1.5,
        zones: list[int] | None = None,
        backend: str = "auto",
        n_workers: int = 1,
        min_representatives: int = 3,
    ):
        """
        Build hierarchical cost matrix.

        Parameters
        ----------
        network : nx.Graph
            Spatial network with 'cost' edge attribute
        base_radius : float, default=5000
            Radius of finest layer (meters or cost units)
        increase_factor : float, default=2
            Factor for exponential layer growth
        overlap_factor : float, default=1.5
            Cutoff multiplier relative to radius for cost calculation
        zones : list[int] or None, default=None
            Subset of network nodes to treat as zones.  When ``None``
            (the default), every node in the network is a zone.  When a
            list is given, only these nodes participate in the hierarchy
            (representatives, groups, cost lookups, interaction matrices)
            while the full network is still used for shortest-path routing.
        backend : str, default='auto'
            Shortest-path backend: 'auto' (scipy), 'scipy', 'networkx',
            or 'graph-tool'.
        n_workers : int, default=1
            Number of parallel workers for cost computation.
            Only used when > 1 and enough representatives exist.
        min_representatives : int, default=3
            Stop building coarser layers when the number of representatives
            falls below this threshold. For large networks, setting this to
            ~1000 avoids expensive all-pairs computation at the coarsest
            layers while retaining good accuracy.

        Notes
        -----
        Construction algorithm (see HIERARCHIES_CONCEPTUAL_DOCUMENTATION.md):
        1. Generate layers with exponentially increasing radii
        2. Select representatives using greedy spatial clustering
        3. Calculate costs between representatives with cutoffs
        4. Propagate representative assignments to coarser layers
        5. Correct any missing costs within groups

        Computational complexity: O(n × k × c) where:
        - n = number of zones
        - k = number of layers ≈ log(spatial_extent / base_radius)
        - c = average number of zones within cutoff
        """
        # --- Input validation ---
        if not isinstance(network, nx.Graph) or isinstance(
            network, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
        ):
            raise TypeError(
                f"network must be an undirected simple networkx.Graph, got {type(network).__name__}"
            )
        if len(network) == 0:
            raise ValueError("network is empty (has no nodes)")
        if not isinstance(base_radius, (int, float)) or base_radius <= 0:
            raise ValueError(f"base_radius must be a positive number, got {base_radius!r}")
        if not isinstance(increase_factor, (int, float)) or increase_factor <= 1:
            raise ValueError(f"increase_factor must be greater than 1, got {increase_factor!r}")
        if not isinstance(overlap_factor, (int, float)) or overlap_factor <= 0:
            raise ValueError(f"overlap_factor must be a positive number, got {overlap_factor!r}")
        if not isinstance(n_workers, int) or n_workers < 1:
            raise ValueError(f"n_workers must be a positive integer, got {n_workers!r}")
        if not isinstance(min_representatives, int) or min_representatives < 2:
            raise ValueError(
                f"min_representatives must be an integer >= 2, got {min_representatives!r}"
            )
        if zones is not None:
            if not isinstance(zones, (list, tuple, set)) or len(zones) == 0:
                raise ValueError("zones must be a non-empty list/tuple/set of node IDs, or None")
            invalid = set(zones) - set(network.nodes())
            if invalid:
                raise ValueError(
                    f"zones contains {len(invalid)} node(s) not in network: {sorted(invalid)[:5]}"
                )

        self.network = network
        self.zones = sorted(zones) if zones is not None else sorted(network.nodes())

        # Enforce integer node-ID contract (all nodes, not just zones,
        # because non-zone nodes participate in routing)
        for n in network.nodes():
            if not isinstance(n, (int, np.integer)):
                raise TypeError(
                    f"All node IDs must be integers, got {type(n).__name__} "
                    f"for node {n!r}. Use nx.convert_node_labels_to_integers() "
                    f"to relabel."
                )

        self.base_radius = base_radius
        self.increase_factor = increase_factor
        self.overlap_factor = overlap_factor
        self._min_representatives = min_representatives

        # Backend and parallelization
        self._backend = select_backend(backend)
        self._n_workers = n_workers

        # Prepare backend-specific graph representation
        self._csr = None
        self._idx_to_node: list[Any] | None = None
        self._node_to_idx: dict[Any, int] | None = None
        self._gt_graph = None
        self._node_to_vertex = None
        self._weight_prop = None

        # Validate edge costs before any backend conversion
        _validate_edge_costs(network)

        if self._backend == "scipy":
            self._csr, self._idx_to_node, self._node_to_idx = convert_nx_to_csr(network)
        elif self._backend == "graph-tool":
            self._gt_graph, self._node_to_vertex, self._weight_prop = convert_nx_to_gt(network)

        # Pre-compute zone set for O(1) membership checks
        self._zone_set: set = set(self.zones)

        # Initialize data structures
        self.repr_zones: dict[float, list[int]] = {}
        self.costs: dict[float, dict[int, dict[int, float]]] = {}
        self.cutoffs: dict[float, float | None] = {}
        self.groups: dict[float, dict[int, int]] = {}
        self.radii: list[float] = []

        # Build the hierarchy
        self._pool = None
        self._shm_pool = None
        self._shm_handles = None
        self._fork_scipy_pool = False
        try:
            if self._n_workers > 1:
                if self._backend == "scipy":
                    try:
                        self._shm_pool, self._shm_handles = self._create_shm_scipy_pool()
                    except Exception:
                        logger.warning(
                            "Shared-memory pool failed; trying fork-based pool",
                            exc_info=True,
                        )
                        try:
                            self._pool = self._create_fork_scipy_pool()
                        except Exception:
                            logger.warning(
                                "Fork-based pool also failed; falling back to "
                                "serial execution (n_workers=1)",
                                exc_info=True,
                            )
                            self._n_workers = 1
                else:
                    self._pool = self._create_process_pool()
            self._build_hierarchy()
        finally:
            if self._shm_pool is not None:
                self._shm_pool.shutdown(wait=True)
                self._shm_pool = None
            if self._shm_handles is not None:
                for shm in self._shm_handles:
                    shm.close()
                    shm.unlink()
                self._shm_handles = None
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None

    def _create_process_pool(self) -> ProcessPoolExecutor:
        """Create a ProcessPoolExecutor for networkx/graph-tool backends."""
        from concurrent.futures import ProcessPoolExecutor

        if self._backend == "graph-tool":
            return ProcessPoolExecutor(
                max_workers=self._n_workers,
                initializer=_gt_pool_initializer,
                initargs=(
                    self._gt_graph,
                    self._node_to_vertex,
                    self._weight_prop,
                ),
            )

        return ProcessPoolExecutor(
            max_workers=self._n_workers,
            initializer=_nx_pool_initializer,
            initargs=(self.network,),
        )

    def _create_shm_scipy_pool(self) -> tuple[ProcessPoolExecutor, list]:
        """Create a ProcessPoolExecutor with CSR matrix in shared memory."""
        from concurrent.futures import ProcessPoolExecutor

        # Check /dev/shm has enough space (SIGBUS if it doesn't, uncatchable)
        shm_needed = self._csr.data.nbytes + self._csr.indices.nbytes + self._csr.indptr.nbytes
        try:
            import shutil

            shm_total, _, shm_free = shutil.disk_usage("/dev/shm")
            if shm_free < shm_needed * 1.1:  # 10% margin
                raise OSError(
                    f"/dev/shm too small: {shm_free / 1e6:.0f} MB free, "
                    f"need {shm_needed / 1e6:.0f} MB"
                )
        except (OSError, FileNotFoundError) as exc:
            raise OSError(f"Cannot use shared memory: {exc}") from exc

        shm_handles, shm_info = create_csr_shared_memory(self._csr)
        pool = ProcessPoolExecutor(
            max_workers=self._n_workers,
            initializer=_shm_pool_initializer,
            initargs=(
                shm_info["shm_data_name"],
                shm_info["shm_indices_name"],
                shm_info["shm_indptr_name"],
                shm_info["data_dtype"],
                shm_info["indices_dtype"],
                shm_info["indptr_dtype"],
                shm_info["data_size"],
                shm_info["indices_size"],
                shm_info["indptr_size"],
                shm_info["matrix_shape"],
                self._idx_to_node,
            ),
        )
        return pool, shm_handles

    def _create_fork_scipy_pool(self) -> ProcessPoolExecutor:
        """Create a fork-based ProcessPoolExecutor for scipy backend.

        Fallback when /dev/shm is too small for shared memory. Workers inherit
        the CSR matrix via fork() copy-on-write semantics.
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("fork")
        pool = ProcessPoolExecutor(
            max_workers=self._n_workers,
            mp_context=ctx,
            initializer=_fork_scipy_pool_initializer,
            initargs=(self._csr, self._idx_to_node),
        )
        self._fork_scipy_pool = True
        return pool

    def _build_hierarchy(self) -> None:
        """
        Build all hierarchical layers from fine to coarse.

        Algorithm:
        1. Generate layer radii
        2. Build finest layer (all zones as representatives)
        3. Build coarser layers with greedy representative selection
        4. Propagate representatives from coarse to fine
        5. Correct missing costs within groups
        """
        self._total_nodes_explored = 0

        # Generate layer radii
        self.radii = self._generate_radii()

        # Build finest layer (all zones represent themselves)
        finest_radius = self.radii[0]
        self.repr_zones[finest_radius] = list(self.zones)
        self.groups[finest_radius] = {z: z for z in self.zones}
        self.cutoffs[finest_radius] = self.overlap_factor * finest_radius

        # Calculate costs at finest layer
        self.costs[finest_radius], explored = self._calculate_costs(
            self.repr_zones[finest_radius], self.cutoffs[finest_radius]
        )
        self._total_nodes_explored += explored

        # Build coarser layers
        for i in range(1, len(self.radii)):
            radius = self.radii[i]
            prev_radius = self.radii[i - 1]

            # Select representatives using greedy algorithm
            representatives = self._select_representatives(radius, prev_radius)

            # Stop if too few representatives (layer not useful)
            if len(representatives) < self._min_representatives and i > 1:
                logger.info(
                    "Pruning layers from radius %.0f onward (%d representatives < %d threshold)",
                    radius,
                    len(representatives),
                    self._min_representatives,
                )
                self.radii = self.radii[:i]
                # The previous layer is now the coarsest — remove its cutoff
                # so it captures all remaining long-range interactions.
                coarsest = self.radii[-1]
                if self.cutoffs.get(coarsest) is not None:
                    logger.info(
                        "Removing cutoff from new coarsest layer (radius %.0f, "
                        "%d reps) — recomputing without cutoff",
                        coarsest,
                        len(self.repr_zones[coarsest]),
                    )
                    self.cutoffs[coarsest] = None
                    self.costs[coarsest], explored = self._calculate_costs(
                        self.repr_zones[coarsest],
                        None,
                    )
                    self._total_nodes_explored += explored
                break

            self.repr_zones[radius] = representatives

            # Calculate cutoff (no cutoff for coarsest layer)
            if i == len(self.radii) - 1:
                self.cutoffs[radius] = None
            else:
                self.cutoffs[radius] = self.overlap_factor * radius

            # Calculate costs between representatives
            self.costs[radius], explored = self._calculate_costs(
                self.repr_zones[radius], self.cutoffs[radius]
            )
            self._total_nodes_explored += explored

        # Propagate representatives to all layers
        self._propagate_representatives()

        # Correct any missing costs within groups
        self._correct_costs()

    @property
    def total_nodes_explored(self) -> int:
        """Total non-inf entries across all Dijkstra calls during hierarchy build."""
        return self._total_nodes_explored

    def _generate_radii(self) -> list[float]:
        """
        Generate layer radii using exponential growth.

        The extent is always estimated in cost units (same units as edge
        weights) by sampling shortest paths, ensuring the layer structure
        matches the actual network diameter regardless of whether node
        coordinates are in degrees, meters, or absent.

        Returns
        -------
        List[float]
            Layer radii from finest to coarsest
        """
        if len(self.zones) == 0:
            return [self.base_radius]

        # Always use cost-based extent — coordinate-based extent can be
        # in different units (e.g. degrees for lat/lon networks) and
        # would produce wrong layer counts when compared to base_radius
        # (which is in cost units like seconds or meters).
        extent = self._estimate_network_extent()

        # Generate radii until we exceed extent
        radii = []
        radius = self.base_radius
        while radius < extent * 1.5 or len(radii) < 2:
            radii.append(radius)
            radius *= self.increase_factor

        # Ensure at least 2 layers, but not too many
        if len(radii) < 2:
            radii.append(radius)

        return radii

    def _estimate_network_extent(self) -> float:
        """
        Estimate network spatial extent from shortest paths.

        Returns
        -------
        float
            Estimated diameter of network
        """
        # Sample a few nodes and find maximum distance
        sample_size = min(10, len(self.zones))
        sample = self.zones[:: max(1, len(self.zones) // sample_size)]

        max_dist = 0
        for source in sample:
            lengths, _ = self._shortest_paths(source)
            if lengths:
                max_dist = max(max_dist, max(lengths.values()))

        return max_dist if max_dist > 0 else self.base_radius * 10

    def _select_representatives(self, radius: float, prev_radius: float) -> list[int]:
        """
        Select representatives at this layer using greedy algorithm with takeover.

        Algorithm uses greedy spatial clustering with takeover:
        - Start with representatives from previous (finer) layer
        - For each candidate, check if existing representative within radius/2
        - If yes: candidate joins that group (if closer than current assignment)
        - If no: candidate becomes new representative AND can take over zones from other groups

        The takeover mechanism ensures representatives are centrally located:
        - When a new representative is created, it checks all non-representatives
        - If the new representative is closer to a zone than its current representative,
          that zone is reassigned (taken over)

        CRITICAL: This method stores group assignments during construction,
        which are then used by _propagate_representatives() to ensure proper
        nesting of groups across layers.

        Parameters
        ----------
        radius : float
            Current layer radius
        prev_radius : float
            Previous (finer) layer radius

        Returns
        -------
        List[int]
            Representative zones at this layer
        """
        new_representatives = []
        threshold = radius / 2.0

        # Initialize groups for this layer
        if radius not in self.groups:
            self.groups[radius] = {}

        # Track distance from each zone to its assigned representative
        repr_dist = {}

        # Set of current representatives for O(1) membership checks
        rep_set: set = set()

        # Set of non-representatives for takeover filtering
        non_rep_set: set = set()

        # Process candidates from previous layer's representatives
        candidates = list(self.repr_zones[prev_radius])

        # Reuse costs from previous layer (candidates are all prev_radius representatives)
        # This avoids redundant Dijkstra computations
        candidate_costs = {}
        for candidate in candidates:
            if candidate in self.costs[prev_radius]:
                # Use stored costs from previous layer
                candidate_costs[candidate] = self.costs[prev_radius][candidate]
            else:
                # Fallback: compute if not found (shouldn't normally happen)
                logger.warning(
                    "No costs found for candidate %s at radius %s; computing on the fly",
                    candidate,
                    radius,
                )
                candidate_costs[candidate] = self._get_costs_to_zones(candidate, candidates)

        for candidate in candidates:
            # Check if there's already a representative within threshold.
            # Instead of scanning all representatives, iterate over the
            # candidate's *neighbors* (cutoff-limited from previous layer)
            # and check which ones are representatives — O(neighborhood)
            # instead of O(n_reps).
            found_representative = False
            best_repr = None
            min_dist = threshold

            costs_of_candidate = candidate_costs.get(candidate, {})
            if rep_set:
                for neighbor, cost in costs_of_candidate.items():
                    if cost < min_dist and neighbor in rep_set:
                        min_dist = cost
                        best_repr = neighbor
                        found_representative = True

            if not found_representative:
                # Become a new representative
                new_representatives.append(candidate)
                rep_set.add(candidate)
                best_repr = candidate

                # TAKEOVER MECHANISM: Check if this new representative is
                # closer to any non-representative zones than their current
                # representative.  Iterate the new rep's neighbors (small,
                # cutoff-limited) and check against non_rep_set — O(neighborhood).
                for neighbor, dist_to_new_repr in costs_of_candidate.items():
                    if neighbor in non_rep_set:
                        if dist_to_new_repr < repr_dist.get(neighbor, float("inf")):
                            repr_dist[neighbor] = dist_to_new_repr
                            self.groups[radius][neighbor] = candidate
            else:
                # Join existing representative's group
                non_rep_set.add(candidate)

            # Store the group assignment and distance
            self.groups[radius][candidate] = best_repr
            repr_dist[candidate] = min_dist

        return new_representatives

    def _get_costs_to_zones(self, source: int, targets: list[int]) -> dict[int, float]:
        """
        Get costs from source to target zones.

        Parameters
        ----------
        source : int
            Source zone
        targets : List[int]
            Target zones

        Returns
        -------
        Dict[int, float]
            Mapping from target zone to cost
        """
        lengths, _ = self._shortest_paths(source)

        costs = {}
        for target in targets:
            if target in lengths:
                costs[target] = lengths[target]
            else:
                costs[target] = np.inf

        return costs

    def _shortest_paths(
        self, source: Any, cutoff: float | None = None
    ) -> tuple[dict[Any, float], int]:
        """Compute single-source shortest path lengths via the configured backend.

        Returns (lengths_dict, nodes_explored).
        """
        if self._backend == "scipy":
            return shortest_paths_scipy(
                self._csr,
                self._node_to_idx[source],
                self._idx_to_node,
                cutoff=cutoff,
            )
        if self._backend == "graph-tool":
            return shortest_paths_gt(
                source,
                self._gt_graph,
                self._node_to_vertex,
                self._weight_prop,
                cutoff=cutoff,
            )
        # NetworkX as backend (fallback)
        return shortest_paths_nx(self.network, source, cutoff=cutoff)

    def _calculate_costs(
        self, representatives: list[int], cutoff: float | None
    ) -> tuple[dict[int, dict[int, float]], int]:
        """
        Calculate costs between representatives with optional cutoff.

        Parameters
        ----------
        representatives : List[int]
            Representative zones at this layer
        cutoff : Optional[float]
            Maximum cost to compute. If None, compute all pairs.

        Returns
        -------
        tuple[dict, int]
            (costs_dict, total_nodes_explored) where costs_dict is
            nested dict costs[source][dest] = cost, and total_nodes_explored
            counts non-inf entries across all Dijkstra calls.
        """
        rep_set = set(representatives)

        # Scipy backend: chunked C Dijkstra with optional threading
        if self._backend == "scipy":
            src_indices = [self._node_to_idx[r] for r in representatives]
            rep_indices = np.array(src_indices, dtype=np.intp)

            if self._n_workers > 1 and len(representatives) >= 4:
                if self._shm_pool is not None:
                    return self._calculate_costs_parallel_scipy(
                        src_indices,
                        rep_indices,
                        cutoff,
                    )
                if self._pool is not None and getattr(self, "_fork_scipy_pool", False):
                    return self._calculate_costs_parallel_scipy(
                        src_indices,
                        rep_indices,
                        cutoff,
                        use_fork_pool=True,
                    )

            return _chunk_dijkstra_scipy(
                self._csr,
                src_indices,
                self._idx_to_node,
                rep_indices=rep_indices,
                cutoff=cutoff,
            )

        # Graph-tool backend: chunked parallel
        if self._backend == "graph-tool":
            if self._n_workers > 1 and len(representatives) >= 4 and self._pool is not None:
                return self._calculate_costs_parallel_gt(representatives, cutoff)
            # Serial fallback
            costs: dict[int, dict[int, float]] = {}
            total_explored = 0
            for source in representatives:
                lengths, explored = self._shortest_paths(source, cutoff=cutoff)
                total_explored += explored
                costs[source] = {d: lengths[d] for d in rep_set if d in lengths}
            return costs, total_explored

        # NetworkX backend
        if self._n_workers > 1 and len(representatives) >= 4 and self._pool is not None:
            return self._calculate_costs_parallel_nx(representatives, rep_set, cutoff)

        costs_nx: dict[int, dict[int, float]] = {}
        total_explored = 0
        for source in representatives:
            lengths, explored = self._shortest_paths(source, cutoff=cutoff)
            total_explored += explored
            costs_nx[source] = {d: lengths[d] for d in rep_set if d in lengths}
        return costs_nx, total_explored

    def _calculate_costs_parallel_scipy(
        self,
        src_indices: list[int],
        rep_indices: np.ndarray,
        cutoff: float | None,
        use_fork_pool: bool = False,
    ) -> tuple[dict[int, dict[int, float]], int]:
        """Process-parallel chunked Dijkstra via scipy.

        Uses shared-memory pool (default) or fork-based pool (fallback when
        /dev/shm is too small). Workers run Dijkstra AND build result dicts,
        avoiding both GIL contention and large-array serialization.
        """
        n_nodes = self._csr.shape[0]
        max_chunk = max(16, int(500_000_000 / (n_nodes * 8)))
        chunk_size = min(2048, max_chunk)
        limit = cutoff if cutoff is not None else np.inf
        idx_to_node = self._idx_to_node
        n_cols = self._csr.shape[1]

        if rep_indices is not None and len(rep_indices) < n_cols:
            col_node_arr = np.array([idx_to_node[int(c)] for c in rep_indices])
            do_slice = True
        else:
            col_node_arr = np.array(idx_to_node)
            do_slice = False

        chunks = [src_indices[i : i + chunk_size] for i in range(0, len(src_indices), chunk_size)]

        tasks = [
            (chunk, rep_indices if do_slice else None, col_node_arr, do_slice, limit)
            for chunk in chunks
        ]

        pool = self._pool if use_fork_pool else self._shm_pool
        worker = _fork_scipy_dijkstra_worker if use_fork_pool else _shm_dijkstra_worker

        costs: dict[int, dict[int, float]] = {}
        total_explored = 0
        for chunk_costs, chunk_explored in pool.map(worker, tasks):
            costs.update(chunk_costs)
            total_explored += chunk_explored
        return costs, total_explored

    def _calculate_costs_parallel_gt(
        self,
        representatives: list[int],
        cutoff: float | None,
    ) -> tuple[dict[int, dict[int, float]], int]:
        """Chunked process-parallel Dijkstra via graph-tool.

        Sources are batched into chunks to minimize pickle overhead.
        Each worker runs single-source Dijkstra for all sources in its
        chunk and extracts results via numpy vectorization.
        """
        rep_vertex_indices = np.array(
            [self._node_to_vertex[r] for r in representatives], dtype=np.intp
        )
        rep_node_arr = np.array(representatives)

        # ~4 chunks per worker for load balancing
        chunk_size = max(16, len(representatives) // (self._n_workers * 4))
        chunks = [
            representatives[i : i + chunk_size] for i in range(0, len(representatives), chunk_size)
        ]

        tasks = [(chunk, rep_vertex_indices, rep_node_arr, cutoff) for chunk in chunks]

        costs: dict[int, dict[int, float]] = {}
        total_explored = 0
        for chunk_costs, chunk_explored in self._pool.map(_gt_dijkstra_worker, tasks):
            costs.update(chunk_costs)
            total_explored += chunk_explored
        return costs, total_explored

    def _calculate_costs_parallel_nx(
        self,
        representatives: list[int],
        rep_set: set,
        cutoff: float | None,
    ) -> tuple[dict[int, dict[int, float]], int]:
        """Process-based parallelism for networkx backend."""
        n = min(self._n_workers, len(representatives))
        chunksize = max(1, len(representatives) // n)
        tasks = [(src, cutoff, rep_set) for src in representatives]

        costs: dict[int, dict[int, float]] = {}
        total_explored = 0
        for source, (source_costs, explored) in zip(
            representatives,
            self._pool.map(_nx_dijkstra_worker, tasks, chunksize=chunksize),
        ):
            costs[source] = source_costs
            total_explored += explored
        return costs, total_explored

    def _propagate_representatives(self) -> None:
        """
        Propagate representative assignments from coarser to finer layers.

        Ensures every zone knows its representative at every layer by
        using the stored representative relationships from construction.

        Key principle: If zone A has repr B at layer k, then at coarser layer k+1,
        zone A should have whatever B has at layer k+1. This ensures proper nesting.

        Ensures proper nesting of group assignments across layers.
        """
        # Process from coarsest to finest (indices from high to low)
        # Build list of (radius, zones_at_that_layer) tuples
        layers_with_reps = []
        for i in range(len(self.radii) - 1, -1, -1):
            radius = self.radii[i]
            layers_with_reps.append((radius, list(self.repr_zones[radius])))

        # Pre-build sets of representatives per layer for O(1) membership checks
        repr_sets: dict[float, set] = {r: set(self.repr_zones[r]) for r in self.radii}

        # Track which coarser layers we've processed
        coarser_layers = [layers_with_reps[0][0]]

        # Process from coarser to finer
        for layer_idx in range(1, len(layers_with_reps)):
            radius, reps = layers_with_reps[layer_idx]

            # For each representative at this finer layer
            groups_at_radius = self.groups[radius]
            for zone in reps:
                # Get this zone's representative at its own layer (should be already set)
                repr_at_this_layer = groups_at_radius.get(zone, zone)

                # Propagate through all coarser layers
                for coarser_layer in reversed(coarser_layers):
                    groups_at_coarser = self.groups[coarser_layer]

                    # Get the representative's representative at the coarser layer
                    next_repr = groups_at_coarser.get(repr_at_this_layer, repr_at_this_layer)

                    # Only update if not already set (avoid overwriting)
                    # This is important for zones that are representatives at coarser layers
                    if zone not in groups_at_coarser or zone not in repr_sets[coarser_layer]:
                        groups_at_coarser[zone] = next_repr

                    # Update for next iteration
                    repr_at_this_layer = next_repr

            # Add this layer to processed coarser layers for next iteration
            coarser_layers.append(radius)

    def _correct_costs(self) -> None:
        """
        Ensure within-group representative pairs have fine-layer costs.

        At layer k, costs are computed with a cutoff. At layer k+1, grouping
        (including takeover) may place layer-k representatives in the same
        group even if they are beyond layer k's cutoff. Without correction,
        ``get_cost`` can fall through to layer k+1, see both zones map to the
        same representative, and return that representative's self-cost 0.

        Fix: for each layer k, look at the groups induced by layer k+1.
        Whenever two layer-k representatives share a layer-(k+1) representative
        but are missing a layer-k cost entry, run a bounded shortest-path search
        with ``cutoff=coarse_radius`` and fill in only the missing costs to
        other members of that same coarse group.
        """
        for k in range(len(self.radii) - 1):
            fine_radius = self.radii[k]
            coarse_radius = self.radii[k + 1]

            # Reverse-index: coarse rep → [fine-layer reps in that group]
            coarse_groups: dict[int, list[int]] = {}
            for zone in self.repr_zones[fine_radius]:
                coarse_rep = self.groups[coarse_radius][zone]
                coarse_groups.setdefault(coarse_rep, []).append(zone)

            # Phase 1: Collect all sources with missing pairs
            source_to_missing: dict[int, set[int]] = {}
            all_members: set[int] = set()
            for members in coarse_groups.values():
                if len(members) < 2:
                    continue
                member_set = set(members)
                for src in members:
                    src_costs = self.costs[fine_radius].get(src, {})
                    missing = member_set.difference(src_costs)
                    if missing:
                        source_to_missing[src] = missing
                        all_members.update(member_set)

            if not source_to_missing:
                continue

            # Phase 2: Dispatch parallel or serial
            sources = list(source_to_missing)
            if self._n_workers > 1 and len(sources) >= 4:
                explored = self._correct_costs_parallel(
                    sources,
                    source_to_missing,
                    all_members,
                    fine_radius,
                    coarse_radius,
                )
            else:
                explored = self._correct_costs_serial(
                    sources,
                    source_to_missing,
                    fine_radius,
                    coarse_radius,
                )
            self._total_nodes_explored += explored

    def _correct_costs_serial(
        self,
        sources: list[int],
        source_to_missing: dict[int, set[int]],
        fine_radius: float,
        coarse_radius: float,
    ) -> int:
        """Run correction Dijkstras serially.

        Returns total nodes explored.
        """
        total_explored = 0
        for src in sources:
            lengths, explored = self._shortest_paths(src, cutoff=coarse_radius)
            total_explored += explored
            src_costs = self.costs[fine_radius].setdefault(src, {})
            for dst in source_to_missing[src]:
                if dst in lengths:
                    src_costs[dst] = lengths[dst]
        return total_explored

    def _correct_costs_parallel(
        self,
        sources: list[int],
        source_to_missing: dict[int, set[int]],
        all_members: set[int],
        fine_radius: float,
        coarse_radius: float,
    ) -> int:
        """Run correction Dijkstras in parallel using the existing pool.

        Dispatches to the appropriate backend-specific parallel path, then
        post-filters results to only write entries in *source_to_missing*.

        Returns total nodes explored.
        """
        if self._backend == "scipy":
            src_indices = [self._node_to_idx[s] for s in sources]
            rep_indices = np.array(
                [self._node_to_idx[m] for m in all_members],
                dtype=np.intp,
            )
            use_fork = False
            if self._shm_pool is not None:
                pass  # default: use shm pool
            elif self._pool is not None and getattr(self, "_fork_scipy_pool", False):
                use_fork = True
            else:
                # No parallel pool available, fall back to serial
                return self._correct_costs_serial(
                    sources,
                    source_to_missing,
                    fine_radius,
                    coarse_radius,
                )
            all_costs, total_explored = self._calculate_costs_parallel_scipy(
                src_indices,
                rep_indices,
                coarse_radius,
                use_fork_pool=use_fork,
            )
        elif self._backend == "graph-tool":
            if self._pool is None:
                return self._correct_costs_serial(
                    sources,
                    source_to_missing,
                    fine_radius,
                    coarse_radius,
                )
            all_costs, total_explored = self._correct_costs_parallel_gt(
                sources,
                all_members,
                coarse_radius,
            )
        else:
            # NetworkX backend
            if self._pool is None:
                return self._correct_costs_serial(
                    sources,
                    source_to_missing,
                    fine_radius,
                    coarse_radius,
                )
            all_costs, total_explored = self._correct_costs_parallel_nx(
                sources,
                all_members,
                coarse_radius,
            )

        # Post-filter: only write the pairs that were actually missing
        for src in sources:
            if src not in all_costs:
                continue
            src_costs = self.costs[fine_radius].setdefault(src, {})
            for dst in source_to_missing[src]:
                if dst in all_costs[src]:
                    src_costs[dst] = all_costs[src][dst]
        return total_explored

    def _correct_costs_parallel_gt(
        self,
        sources: list[int],
        all_members: set[int],
        cutoff: float,
    ) -> tuple[dict[int, dict[int, float]], int]:
        """Parallel correction Dijkstras via graph-tool."""
        member_list = sorted(all_members)
        rep_vertex_indices = np.array(
            [self._node_to_vertex[m] for m in member_list],
            dtype=np.intp,
        )
        rep_node_arr = np.array(member_list)

        chunk_size = max(16, len(sources) // (self._n_workers * 4))
        chunks = [sources[i : i + chunk_size] for i in range(0, len(sources), chunk_size)]
        tasks = [(chunk, rep_vertex_indices, rep_node_arr, cutoff) for chunk in chunks]

        costs: dict[int, dict[int, float]] = {}
        total_explored = 0
        for chunk_costs, chunk_explored in self._pool.map(_gt_dijkstra_worker, tasks):
            costs.update(chunk_costs)
            total_explored += chunk_explored
        return costs, total_explored

    def _correct_costs_parallel_nx(
        self,
        sources: list[int],
        all_members: set[int],
        cutoff: float,
    ) -> tuple[dict[int, dict[int, float]], int]:
        """Parallel correction Dijkstras via networkx."""
        n = min(self._n_workers, len(sources))
        chunksize = max(1, len(sources) // n)
        tasks = [(src, cutoff, all_members) for src in sources]

        costs: dict[int, dict[int, float]] = {}
        total_explored = 0
        for source, (source_costs, explored) in zip(
            sources,
            self._pool.map(_nx_dijkstra_worker, tasks, chunksize=chunksize),
        ):
            costs[source] = source_costs
            total_explored += explored
        return costs, total_explored

    def get_cost(self, source_zone: int, dest_zone: int) -> float:
        """
        Look up the approximate cost between two zones via hierarchical
        representative mapping.

        Iterates through layers from finest to coarsest.  At each layer the
        source and destination are mapped to their respective representatives,
        and the stored cost between those representatives is returned as soon
        as one is found.

        Parameters
        ----------
        source_zone : int
            Source zone ID
        dest_zone : int
            Destination zone ID

        Returns
        -------
        float
            Approximate travel cost between the two zones.
            Returns inf if zones are disconnected.

        Notes
        -----
        This is a **hierarchical approximation**, not an exact shortest path.
        The returned value is the cost between the *representatives* of the
        source and destination at the finest layer where a stored cost exists.
        It is exact when both zones are their own representatives at that
        layer, but approximate otherwise — the error depends on how far each
        zone is from its representative.

        Accuracy is controlled primarily by ``overlap_factor`` (higher values
        store more costs at fine layers) and ``base_radius`` (smaller values
        create finer initial groupings).  See the Parameter Tuning section in
        the README for guidance.

        Examples
        --------
        >>> cost = hierarchy.get_cost(0, 10)
        >>> print(f"Approx. cost from zone 0 to zone 10: {cost:.1f}")
        """
        if source_zone not in self._zone_set:
            sample = list(self._zone_set)[:5]
            raise ValueError(
                f"source_zone {source_zone!r} not in hierarchy. Valid zones (sample): {sample}"
            )
        if dest_zone not in self._zone_set:
            sample = list(self._zone_set)[:5]
            raise ValueError(
                f"dest_zone {dest_zone!r} not in hierarchy. Valid zones (sample): {sample}"
            )
        # Iterate from finest to coarsest layer
        for radius in self.radii:
            # Get representatives at this layer
            source_repr = self.groups[radius][source_zone]
            dest_repr = self.groups[radius][dest_zone]

            # Check if cost exists at this layer
            if source_repr in self.costs[radius]:
                if dest_repr in self.costs[radius][source_repr]:
                    return self.costs[radius][source_repr][dest_repr]

        # Should never reach here if hierarchy was correctly constructed
        return np.inf

    def get_density(self) -> float:
        """
        Calculate storage density compared to dense matrix.

        Reports the total storage across all layers as a fraction of dense.
        Note: This can exceed 1.0 because we store costs at multiple layers.

        Returns
        -------
        float
            Total costs stored across all layers / dense matrix size (n^2)
            Can be > 1.0 for small networks with many layers

        Examples
        --------
        >>> density = hierarchy.get_density()
        >>> print(f"Total storage: {density*100:.1f}% of single dense matrix")
        >>>
        >>> # For effective sparsity, look at finest layer only:
        >>> finest_radius = hierarchy.radii[0]
        >>> finest_costs = sum(len(hierarchy.costs[finest_radius][s])
        ...                   for s in hierarchy.costs[finest_radius])
        >>> finest_density = finest_costs / (len(hierarchy.zones)**2)
        """
        n = len(self.zones)
        dense_size = n * n

        # Count stored costs across all layers
        stored_count = 0
        for radius in self.radii:
            for source in self.costs[radius]:
                stored_count += len(self.costs[radius][source])

        return stored_count / dense_size if dense_size > 0 else 0.0

    def get_finest_layer_density(self) -> float:
        """
        Calculate storage density at finest layer only.

        This gives a better measure of actual sparsity since it only
        counts costs stored at the most detailed level.

        Returns
        -------
        float
            Fraction of costs stored at finest layer vs. dense matrix
        """
        n = len(self.zones)
        dense_size = n * n

        if not self.radii or dense_size == 0:
            return 0.0

        finest_radius = self.radii[0]
        finest_count = sum(len(self.costs[finest_radius][s]) for s in self.costs[finest_radius])

        return finest_count / dense_size

    def verify_group_nesting(self) -> tuple[bool, list[str]]:
        """
        Verify that groups properly nest across layers.

        Checks that if two zones are in the same group at a finer layer,
        they must also be in the same group at all coarser layers.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_errors)
            is_valid: True if all groups nest properly
            list_of_errors: List of error messages if validation fails

        Examples
        --------
        >>> is_valid, errors = hierarchy.verify_group_nesting()
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(error)
        """
        errors = []

        # Check each pair of layers (finer, coarser)
        for i in range(len(self.radii) - 1):
            finer_radius = self.radii[i]

            for j in range(i + 1, len(self.radii)):
                coarser_radius = self.radii[j]

                # Get all groups at finer layer
                finer_groups = defaultdict(set)
                for zone, repr in self.groups[finer_radius].items():
                    finer_groups[repr].add(zone)

                # Check each group at finer layer
                for repr, zones_in_group in finer_groups.items():
                    if len(zones_in_group) < 2:
                        continue  # Single-zone groups automatically nest

                    # Get coarse representatives for all zones in this fine group
                    coarse_reprs = set()
                    for zone in zones_in_group:
                        if zone in self.groups[coarser_radius]:
                            coarse_reprs.add(self.groups[coarser_radius][zone])

                    # All zones in the fine group should have the same coarse representative
                    if len(coarse_reprs) > 1:
                        errors.append(
                            f"Group nesting violation: Zones "
                            f"{list(zones_in_group)[:5]}... "
                            f"in group {repr} at layer "
                            f"{finer_radius:.0f} "
                            f"are split into {len(coarse_reprs)}"
                            f" groups at layer "
                            f"{coarser_radius:.0f}: "
                            f"{list(coarse_reprs)}"
                        )

        return len(errors) == 0, errors

    def save(self, path, *, include_network: bool = False, backend: str = "auto"):
        """Save this hierarchy to disk.

        Parameters
        ----------
        path : str or Path
            Destination path.  ``.npz`` or ``.h5`` is appended if missing.
        include_network : bool, default False
            Whether to embed the NetworkX graph.
        backend : str, default "auto"
            ``"auto"``, ``"npz"``, or ``"hdf5"``.

        Returns
        -------
        Path
            The path the file was written to.
        """
        from hierx.storage import save_hierarchy

        return save_hierarchy(self, path, include_network=include_network, backend=backend)

    @classmethod
    def load(cls, path, *, network=None):
        """Load a Hierarchy from a ``.npz`` or ``.h5`` file.

        Parameters
        ----------
        path : str or Path
            File to load.
        network : nx.Graph or None, optional
            If provided, used as the network.

        Returns
        -------
        Hierarchy
        """
        from hierx.storage import load_hierarchy

        return load_hierarchy(path, network=network)

    def get_layer_info(self) -> list[dict]:
        """
        Get information about each hierarchical layer.

        Returns
        -------
        List[Dict]
            List of layer info dictionaries with keys:
            - 'radius': layer radius
            - 'n_representatives': number of representatives
            - 'cutoff': cutoff distance used
            - 'n_costs': number of costs stored

        Examples
        --------
        >>> for layer in hierarchy.get_layer_info():
        ...     print(f"Layer r={layer['radius']:.0f}: "
        ...           f"{layer['n_representatives']} representatives")
        """
        info = []
        for radius in self.radii:
            n_costs = sum(len(self.costs[radius][s]) for s in self.costs[radius])
            info.append(
                {
                    "radius": radius,
                    "n_representatives": len(self.repr_zones[radius]),
                    "cutoff": self.cutoffs[radius],
                    "n_costs": n_costs,
                }
            )
        return info
