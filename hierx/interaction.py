"""
Hierarchical interaction matrix implementation.

This module implements the InteractionHierarchy class which builds upon
the Hierarchy to compute interaction matrices efficiently as a linear operator.

See HIERARCHIES_CONCEPTUAL_DOCUMENTATION.md for conceptual details on:
- Correction matrices
- Matrix-vector products
- Avoiding double-counting

"""

from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from hierx.hierarchy import Hierarchy


class InteractionHierarchy(LinearOperator):
    """
    Hierarchical interaction matrix as a linear operator.

    Applies a distance-decay function to the hierarchical cost matrix and
    implements efficient matrix-vector multiplication for accessibility computation.

    The key challenge is avoiding double-counting: zone pairs appear at multiple
    layers, and correction matrices subtract coarse-layer contributions when
    fine-layer data exists.

    Attributes
    ----------
    hierarchy : Hierarchy
        The underlying hierarchical cost matrix
    interaction_fn : Callable[[float], float]
        Distance-to-interaction function f(cost) → interaction
    self_interaction : bool
        Whether to include self-interaction (diagonal) at the finest layer
    D : dict[float, sp.csr_matrix]
        Interaction matrices at each layer (sparse)
    D_dict : dict[float, dict[int, dict[int, float]]]
        Interaction values keyed by zone IDs: D_dict[radius][src][dst]
    Corr : dict[float, sp.csr_matrix]
        Correction matrices at each layer (sparse)
    G : dict[float, sp.csr_matrix]
        Group membership matrices at each layer (sparse, boolean)
    zone_indices : dict[float, dict[int, int]]
        Mapping from zone ID to matrix index at each layer

    Examples
    --------
    >>> from hierx import power_law_interaction
    >>> interaction_fn = lambda c: (c + 5)**(-1)
    >>> ih = InteractionHierarchy(hierarchy, interaction_fn)
    >>> activity = np.array([5, 10, 8, 3])  # Population
    >>> accessibility = ih.matvec(activity)  # Compute accessibility
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        interaction_fn: Callable[[float], float],
        self_interaction: bool = True,
    ):
        """
        Build hierarchical interaction matrix.

        Parameters
        ----------
        hierarchy : Hierarchy
            The hierarchical cost matrix
        interaction_fn : Callable[[float], float]
            Function mapping cost to interaction: f(cost) → interaction
            Example: lambda c: (c + 5000)**(-2)  # Power law
        self_interaction : bool, default True
            Whether to include self-interaction at the finest layer. When
            False, the diagonal of D at the finest layer is zeroed out.
            Coarser layers are unaffected because their "self-interaction"
            is actually inter-zone interaction via representative aggregation.

        Notes
        -----
        Construction algorithm (see HIERARCHIES_CONCEPTUAL_DOCUMENTATION.md):
        1. Build interaction matrices D[k] = f(costs[k])
        2. Build correction matrices Corr[k] using the correction algorithm
        3. Build group membership matrices G[k]

        Computational complexity: O(n × k × c) where:
        - n = number of zones
        - k = number of layers
        - c = average zones within cutoff
        """
        self.hierarchy = hierarchy
        self.interaction_fn = interaction_fn
        self.self_interaction = self_interaction

        # Initialize data structures
        self.D: dict[float, sp.csr_matrix] = {}
        self.D_dict: dict[float, dict[int, dict[int, float]]] = {}
        self.Corr: dict[float, sp.csr_matrix] = {}
        self.G: dict[float, sp.csr_matrix] = {}
        self.zone_indices: dict[float, dict[int, int]] = {}

        # Build interaction and correction matrices
        self._build_interaction_matrices()
        self._build_group_matrices()
        self._build_correction_matrices()

        # Precompute net operator per layer: D[k] - Corr[k]
        self._net: dict[float, sp.csr_matrix] = {}
        for radius in self.hierarchy.radii:
            self._net[radius] = (self.D[radius] - self.Corr[radius]).tocsr()

        # Initialize LinearOperator base class
        n_zones = len(hierarchy.zones)
        super().__init__(shape=(n_zones, n_zones), dtype=np.float64)

    def _build_interaction_matrices(self) -> None:
        """
        Build interaction matrices D[k] from cost matrices.

        For each layer k, D[k][i,j] = f(C[k][i,j]) where f is the
        interaction function.

        Also populates D_dict[k][src_zone][dst_zone] = interaction_value
        for O(1) element access by zone ID.
        """
        finest_radius = self.hierarchy.radii[0]

        for radius in self.hierarchy.radii:
            representatives = self.hierarchy.repr_zones[radius]
            n_reps = len(representatives)

            # Create index mapping
            self.zone_indices[radius] = {zone: i for i, zone in enumerate(representatives)}

            # Build sparse interaction matrix and D_dict simultaneously
            row_indices = []
            col_indices = []
            data = []
            d_dict_layer: dict[int, dict[int, float]] = {}

            costs = self.hierarchy.costs[radius]
            for source in representatives:
                source_idx = self.zone_indices[radius][source]

                if source in costs:
                    for dest, cost in costs[source].items():
                        # Only include if dest is also a representative at this layer
                        if dest in self.zone_indices[radius]:
                            dest_idx = self.zone_indices[radius][dest]
                            interaction = self.interaction_fn(cost)

                            row_indices.append(source_idx)
                            col_indices.append(dest_idx)
                            data.append(interaction)

                            d_dict_layer.setdefault(source, {})[dest] = interaction

            # Create sparse matrix
            self.D[radius] = sp.csr_matrix(
                (data, (row_indices, col_indices)), shape=(n_reps, n_reps)
            )

            # Zero diagonal at finest layer when self_interaction is disabled
            if not self.self_interaction and radius == finest_radius:
                D_lil = self.D[radius].tolil()
                for i in range(n_reps):
                    D_lil[i, i] = 0.0
                self.D[radius] = D_lil.tocsr()
                self.D[radius].eliminate_zeros()
                # Also remove diagonal entries from D_dict
                for zone in representatives:
                    if zone in d_dict_layer and zone in d_dict_layer[zone]:
                        del d_dict_layer[zone][zone]

            self.D_dict[radius] = d_dict_layer

    def _build_group_matrices(self) -> None:
        """
        Build group membership matrices G[k].

        G[k] is a sparse boolean matrix where G[k][rep_idx, zone_idx] = 1
        if zone belongs to representative's group at layer k.

        Matrix dimensions: (n_representatives[k], n_zones)
        """
        n_zones = len(self.hierarchy.zones)
        zone_to_index = {zone: i for i, zone in enumerate(self.hierarchy.zones)}

        for radius in self.hierarchy.radii:
            representatives = self.hierarchy.repr_zones[radius]
            n_reps = len(representatives)

            row_indices = []
            col_indices = []
            data = []

            # For each zone, find its representative at this layer
            for zone in self.hierarchy.zones:
                zone_idx = zone_to_index[zone]
                rep = self.hierarchy.groups[radius][zone]
                rep_idx = self.zone_indices[radius][rep]

                row_indices.append(rep_idx)
                col_indices.append(zone_idx)
                data.append(1.0)

            # Create sparse matrix
            self.G[radius] = sp.csr_matrix(
                (data, (row_indices, col_indices)), shape=(n_reps, n_zones)
            )

    def _build_correction_matrices(self) -> None:
        """
        Build correction matrices to prevent double-counting.

        Algorithm (see HIERARCHIES_CONCEPTUAL_DOCUMENTATION.md):
        For each finer layer k':
            For each coarser layer k > k':
                For each representative pair (i, j) at layer k':
                    Find their representatives at layer k
                    If coarse interaction exists AND fine interaction exists:
                        Subtract coarse interaction from fine layer

        This ensures each zone pair contributes exactly once at the
        finest available resolution.

        Implementation is vectorized: for each fine layer, all nnz entries
        are processed as numpy arrays rather than Python loops.
        """
        radii = self.hierarchy.radii

        # Initialize correction matrices
        for radius in radii:
            n_reps = len(self.hierarchy.repr_zones[radius])
            self.Corr[radius] = sp.csr_matrix((n_reps, n_reps))

        # Pre-build coarse D matrices as CSR for fast row-slicing
        # and dense lookup dicts for small layers
        coarse_D: dict[float, sp.csr_matrix] = {}
        for radius in radii:
            coarse_D[radius] = self.D[radius].tocsr()

        # For each layer (except coarsest)
        for k_fine in range(len(radii) - 1):
            fine_radius = radii[k_fine]
            fine_reps = self.hierarchy.repr_zones[fine_radius]
            n_fine_reps = len(fine_reps)

            D_fine_coo = self.D[fine_radius].tocoo()
            nnz = D_fine_coo.nnz
            if nnz == 0:
                continue

            # Extract all fine (i, j) pairs as arrays
            fine_rows = D_fine_coo.row  # int array
            fine_cols = D_fine_coo.col  # int array

            # Map fine matrix indices → zone IDs (vectorized)
            fine_reps_arr = np.array(fine_reps)
            source_zones = fine_reps_arr[fine_rows]  # zone IDs for each nnz
            dest_zones = fine_reps_arr[fine_cols]

            # Track which entries still need a correction value
            corr_values = np.zeros(nnz, dtype=np.float64)
            uncorrected = np.ones(nnz, dtype=bool)

            # Check coarser layers from finest to coarsest
            for k_coarse in range(k_fine + 1, len(radii)):
                if not np.any(uncorrected):
                    break

                coarse_radius = radii[k_coarse]
                groups = self.hierarchy.groups[coarse_radius]
                ci = self.zone_indices[coarse_radius]
                D_coarse = coarse_D[coarse_radius]

                # Get indices of still-uncorrected entries
                active = np.where(uncorrected)[0]
                if len(active) == 0:
                    break

                # Vectorized group mapping for active entries
                src_z = source_zones[active]
                dst_z = dest_zones[active]

                # Map zones to coarse representatives
                # (groups is a dict, need to vectorize lookup)
                src_reps_coarse = np.array([groups[z] for z in src_z])
                dst_reps_coarse = np.array([groups[z] for z in dst_z])

                # Map coarse reps to matrix indices (if they exist)
                # Build vectorized lookup
                src_in_ci = np.array([ci.get(r, -1) for r in src_reps_coarse])
                dst_in_ci = np.array([ci.get(r, -1) for r in dst_reps_coarse])

                # Both must be valid indices
                both_valid = (src_in_ci >= 0) & (dst_in_ci >= 0)
                valid_active = active[both_valid]
                valid_src_idx = src_in_ci[both_valid]
                valid_dst_idx = dst_in_ci[both_valid]

                if len(valid_active) == 0:
                    continue

                # Batch lookup coarse interaction values from CSR matrix
                # Use direct element access via dense array for small coarse matrices
                n_coarse = D_coarse.shape[0]
                if n_coarse <= 10000:
                    # Small enough to convert to dense for fast element access
                    D_dense = D_coarse.toarray()
                    values = D_dense[valid_src_idx, valid_dst_idx]
                else:
                    # For larger matrices, use sparse element access
                    values = np.array(D_coarse[valid_src_idx, valid_dst_idx]).ravel()

                # Apply corrections where coarse value > 0
                has_value = values > 0
                corrected = valid_active[has_value]
                corr_values[corrected] = values[has_value]
                uncorrected[corrected] = False

            # Create correction matrix for this layer
            has_corr = corr_values > 0
            if np.any(has_corr):
                self.Corr[fine_radius] = sp.csr_matrix(
                    (corr_values[has_corr], (fine_rows[has_corr], fine_cols[has_corr])),
                    shape=(n_fine_reps, n_fine_reps),
                )

    def _matvec(self, activity: np.ndarray) -> np.ndarray:
        """
        Compute matrix-vector product: accessibility = InteractionMatrix × activity.

        This is the core operation for computing accessibility from activity
        distribution using the hierarchical structure efficiently.

        Algorithm (see HIERARCHIES_CONCEPTUAL_DOCUMENTATION.md):
        1. Aggregate activity to each layer: aHier[k] = G[k] × activity
        2. For each layer k:
               contrib[k] = (D[k] - Corr[k]) × aHier[k]
        3. Expand and sum: result = Σ_k G[k]ᵀ × contrib[k]

        Parameters
        ----------
        activity : np.ndarray
            Activity vector of shape (n_zones,)
            E.g., population, employment, opportunities

        Returns
        -------
        np.ndarray
            Accessibility vector of shape (n_zones,)
            Entry i is the accessibility of zone i
        """
        n_zones = self.shape[0]
        result = np.zeros(n_zones)

        # Process each layer
        for radius in self.hierarchy.radii:
            # Step 1: Aggregate activity to this layer
            aHier = self.G[radius].dot(activity)

            # Step 2: Compute contribution using precomputed net operator
            contrib = self._net[radius].dot(aHier)

            # Step 3: Expand back to zones and add to result
            result += self.G[radius].T.dot(contrib)

        return result

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        """Transpose matrix-vector product.

        The operator is symmetric for undirected graphs (D and Corr are
        symmetric, G[k]^T (D[k]-Corr[k]) G[k] is symmetric), so this
        delegates to ``_matvec``.
        """
        return self._matvec(x)

    def _matmat(self, X: np.ndarray) -> np.ndarray:
        """Efficient matrix-matrix product for multiple right-hand sides."""
        n_zones = self.shape[0]
        result = np.zeros((n_zones, X.shape[1]))

        for radius in self.hierarchy.radii:
            aHier = self.G[radius].dot(X)
            contrib = self._net[radius].dot(aHier)
            result += self.G[radius].T.dot(contrib)

        return result

    def get_row(self, source_zone: int) -> np.ndarray:
        """
        Get interaction row for a single zone.

        This computes the interaction from source_zone to all other zones,
        equivalent to matvec with a one-hot vector.

        Parameters
        ----------
        source_zone : int
            Source zone ID

        Returns
        -------
        np.ndarray
            Interaction values from source_zone to all zones
            Entry i is interaction from source_zone to zone i

        Examples
        --------
        >>> interactions = ih.get_row(0)
        >>> print(f"Interaction from zone 0 to zone 5: {interactions[5]:.4f}")
        """
        n_zones = len(self.hierarchy.zones)
        try:
            zone_index = self.hierarchy.zones.index(source_zone)
        except ValueError:
            sample = self.hierarchy.zones[:5]
            raise ValueError(
                f"source_zone {source_zone!r} not in hierarchy. Valid zones (sample): {sample}"
            )

        # Create one-hot vector
        one_hot = np.zeros(n_zones)
        one_hot[zone_index] = 1.0

        # Use matvec
        return self.matvec(one_hot)

    def get_density(self) -> float:
        """
        Calculate storage density of interaction matrices.

        Returns
        -------
        float
            Fraction of interactions stored vs. dense matrix
        """
        n_zones = len(self.hierarchy.zones)
        dense_size = n_zones * n_zones

        # Count non-zeros in all D matrices
        stored_count = sum(self.D[r].nnz for r in self.hierarchy.radii)

        return stored_count / dense_size if dense_size > 0 else 0.0

    def save(
        self,
        path,
        *,
        include_network: bool = False,
        backend: str = "auto",
        interaction_fn_hint: str = "",
    ):
        """Save this interaction hierarchy to a ``.npz`` or ``.h5`` file.

        Parameters
        ----------
        path : str or Path
            Destination path.  The appropriate extension is appended if missing.
        include_network : bool, default False
            Whether to embed the NetworkX graph.
        backend : str, default "auto"
            ``"auto"``, ``"npz"``, or ``"hdf5"``.
        interaction_fn_hint : str, default ""
            Human-readable description of the interaction function.

        Returns
        -------
        Path
            The path the file was written to.
        """
        from hierx.storage import save_interaction

        return save_interaction(
            self,
            path,
            include_network=include_network,
            backend=backend,
            interaction_fn_hint=interaction_fn_hint,
        )

    @classmethod
    def load(cls, path, *, interaction_fn=None, network=None):
        """Load an InteractionHierarchy from a saved file.

        Parameters
        ----------
        path : str or Path
            File to load.
        interaction_fn : callable or None, optional
            Interaction function to attach.
        network : nx.Graph or None, optional
            If provided, used as the network.

        Returns
        -------
        InteractionHierarchy
        """
        from hierx.storage import load_interaction

        return load_interaction(path, interaction_fn=interaction_fn, network=network)

    def get_layer_info(self) -> list[dict]:
        """
        Get information about interaction matrices at each layer.

        Returns
        -------
        List[Dict]
            Layer information with keys:
            - 'radius': layer radius
            - 'n_representatives': number of representatives
            - 'n_interactions': number of stored interactions
            - 'n_corrections': number of correction terms
        """
        info = []
        for radius in self.hierarchy.radii:
            info.append(
                {
                    "radius": radius,
                    "n_representatives": len(self.hierarchy.repr_zones[radius]),
                    "n_interactions": self.D[radius].nnz,
                    "n_corrections": self.Corr[radius].nnz,
                }
            )
        return info
