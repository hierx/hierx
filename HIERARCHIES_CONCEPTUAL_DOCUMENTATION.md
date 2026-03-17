# Hierarchical Cost and Interaction Matrices - Conceptual Documentation

## Overview

The hierarchical implementation provides an efficient method for computing and storing cost matrices and interaction matrices for large-scale spatial networks. The primary goals are:

1. **Computational Efficiency**: Reduce the number of shortest path calculations from O(n²) to a manageable subset
2. **Memory Efficiency**: Store only necessary costs rather than full dense matrices
3. **Scalability**: Enable analysis of country- and continent-scale networks with many hundreds of thousands of zones

The implementation achieves this through a multi-level hierarchical grouping strategy where zones are progressively aggregated into coarser representations.

**Units convention**: Throughout this documentation, cost values represent travel times in minutes (or generalized impedance units in multimodal contexts). All hierarchical layers preserve the same cost units. Interaction values are dimensionless (probability or intensity of interaction).

## Glossary of Key Terms

- **Zone (z)**: A spatial unit representing a location in the network. The fundamental unit of analysis.
- **Representative (R_k)**: At hierarchical layer k, a subset of zones that represent groups. In notation: z ∈ R_k means zone z is a representative at layer k.
- **Layer/Level (k)**: A hierarchical level characterized by radius r_k. Layers are ordered from finest (k=0) to coarsest.
- **Group**: All zones sharing the same representative at a given layer. Forms a spatial cluster.
- **Group membership matrix (G_k)**: Sparse boolean matrix mapping zones to representatives at layer k. G_k[r,z] = 1 if zone z belongs to representative r's group.
- **Cost (C_k)**: Network travel cost (time/distance) between representatives at layer k. Stored sparsely.
- **Interaction matrix (D_k)**: Interaction values between representatives at layer k, computed as D_k = f(C_k) where f is the distance-decay function.
- **Correction matrix (Corr_k)**: Subtracts coarse-layer interactions already counted in finer layers. Ensures each zone pair contributes exactly once.
- **Cutoff**: Maximum cost beyond which pairs are not computed at a given layer. Controls sparsity.
- **Base radius (base_radius, manuscript: ρ₀)**: The radius of the finest hierarchical layer (e.g., 5000 meters).
- **Growth factor (increase_factor, manuscript: γ)**: Multiplier for layer radii (typically 2): r_{k+1} = increase_factor × r_k
- **Overlap factor (overlap_factor, manuscript: α)**: Controls cutoff size relative to layer radius. cutoff_k = overlap_factor × r_k

## Visual Structure

### Hierarchical Layer Organization

```
Layer r₀ (finest):    [A] [B] [C] [D] [E] [F] [G] [H]
                       All zones are representatives

Layer r₁:             [A━━━B━━━C] [D━━━E] [F━━━G━━━H]
                       3 representatives: A, D, F
                       Groups: {A,B,C}, {D,E}, {F,G,H}

Layer r₂ (coarsest):  [A━━━━━━━━━━━━━B━━━━━D━━━━━E━━━━━F━━━━━━━━━G━━━━━H]
                       1 representative: A
                       Group: {A,B,C,D,E,F,G,H}

Arrows show group membership:
     ↓                     ↓          ↓      ↓           ↓
     B belongs to A's      E to D     G to F  H to F
     group at layer r₁
```

### Cost Storage Pattern

```
For layer r₁ with cutoff:

Representative A can reach:
    A -> A : cost = 0
    A -> B : cost = c_AB  (within cutoff)
    A -> C : cost = c_AC  (within cutoff)
    A -> D : cost = c_AD  (within cutoff)
    A -> F : NOT STORED   (beyond cutoff)

Representative D can reach:
    D -> D : cost = 0
    D -> A : cost = c_DA  (within cutoff)
    D -> E : cost = c_DE  (within cutoff)
    D -> F : cost = c_DF  (within cutoff)

Sparse storage: Only costs within cutoff are computed and stored
```

## Worked Numerical Example

To illustrate the hierarchical approach concretely, consider a toy network with 4 zones:

### Network Topology

```
Linear network:  A ──10── B ──15── C ──20── D

Network costs (minutes):
  A→B: 10    B→A: 10
  B→C: 15    C→B: 15
  C→D: 20    D→C: 20

All-pairs shortest path costs:
     A    B    C    D
A [  0   10   25   45 ]
B [ 10    0   15   35 ]
C [ 25   15    0   20 ]
D [ 45   35   20    0 ]
```

### Hierarchical Structure

**Layer r₀ (finest, radius = 10):**

- Representatives: R₀ = {A, B, C, D} (all zones)
- Cutoff: 15 (= 1.5 × 10)
- Costs computed within cutoff:

```
C₀ stored (only pairs within cutoff = 15):
     A    B    C    D
A [  0   10   -    - ]
B [ 10    0   15   - ]
C [  -   15    0   - ]
D [  -    -    -   0 ]

Dash (-) means not stored (beyond cutoff)
```

**Layer r₁ (coarse, radius = 20):**

- Group formation: B and C are close (15 < 20/2), so C joins B's group. A and D become representatives.
- Representatives: R₁ = {A, B, D}
- Cutoff: None (coarsest layer, all pairs computed)
- Costs:

```
C₁ (all pairs, using representatives):
     A    B    D
A [  0   10   45 ]
B [ 10    0   35 ]
D [ 45   35    0 ]

Note: Zone C is represented by B at this layer
```

### Interaction Computation

**Interaction function**: f(c) = (c + 5)^(-1)  [power law with offset]

**Layer r₀ interactions**:

```
D₀ = f(C₀):
     A      B      C      D
A [ 1.00   0.067  -      -   ]
B [ 0.067  1.00   0.050  -   ]
C [ -      0.050  1.00   -   ]
D [ -      -      -      1.00]
```

**Layer r₁ interactions**:

```
D₁ = f(C₁):
     A      B      D
A [ 1.00   0.067  0.020]
B [ 0.067  1.00   0.025]
D [ 0.020  0.025  1.00 ]
```

### Correction Matrix

At layer r₀, we need corrections to avoid double-counting:

```
Corr₀: For each pair (i,j) in r₀, subtract the coarse interaction from r₁

Example: Pair (A,B) appears in both layers
  - In D₁: A and B interact with value 0.067
  - In D₀: A and B interact with value 0.067
  - Correction: Corr₀[A,B] = D₁[A,B] = 0.067

Full Corr₀:
     A      B      C      D
A [  0     0.067  0      0   ]
B [ 0.067  0      0.050  0   ]
C [  0     0.050  0      0   ]
D [  0     0      0      0   ]

Explanation:
- Corr₀[A,B] = 0.067: Remove A-B interaction from r₁ (it's in r₀)
- Corr₀[B,C] = 0.050: Remove B-C interaction from r₁ (it's in r₀)
- Corr₀[C,C] = 0: Zone C is represented by B at r₁, so no correction needed
- Pairs not in D₀ (like A-D) have no correction at r₀
```

### Group Membership Matrices

```
G₀ (all zones represent themselves):
     A   B   C   D
A [  1   0   0   0 ]
B [  0   1   0   0 ]
C [  0   0   1   0 ]
D [  0   0   0   1 ]

G₁ (C belongs to B's group):
     A   B   C   D
A [  1   0   0   0 ]
B [  0   1   1   0 ]  ← B represents both B and C
D [  0   0   0   1 ]
```

### Matrix-Vector Product Example

Given activity vector: a = [5, 10, 8, 3]ᵀ (population in thousands)

**Step 1: Aggregate to each layer**

```
aHier₀ = G₀ × a = [5, 10, 8, 3]ᵀ  (unchanged, all are representatives)

aHier₁ = G₁ × a = [5, 18, 3]ᵀ     (B's group has 10+8=18)
                   A   B   D
```

**Step 2: Compute contributions at each layer**

```
Layer r₀:
  prod₀ = D₀ × aHier₀ = [5.67, 10.95, 8.50, 3.00]ᵀ
  corr₀ = Corr₀ × aHier₀ = [1.00, 0.74, 0.67, 0]ᵀ
  contrib₀ = prod₀ - corr₀ = [4.67, 10.21, 7.83, 3.00]ᵀ

Layer r₁:
  prod₁ = D₁ × aHier₁ = [1.695, 1.585, 0.625]ᵀ  (for reps A, B, D)
  corr₁ = 0 (no corrections at coarsest layer)
  contrib₁ = [1.695, 1.585, 0.625]ᵀ
```

**Step 3: Expand and sum**

```
From layer r₀ (already at zone level):
  result₀ = contrib₀ = [4.67, 10.21, 7.83, 3.00]ᵀ

From layer r₁ (expand using G₁ᵀ):
  result₁ = G₁ᵀ × contrib₁ = [1.695, 1.585, 1.585, 0.625]ᵀ
                               A      B      C      D
            (C gets B's value: 1.585)

Final accessibility:
  result = result₀ + result₁ = [6.36, 11.80, 9.42, 3.63]ᵀ
```

**Interpretation**: Zone B has highest accessibility (11.80) because it:

- Has high local activity (10 in its own zone)
- Is near zone C (8 activity, 15 minutes away)
- Benefits from being central in the network

This example demonstrates:

1. ✓ How zones are grouped into representatives at coarser layers
2. ✓ How sparse storage works (only nearby pairs at fine layers)
3. ✓ How correction matrices prevent double-counting
4. ✓ How activity aggregates and then expands through layers
5. ✓ Why the hierarchy is exact for stored costs, approximate for inherited ones

## Key Concepts

### Zones and Representatives

- **Zone**: A spatial unit in the network - the fundamental unit of analysis
- **Representative**: At each hierarchical level, some zones act as representatives for groups of zones
- **Group**: All zones that share the same representative at a given level belong to the same group

At the finest level, all zones represent themselves (each zone is its own representative). At coarser levels, fewer zones act as representatives, with each representing a spatial cluster of zones.

### Hierarchical Levels

The hierarchy consists of multiple **layers** (or **levels**), each characterized by a radius parameter `r`:

- **Finest layer** (smallest `r`): All zones are representatives
- **Intermediate layers**: Progressively fewer representatives
- **Coarsest layer** (largest `r`): Fewest representatives, typically covering the entire spatial extent

Layers are constructed with exponentially increasing radii using an `increase_factor` (typically 2), starting from a `base_radius` (e.g., 5000 meters).

## The Hierarchy Class

The `Hierarchy` class implements the hierarchical cost matrix representation.

### Construction Algorithm

The hierarchy is built progressively from fine to coarse through the following process:

#### 1. Layer Generation

Starting from `base_radius`, layers are created with radii:

```
r₀ = base_radius
r₁ = r₀ × increase_factor
r₂ = r₁ × increase_factor
...
```

Construction continues until the radius exceeds 1.5× the estimated spatial extent of the network (estimated by sampling shortest paths). A minimum of 2 layers is always generated.

#### 2. Representative Selection (Greedy Algorithm with Takeover)

For each layer beyond the finest:

1. **Start with representatives from previous (finer) layer**
2. **For each candidate zone**, check if there's an existing representative within `cutoff = r/2`
3. **If a close representative exists**: The candidate joins the **closest** representative's group
4. **If no close representative exists**: The candidate becomes a new representative
5. **Takeover mechanism**: When a new representative is created, it checks all non-representative zones and "takes over" any zone for which it is closer than the zone's current representative

**Key principle**: Each zone is always assigned to its **closest** representative at each layer. The takeover mechanism ensures representatives are centrally located within their groups.

This is a **path-dependent, greedy algorithm** where the order of processing affects the final grouping. Note that the grouping threshold (`r/2`) is distinct from the cost computation cutoff (`overlap_factor × r`) described in the next section.

#### 3. Cost Calculation with Cutoff

For each layer `r`, costs (shortest path lengths) are computed:

- **Source zones**: All representatives at this layer
- **Destination zones**: All representatives at this layer (only)
- **Cutoff**: `overlap_factor × r` (or `None` for the coarsest layer)

**Hierarchical storage principle**: At each layer, costs are stored **only between representatives** at that layer, not from representatives to non-representative group members. This is fundamental to efficiency:

- Layer with n_k representatives: stores O(n_k²) costs (within cutoff)
- NOT O(n_k × n) costs to all zones

Costs are only stored if they are within the cutoff distance, creating sparse cost dictionaries. This dramatically reduces the number of shortest path calculations compared to a full O(n²) dense matrix.

#### 4. Representative Propagation

After the main hierarchy is constructed, representative assignments are **propagated** from coarser to finer layers using the **stored relationships** from the construction phase. This ensures:

1. Every zone knows its representative at every layer
2. **Group nesting property**: Groups at finer layers are subsets of groups at coarser layers
3. Transitivity: If zone A belongs to representative B at layer k, and B belongs to representative C at layer k+1, then A belongs to C at layer k+1

**Critical**: Propagation must use the stored representative assignments from construction, not recalculate them. Recalculation can violate the nesting property required for correct correction matrices.

#### 5. Within-Group Cost Gap Filling

After grouping at a coarser layer, two fine-layer representatives may end up in the same coarse group despite being beyond the fine layer's cost cutoff. Without correction, `get_cost()` would fall through to the coarse layer and return the coarse representative's self-cost (0). To prevent this, a bounded Dijkstra search fills in missing cost pairs between fine-layer representatives that share a coarse-layer group. This step runs with `cutoff = coarse_radius` and typically adds a small number of costs.

#### Summary of Key Principles

The construction algorithm ensures four critical properties:

1. **Representative centrality**: Each zone is assigned to its closest representative (via takeover mechanism)
2. **Group nesting**: Finer groups are subsets of coarser groups (via propagation using stored relationships)
3. **Hierarchical storage**: Only costs between representatives are stored at each layer
4. **Sparsity via cutoff**: Only costs within cutoff distance are computed, ensuring scalability

These properties work together to guarantee correctness of the correction matrices and efficiency of the hierarchical approach.

### Cost Lookup

The `get_cost(source_zone, dest_zone)` method returns the **approximate** cost between any two zones via hierarchical representative mapping:

1. **Iterate through layers from finest to coarsest**
2. **For each layer**, find the representatives for source and destination
3. **Check if** the cost between these representatives exists at this layer
4. **Return the first cost found** (the finest-level approximation available)

```
Algorithm: get_cost(source_zone, dest_zone)
──────────────────────────────────────
for each layer k from finest to coarsest:
    repr_i = representative of zone_i at layer k
    repr_j = representative of zone_j at layer k

    if cost(repr_i, repr_j) exists at layer k:
        return cost(repr_i, repr_j)

└─ At coarsest layer, all pairs exist
   └─ Always returns a cost
```

The algorithm guarantees finding a cost because:

- All zones are represented at all levels
- The coarsest layer has costs computed between all representative pairs (without cutoff)
- Therefore, at worst, the coarsest layer provides the cost

### Data Structure

The `Hierarchy` class stores:

- `zones`: List of all zone IDs
- `repr_zones[r]`: List of representative zone IDs at each layer `r`
- `costs[r]`: Dictionary of costs at layer `r`: `{source_id: {dest_id: cost, ...}, ...}`
- `cutoffs[r]`: The cutoff used for cost calculations at each layer
- `groups[r][zone_id]`: For each zone, its representative at each layer `r`

## The InteractionHierarchy Class

The `InteractionHierarchy` class builds upon the `Hierarchy` to compute interaction matrices efficiently. It applies a distance-decay function (interaction function) to the costs and handles the hierarchical aggregation correctly.

### Interaction Function

A distance-to-interaction function `f(cost)` transforms costs into interaction values. Common forms include:

- **Power law**: `f(c) = a × (c + offset)^b` where `b < 0` for decay (see `power_law_interaction(cost, a=1.0, b=-2.0, offset=5000.0)`)
- **Exponential**: `f(c) = exp(-a × c)` (see `exponential_interaction(cost, a=1e-3)`)

The interaction represents the probability or intensity of interaction between zones as a function of travel cost.

### Mathematical Notation

Let:

- `n` = total number of zones
- `K` = set of hierarchical layers (indices k), ordered from finest to coarsest
- `R_k` = set of representative zones at layer k
- `n_k` = |R_k| = number of representatives at layer k
- `C_k[i,j]` = cost between representatives i and j at layer k (sparse)
- `D_k[i,j]` = interaction = f(C_k[i,j]) at layer k (called `F_k` in the manuscript)
- `G_k[r,z]` = group membership: 1 if zone z is in representative r's group at layer k, else 0
- `Corr_k[i,j]` = correction term at layer k

### Key Challenge: Avoiding Double-Counting

When computing interactions hierarchically, a naive approach would sum contributions from all layers. However, this leads to **double-counting**: a zone pair might be represented at multiple layers, and we must avoid counting their interaction multiple times.

**Prerequisite for correction matrices**: The **group nesting property** is fundamental. If two zones are in the same group at a finer layer, they must remain in the same group at all coarser layers. This ensures that corrections can properly identify which interactions are redundant. Violating this property leads to incorrect accessibility calculations.

```
Example of double-counting problem:

Zones A and B at layer r₁:
    A and B both represented by representative X
    D[r₁][X,X] includes A-B interaction

Zones A and B at finer layer r₀:
    A and B are their own representatives
    D[r₀][A,B] includes A-B interaction

Without correction: A-B interaction counted TWICE!
```

### Correction Matrices

The solution is **correction matrices** that subtract the contributions already accounted for in finer layers:

For each layer `k`:

1. Compute the **interaction matrix** `D[k]` by applying `f(cost)` to the costs between representatives
2. Compute a **correction matrix** `Corr[k]` that removes interactions already counted in finer layers

The correction matrices are built by examining all finer layers and identifying which zone pairs have already had their interaction computed at a finer resolution.

**Mathematical Formulation:**

For layer k and finer layers k' < k (where k' is finer):

```
Corr[k][i,j] = Σ (over all k' < k where i,j not yet corrected in k' < k'' < k)
               D[k][repr_k(i), repr_k(j)]
```

Where:

- `repr_k(i)` = representative of zone i at layer k
- The sum includes contribution from coarse layer k for all zone pairs (i,j) that:
  - Have representatives at layer k whose interaction exists
  - Have NOT been corrected by an intermediate layer k' < k'' < k

This ensures each zone pair's interaction is counted exactly once at the finest available resolution.

**Algorithmic Implementation:**

```
Algorithm: Build Correction Matrices
─────────────────────────────────────────────────
Input: Layers K = {k₀, k₁, ..., k_m} from finest to coarsest
       Interaction matrices D[k] for each layer k

Output: Correction matrices Corr[k] for each layer k

Initialize: Corr[k] = empty sparse matrix for all k

For each layer k in K (from coarsest to finest):
    For each finer layer k' where k' < k:
        For each zone pair (i, j) in layer k':
            repr_i = representative of i at layer k
            repr_j = representative of j at layer k

            # Check if coarse layer has this pair
            if (repr_i, repr_j) exists in D[k]:

                # Check if pair already corrected in intermediate layers
                already_corrected = False
                for k'' in (layers between k and k'):
                    if (i, j) exists in D[k'']:
                        already_corrected = True
                        break

                # Add correction term if not already handled
                if not already_corrected:
                    Corr[k'][i, j] += D[k][repr_i, repr_j]

Return Corr
```

**Enforcement Mechanism:**

The "exactly once" property is enforced through a vectorized implementation in `_build_correction_matrices()`. For each layer pair (k, k'), a boolean mask tracks which representative pairs have already been corrected at an intermediate layer. This numpy array (`uncorrected`) is checked per coarser layer to prevent double-counting without explicitly storing a set of counted zone pairs.

### Matrix-Vector Product (Computing Accessibility)

The hierarchical interaction matrix acts as a **linear operator** that can multiply activity vectors efficiently.

Given an activity (or attraction) vector `a` representing the activity at each zone, we want to compute:

```
accessibility = InteractionMatrix × a
```

This gives the accessibility of each zone (how much activity it can access).

#### Algorithm

1. **Aggregate activity to each layer**: For each layer, sum the activity of all zones within each representative's group, creating `aHier[k]`

```
Aggregation step:
                           Layer k representatives
                           ┌────┬────┬────┐
Activity by zone:          │ A  │ D  │ F  │
[a₁,a₂,a₃,a₄,a₅,a₆,a₇,a₈]  └────┴────┴────┘
 └─A──┘ └─D─┘ └───F────┘
  a₁+a₂  a₃+a₄  a₅+a₆+a₇+a₈   = aHier[k]
```

1. **For each layer** (from finest to coarsest):
  - Compute the contribution: `prod[k] = D[k] × aHier[k]`
  - Compute the correction: `corr[k] = Corr[k] × aHier[k]`
  - Compute the net contribution: `contrib[k] = prod[k] - corr[k]`
2. **Expand and sum**: Use group matrices (`G[k]`) to expand each layer's contribution back to individual zones and sum across all layers:

```
Expansion step:
                    Layer k result (representatives)
                    ┌────┬────┬────┐
                    │ r_A│ r_D│ r_F│  = contrib[k]
                    └────┴────┴────┘
                       ↓    ↓    ↓
G[k]ᵀ maps back:   [z₁,z₂,z₃,z₄,z₅,z₆,z₇,z₈]
                    └─A──┘ └─D─┘ └───F────┘
                    r_A for r_D for  r_F for all
                    zones   zones    zones in F
                    in A    in D
```

**Mathematical Formula:**

```
aHier[k] = G[k] × a               (aggregate to representatives)
contrib[k] = (D[k] - Corr[k]) × aHier[k]   (compute at layer k)
result = Σ_{k∈K} G[k]ᵀ × contrib[k]        (expand and sum)
```

The **group matrices** `G[k]` are sparse boolean matrices where `G[k][r,z] = 1` if zone z is represented by representative r at layer k, else 0.

### Approximation Quality

The hierarchical approach is an **approximation** of the true dense interaction matrix:

- **Exact** for zone pairs whose representatives have computed costs
- **Approximate** for zone pairs whose costs are inherited from coarser layers
- The approximation is controlled by the `overlap_factor` and `base_radius` parameters

```
Approximation hierarchy:
┌─────────────────────────────────────────┐
│ True dense matrix: I[i,j] = f(c[i,j])   │ ← Ground truth (expensive)
└─────────────────────────────────────────┘
                  ↓ approximated by
┌─────────────────────────────────────────┐
│ Hierarchical matrix: H × a ≈ I × a      │ ← Efficient approximation
└─────────────────────────────────────────┘

Quality depends on:
- overlap_factor (larger = more stored costs = better)
- base_radius (smaller = finer initial resolution = better)
- Network structure (hierarchical networks approximate better)
```

The correction matrices ensure mathematical consistency: the hierarchical computation produces the correct result for the costs that are actually stored, with approximations only for cost values interpolated from coarser layers.

## Edge Cases and Special Handling

### Disconnected Networks

**Behavior**: If the network contains disconnected components:

- Costs between disconnected zones remain `inf`
- Interaction function should handle `f(inf) = 0` appropriately
- Zones in different components may be assigned to different groups at coarser layers
- Correction matrices handle infinite costs gracefully (not added to corrections)

### Non-Spatial Networks

**Applicability**: The hierarchy is optimized for spatial networks with:

- Approximate metric properties (triangle inequality roughly holds)
- Spatial clustering (nearby zones are "similar")

For non-spatial networks (e.g., social networks):

- The cutoff strategy may be less effective
- Representative selection may not capture network structure well
- Consider network-specific clustering algorithms instead

### Asymmetric Costs

**Current assumption**: The implementation assumes **symmetric costs**: cost(i,j) = cost(j,i).
The underlying network must be an undirected `nx.Graph`.

Note: costs are stored in both directions (`costs[A][B]` and `costs[B][A]`),
so the current storage format already accommodates asymmetric values.
The symmetry assumption affects representative selection (group membership
based on nearest neighbor) and the undirected Dijkstra calls.

**Extension to asymmetric costs**: The correction logic would still hold
mathematically, but:

- Dijkstra must run in both directions (or on a directed graph)
- Cutoff logic needs direction-specific handling
- Representative selection may need revision (e.g. bidirectional centrality)

### Self-Interaction

Controlled by the `self_interaction` parameter of
`InteractionHierarchy(hierarchy, interaction_fn, self_interaction=True)`:

- If `False`: diagonal set to 0 at finest layer (zone cannot interact with itself)
- If `True` (default): diagonal follows normal interaction function

## Parameters and Tuning

### Key Parameters

- `**base_radius`** (default 5000): The radius of the finest hierarchical layer. Smaller values create finer initial groupings.
- `**increase_factor**` (default 2): Controls how rapidly layers grow. Higher values create fewer, more distinct layers.
- `**overlap_factor**` (default 1.5): Controls the cost computation cutoff relative to layer radius (`cutoff = overlap_factor × r`). Higher values store more costs per layer, improving approximation quality but increasing computation and memory. Note: this is separate from the grouping threshold (`r/2`) used during representative selection.
- `**min_representatives**` (default 3): Minimum number of representatives required for a layer to be retained. When a layer has fewer representatives, it is pruned and the previous layer becomes the coarsest (recomputed without cutoff).

### Tuning Guidance

Empirical findings from the manuscript (1800 parameter configurations tested):

- `**overlap_factor` is the primary accuracy knob**: increasing from 1.0 to 3.0 reduces median error ~7×. Recommended: ≥ 2.0 for production use.
- `**base_radius`**: should be at least 3× the mean edge cost for good accuracy. Values too small create excessive layers without proportional accuracy gains.
- `**increase_factor**`: 2 is recommended in most cases. Higher values reduce layer count but can create coarse jumps.
- **Larger networks** (> 10,000 zones): increase `base_radius` to keep per-layer representative counts manageable.

## Computational Complexity

For a network with `n` zones:

- **Dense approach**:
  - Shortest path calculations: O(n²)
  - Memory: O(n²)
  - Matrix-vector product: O(n²)
- **Hierarchical approach**:
  - **Construction** (one-time):
    - Shortest path calculations: O(n × k × c) where:
      - k = number of layers ≈ log₂(spatial_extent/base_radius)
      - c = average number of zones within cutoff
    - Correction matrix computation: O(n × k² × c)
    - Total construction: O(n × k² × c)
  - **Storage** (persistent):
    - Cost dictionaries: O(n × k × c)
    - Interaction matrices D[k]: O(n × k × c)
    - Correction matrices Corr[k]: O(n × k × c_corr) where c_corr ≤ c
    - Group matrices G[k]: O(n × k) (sparse boolean)
  - **Matrix-vector product** (repeated operations):
    - Aggregation: O(n × k)
    - Layer products: O(n × k × c)
    - Expansion: O(n × k)
    - Total per product: O(n × k × c)
  - Typically reduces computations and memory by 1-2 orders of magnitude for large networks

```
Complexity comparison:

Dense:        n² operations, n² memory
              ████████████████████████████████

Hierarchical: n·k·c operations, n·k·c memory
              ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░

For typical large network (n=10,000, k=6, c=50):
Dense:        100,000,000 operations
Hierarchical: 3,000,000 operations (~30x reduction)
```

### Empirical Scaling

**Correction matrix sparsity**: In practice, correction matrices tend to be sparse:

- Typical density: 1-5% of representative pairs at each layer
- Memory overhead: ~10-20% of base interaction matrices
- Does not threaten O(n log n) overall scaling

**Parallelization**:

- **Currently parallelized**: Shortest path calculations (embarrassingly parallel)
  - scipy backend: shared-memory or fork-based process pools (Linux); serial fallback on Windows/macOS
  - networkx backend: process-based multiprocessing
  - graph-tool backend: process-based multiprocessing
- **Not currently parallelized**: Correction matrix construction, matrix-vector products
- **GPU suitability**: Sparse matrix-vector products are GPU-friendly but not yet implemented

### Benchmarking

*Note: The following are typical observed behaviors. Specific performance depends on network topology, spatial distribution, and hardware.*

**Expected timing (single-threaded, scipy backend):**

```
n zones    Dense time    Hierarchical time    Speedup
─────────  ────────────  ───────────────────  ────────
1,000      ~30s          ~3s                  ~10x
10,000     ~50min        ~3min                ~17x
100,000    ~2-3 days     ~2-3 hours           ~20-30x
```

**Memory usage:**

```
n zones    Dense memory    Hierarchical memory   Reduction
─────────  ──────────────  ────────────────────  ──────────
1,000      ~8 MB           ~1 MB                 ~8x
10,000     ~800 MB         ~50-100 MB            ~10-15x
100,000    ~80 GB          ~3-5 GB               ~20-25x
```

*These are order-of-magnitude estimates. Actual performance varies with network properties.*

## Backend Support

The shortest path calculations can be performed using different backends:

- **scipy** (default): C-optimized Dijkstra on CSR matrices via `scipy.sparse.csgraph.dijkstra`. Supports process-parallel construction using shared-memory pools (Linux) or fork-based pools (fallback). This is the fastest backend and is always available.
- **networkx**: Pure-Python reference implementation via `nx.single_source_dijkstra_path_length`. Supports process-parallel construction via `multiprocessing`. Useful for guaranteed cross-platform parallelism.
- **graph-tool**: C++/Boost backend (optional, requires separate install). Supports process-parallel construction via `multiprocessing`.
- **auto**: Resolves to scipy.

The hierarchy construction is backend-agnostic and will use the configured backend transparently. Set the backend via the `backend` parameter of the `Hierarchy` constructor.

## Theoretical Foundation

The hierarchical approach is based on exploiting the spatial structure of networks:

### Key Insight

Spatial networks exhibit **distance decay**: interactions between nearby zones are typically stronger and more important than those between distant zones. The hierarchy exploits this by:

1. Storing detailed costs for nearby zones (fine layers)
2. Using coarser approximations for distant zones (coarse layers)
3. Ensuring mathematical consistency through correction terms

### Correctness Guarantee

The hierarchical interaction computation is exact for the stored costs and provides controlled approximation for inherited costs:

**Theorem (informal):** For any activity vector `a`, the hierarchical computation satisfies:

```
H × a = Σ_{k∈K} G[k]ᵀ × (D[k] - Corr[k]) × G[k] × a
```

Where each zone pair contributes exactly once at the finest available resolution through the correction mechanism.

This can be understood as a **sparse-plus-correction decomposition**:

- The finest available interaction for each pair is included once (sparse term)
- Coarse-layer contributions are subtracted where finer data exists (correction term)
- The result approximates the full dense matrix `I` where `I[i,j] = f(cost[i,j])`

### Approximation Error Characterization

The approximation error for a zone pair (i, j) depends on:

1. **Layer of first appearance**: Pairs appearing in finer layers have lower error
2. **Cost inheritance**: When cost(i,j) is inherited from representatives at a coarser layer k:
  ```
   error(i,j) ≈ |f(cost[i,j]) - f(cost[repr_k(i), repr_k(j)])|
  ```
3. **Network geometry**: Error is typically larger when:
  - Zones i and j are near group boundaries
  - The network has high local cost variation
  - Representatives are not centrally located

**Error bounds** (see manuscript for formal treatment):

If the decay function f has Lipschitz constant L_f, and the maximum distance from any zone to its representative at layer k is δ_k, then the per-pair error is bounded by:

```
|f(c_ij) - f(c_{rep_i, rep_j})| ≤ 2 L_f δ_k
```

**Expected error behavior:**

- For pairs within cutoff at finest layer: error = 0 (exact)
- For pairs first appearing at layer k: error bounded by the Lipschitz constant and representative displacement δ_k
- Empirically: with `base_radius` ≥ 3× mean edge cost and `overlap_factor` ≥ 2, RMSE stays below 2.5% up to n = 5000 (see manuscript Table 3)

### Design Principles

1. **Spatial clustering**: Group nearby zones to reduce dimensionality at coarser scales
2. **Sparse computation**: Only compute costs within meaningful distances (cutoffs)
3. **Multi-resolution**: Store costs at multiple spatial scales
4. **Correction for consistency**: Subtract coarse-layer contributions when fine-layer data available
5. **Linear operator interface**: Enable efficient matrix-vector products without materializing dense matrices

### Relation to Other Methods

The hierarchical approach shares conceptual similarities with several established techniques (see manuscript Section 2 for detailed positioning):

```
┌──────────────────┬──────────────────┬──────────────┬──────────────────────┐
│ Method           │ Domain           │ Scaling      │ Key Mechanism        │
├──────────────────┼──────────────────┼──────────────┼──────────────────────┤
│ This work        │ Networks         │ O(n log n)   │ Spatial clustering + │
│ (Hierarchies)    │ (spatial)        │ (sparse)     │ corrections          │
├──────────────────┼──────────────────┼──────────────┼──────────────────────┤
│ Fast Multipole   │ N-body           │ O(n)         │ Multipole expansions │
│                  │ (continuous)     │              │                      │
├──────────────────┼──────────────────┼──────────────┼──────────────────────┤
│ H-matrices       │ Integral eqs.    │ O(n log² n)  │ Low-rank blocks      │
├──────────────────┼──────────────────┼──────────────┼──────────────────────┤
│ Nyström approx.  │ Kernel matrices  │ O(nk²)       │ Landmark sampling    │
├──────────────────┼──────────────────┼──────────────┼──────────────────────┤
│ Distance cutoff  │ Spatial networks │ O(n·c)       │ Truncation           │
├──────────────────┼──────────────────┼──────────────┼──────────────────────┤
│ Distance oracles │ Shortest paths   │ O(n^{1+ε})   │ Precomputed lookups  │
├──────────────────┼──────────────────┼──────────────┼──────────────────────┤
│ Multigrid        │ PDEs             │ O(n)         │ Coarse-grid corr.    │
│                  │ (continuous)     │              │                      │
└──────────────────┴──────────────────┴──────────────┴──────────────────────┘
```

Key differences from the closest alternatives:

- **vs FMM/H-matrices**: Those exploit smoothness in continuous space; this approach operates on discrete network topology where costs follow graph shortest paths, not Euclidean distance
- **vs Nyström**: Nyström approximates the kernel matrix via landmark sampling and assumes approximate low-rank structure; fails on steep decay kernels where the interaction matrix is far from low-rank (manuscript shows ~74% error on steep power-law kernels)
- **vs distance cutoff**: Cutoff truncates beyond a threshold; the hierarchy captures all scales via multi-resolution, avoiding the hard accuracy/work tradeoff
- **vs distance oracles/contraction hierarchies**: Those accelerate individual shortest-path queries; the hierarchy approximates the entire interaction operator for repeated matrix-vector products

## Summary of Data Structures

### Hierarchy Class

```
Primary structures:
├─ zones: list[zone_id]                   # All zones
├─ repr_zones[r]: list[zone_id]           # Representatives at layer r
├─ costs[r][source][dest] = cost          # Sparse cost storage
├─ cutoffs[r] = cutoff_distance           # Cutoff used at layer r
└─ groups[r][zone_id] = repr_id           # Group membership
```

### InteractionHierarchy Class

```
Primary structures:
├─ D[k]: sparse matrix (n_k × n_k)        # Interaction at layer k
├─ Corr[k]: sparse matrix (n_k × n_k)    # Correction at layer k
├─ G[k]: sparse matrix (n_k × n)          # Group membership
└─ zone_indices[k][zone_id] = idx          # Zone index at layer k

Matrix dimensions:
- n = total zones
- n_k = representatives at layer k
- G[k] is rectangular: maps n zones to n_k representatives
- D[k] and Corr[k] are square: n_k × n_k
```

## Validation and Testing

### Unit Tests

**Test coverage** (see `tests/` directory):

- Comparison tests: Verify hierarchical vs. dense results match within threshold

**Validation approach:**

```python
from numpy.linalg import norm
from hierx import (
    Hierarchy, InteractionHierarchy,
    compute_dense_cost_matrix, compute_dense_interaction_matrix,
)

def test_hierarchy_accuracy(network, interaction_fn, threshold=0.01):
    # Build hierarchy
    h = Hierarchy(network, base_radius=5000, increase_factor=2, overlap_factor=2.0)
    ih = InteractionHierarchy(h, interaction_fn)

    # Build dense for comparison
    cost_matrix = compute_dense_cost_matrix(network, h.zones)
    dense_interaction = compute_dense_interaction_matrix(cost_matrix, interaction_fn)

    # Compare matrix-vector products
    activity = np.ones(len(h.zones))
    hierarchical_result = ih.matvec(activity)
    dense_result = dense_interaction @ activity

    relative_error = norm(hierarchical_result - dense_result) / norm(dense_result)
    assert relative_error < threshold
```

### Visual Diagnostics

**Recommended diagnostic plots:**

1. **Representative density by layer:**
  ```
   Layer    Representatives    Density
   r₀       10,000            100%
   r₁       2,500             25%
   r₂       625               6.25%
   r₃       156               1.56%
   r₄       39                0.39%
  ```
2. **Cost coverage plot:** Fraction of zone pairs with computed costs at each layer
3. **Error distribution:** Histogram of approximation errors for sample zone pairs
4. **Spatial grouping visualization:** Map showing group assignments at each layer

### Quality Metrics

**Runtime quality indicators:**

- `get_density()`: Returns fraction of costs stored vs. dense matrix
- `get_finest_layer_density()`: Returns density at the finest layer only
- Layer statistics: Average group size, cost pairs per representative

## Implementation Files

- `hierx/hierarchy.py`: `Hierarchy` class — hierarchical cost matrix construction and lookup
- `hierx/interaction.py`: `InteractionHierarchy` class — sparse interaction operator with correction matrices
- `hierx/backends.py`: Backend-agnostic shortest path computation (scipy, graph-tool, networkx)
- `hierx/utils.py`: Interaction functions, network generators, and dense baselines
- `hierx/storage.py`: Save/load hierarchies and interaction operators to `.npz` / `.h5` files

## Future Directions

### Immediate Extensions

1. **Dynamic graphs with changing costs**:
  - When edge weights change (e.g., congestion), affected cost entries and corrections can be updated incrementally
  - Propagation complexity: O(Δ · K) where Δ is the number of affected zone pairs and K is the number of layers
  - The manuscript characterizes this framework formally; empirical validation of dynamic updates is ongoing
2. **Asymmetric costs**:
  - Requires directed Dijkstra and directed graph input
  - Representative selection: Use out-degree or bidirectional centrality
  - Correction logic: Unchanged mathematically; storage format already holds both directions

### Known Limitations

1. **Path-dependence**: Representative selection depends on processing order
  - Impact: Group boundaries may vary slightly between runs
  - Mitigation: Sort zones consistently before processing
2. **Fixed layer sizes**: Exponential growth may not suit all networks
  - Impact: May have too many or too few layers
  - Mitigation: Tune `increase_factor` for network characteristics
3. **Cutoff sensitivity**: Performance depends on choosing good cutoffs
  - Impact: Too small = information loss; too large = computation cost
  - Mitigation: Use `overlap_factor` ≥ 1.5, adjust based on validation

---

## Quick Reference: Notation Summary


| Code / Doc      | Manuscript | Meaning                                 | Type                  |
| --------------- | ---------- | --------------------------------------- | --------------------- |
| n               | n          | Total number of zones                   | Scalar                |
| k               | k          | Layer index (0=finest)                  | Index                 |
| K               | K          | Number of layers (or set of layers)     | Scalar / Set          |
| R_k             | R_k        | Representative zones at layer k         | Set                   |
| n_k             | n_k        | Number of representatives at layer k    | Scalar                |
| z, i, j         | i, j       | Zone indices                            | Index                 |
| C_k[i,j]        | c_{ij}     | Cost between representatives at layer k | Matrix (sparse)       |
| D_k[i,j]        | F_k[i,j]   | Interaction at layer k: f(C_k[i,j])     | Matrix (sparse)       |
| Corr_k[i,j]     | Corr_k     | Correction term at layer k              | Matrix (sparse)       |
| G_k[r,z]        | G_k        | Group membership: 1 if z in r's group   | Matrix (sparse, bool) |
| a               | x          | Activity/attraction vector              | Vector (n×1)          |
| f(c)            | f(c)       | Distance-to-interaction function        | Function              |
| repr_k(z)       | —          | Representative of zone z at layer k     | Function              |
| r_k             | ρ_k        | Radius parameter of layer k             | Scalar (cost units)   |
| base_radius     | ρ_0        | Finest layer radius                     | Scalar (cost units)   |
| increase_factor | γ          | Layer growth multiplier                 | Scalar                |
| overlap_factor  | α          | Cutoff multiplier                       | Scalar                |
| cutoff_k        | α·ρ_k      | Maximum cost computed at layer k        | Scalar (cost units)   |


---

**Document Status**: Current as of March 2026
**Last Updated**: Based on implementation in `hierx/` package
**Scope**: Standalone conceptual documentation of hierarchical cost and interaction computation
**Version**: 0.1.1