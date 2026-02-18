"""
THE GOD ALGORITHM V5.0: THERMO-DIFFERENTIAL WDD-SSSP
====================================================
Fusion of Computational Physics and Vectorized Graph Processing.

Concept:
1. "Pre-Heating": Solve Heat Diffusion equation to find global structure.
2. "Entropic Layout": Reorder memory based on heat (temperature).
3. "Vectorized Core": Run SPFA on cache-optimized topology.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
import time
import sys

# ==============================================================================
# PHYSICS ENGINE: HEAT DIFFUSION
# ==============================================================================
class ThermoEngine:
    def __init__(self, adj: sp.csr_matrix, n: int):
        self.adj = adj
        self.n = n

    def compute_heat_map(self, source: int, diffusion_time=0.1):
        """
        Solves the heat equation (I + t*L)u = f
        where L is the Graph Laplacian.
        """
        # Construct Laplacian: L = D - A
        # For performance, we use simplified Laplacian or even just I - alpha*A (PageRank-like)
        # which is faster and provides similar locality hints.
        # Let's use standard Laplacian approximation for diffusion.

        degrees = np.array(self.adj.sum(axis=1)).flatten()
        # Normalized Laplacian usually converges better: L = I - D^-1/2 A D^-1/2
        # But D - A is simpler. Let's use a damped system: (I + t*L) x = source_impulse
        # x = (I + t*(D - A))^-1 source

        # Optimized construction:
        # Operator = I + t*D - t*A
        # This preserves sparsity.

        t = diffusion_time

        # Main diagonal: 1 + t * degrees
        diag_values = 1.0 + t * degrees

        # Off-diagonal: -t * A
        # We can construct the linear operator implicitly or explicitly.
        # Explicit CSR is fast enough for 1M nodes if sparse.

        # Copying A structure
        # Must cast to float64 to avoid type casting errors during multiplication
        L_op = self.adj.astype(np.float64)
        L_op.data *= -t

        # Add diagonal
        # scipy sparse matrices allow setdiag? better to add separate diag matrix
        I_plus_tD = sp.diags(diag_values, 0, format='csr')
        SystemMatrix = I_plus_tD + L_op

        # RHS: Impulse at source
        b = np.zeros(self.n)
        b[source] = 1.0

        # Solve with CG (Conjugate Gradient) since SystemMatrix is symmetric positive definite (usually)
        # Low tolerance is fine for heuristic reordering! We don't need exact physics.
        # tol=1e-3 is sufficient to sort nodes.
        # Note: SciPy API uses 'tol' or 'rtol'. In newer versions (like 1.17+), 'rtol' is preferred or 'tol' is deprecated.
        # Let's check what arguments are supported or use both? No, simpler: check scipy version or use positional?
        # Standard signature is (A, b, x0, tol, maxiter, ...)
        # Using kwargs can be version dependent.
        heat_map, info = spla.cg(SystemMatrix, b, rtol=1e-3, maxiter=50)

        return heat_map

    def compute_fast_heat_map(self, source: int, iterations=10):
        """
        Approximates heat diffusion using simple matrix multiplication (Power Iteration).
        Much faster than solving the linear system, but less physically accurate.
        Good enough for cache locality hints.

        New State = Old State + Diffusion * (Neighbors - Current)
        Simulates explicit time-stepping of heat equation.
        """
        # Ensure float64 for precision during iteration
        # Normalized Adjacency (Stochastic Matrix approximate)
        # D^-1 A
        degrees = np.array(self.adj.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1.0 # Avoid div by zero
        inv_degrees = 1.0 / degrees

        # Create diffusion operator: M = D^-1 * A
        D_inv = sp.diags(inv_degrees, 0, format='csr')
        # Casting to float is crucial
        M = D_inv.dot(self.adj.astype(np.float64))

        # Initial Heat Vector (Hot Source)
        heat = np.zeros(self.n, dtype=np.float64)
        heat[source] = 1.0

        # Iterative Diffusion (SpMV - Sparse Matrix Vector mult)
        # This is extremely optimized in SciPy/MKL
        for _ in range(iterations):
            # heat_new = 0.5 * heat + 0.5 * (M * heat)
            # Damping factor 0.5 prevents oscillation
            heat = 0.5 * heat + 0.5 * M.dot(heat)

        return heat

# ==============================================================================
# MEMORY CONTROLLER: WDD LAYOUT
# ==============================================================================
class LayoutEngine:
    def __init__(self, n):
        self.n = n
        self.perm = None
        self.inv_perm = None

    def reorder(self, adj: sp.csr_matrix, metric: np.ndarray):
        """
        Reorders the adjacency matrix based on the metric (Heat).
        Hot nodes (high value) -> Index 0.
        """
        # Sort descending: hottest first
        self.perm = np.argsort(-metric).astype(np.int32)

        # Build inverse permutation
        self.inv_perm = np.zeros(self.n, dtype=np.int32)
        self.inv_perm[self.perm] = np.arange(self.n, dtype=np.int32)

        # Permute Matrix: P A P^T
        # Row permutation
        adj_perm = adj[self.perm, :]
        # Col permutation
        adj_perm = adj_perm[:, self.perm]

        return adj_perm

# ==============================================================================
# COMPUTE CORE: V18 (VECTORIZED SPFA)
# ==============================================================================
class VectorizedCore:
    def __init__(self, n, adj_csr):
        self.n = n
        self.adj = adj_csr

    def solve(self, source_idx):
        inf = 1e14
        dist = np.full(self.n, inf, dtype=np.float64)
        dist[source_idx] = 0
        active_nodes = np.array([source_idx], dtype=np.int32)

        # Statistics
        hops = 0
        ops = 0

        while active_nodes.size > 0:
            hops += 1

            # 1. Gather (Block Read)
            # Memory Locality Critical Step:
            # If active_nodes are close in ID (e.g., 0, 1, 2...),
            # this accesses contiguous rows in CSR -> Cache Hit!
            sub_adj = self.adj[active_nodes]
            if sub_adj.nnz == 0: break

            coo = sub_adj.tocoo()
            global_u = active_nodes[coo.row]
            global_v = coo.col
            weights = coo.data

            ops += len(weights)

            # 2. Relax (ALU)
            new_dists = dist[global_u] + weights

            # 3. Filter
            mask = new_dists < (dist[global_v] - 1e-9)
            if not np.any(mask): break

            valid_v = global_v[mask]
            valid_d = new_dists[mask]

            # 4. Update
            np.minimum.at(dist, valid_v, valid_d)

            # 5. Next Frontier
            active_nodes = np.unique(valid_v)

        return dist, hops, ops

# ==============================================================================
# THE GOD ALGORITHM INTEGRATION
# ==============================================================================
class ThermoWDD_Solver:
    def __init__(self, n, edges_list, symmetric=True):
        self.n = n
        # Build initial CSR
        sources = [u for u,v,w in edges_list]
        targets = [v for u,v,w in edges_list]
        weights = [w for u,v,w in edges_list]

        if symmetric:
            sources_sym = sources + targets
            targets_sym = targets + sources
            weights_sym = weights + weights
            self.adj = sp.csr_matrix((weights_sym, (sources_sym, targets_sym)), shape=(n, n))
        else:
            self.adj = sp.csr_matrix((weights, (sources, targets)), shape=(n, n))

    def solve(self, source_node, method='exact'):
        stats = {}
        t0 = time.perf_counter()

        # --- PHASE 1: PHYSICS (Heat Diffusion) ---
        t_phy_start = time.perf_counter()
        thermo = ThermoEngine(self.adj, self.n)

        if method == 'fast':
            # 10 iterations of SpMV is extremely fast
            heat_map = thermo.compute_fast_heat_map(source_node, iterations=10)
        else:
            # Original Exact CG Solver
            heat_map = thermo.compute_heat_map(source_node, diffusion_time=0.1)

        stats['time_physics'] = (time.perf_counter() - t_phy_start) * 1000

        # --- PHASE 2: LAYOUT (WDD / Reordering) ---
        t_layout_start = time.perf_counter()
        layout = LayoutEngine(self.n)
        adj_optimized = layout.reorder(self.adj, heat_map)

        # Map source node to new ID (it should become 0 if it's the hottest!)
        source_mapped = layout.inv_perm[source_node]
        stats['time_layout'] = (time.perf_counter() - t_layout_start) * 1000

        # --- PHASE 3: CORE (Vectorized Solve) ---
        t_core_start = time.perf_counter()
        v18 = VectorizedCore(self.n, adj_optimized)
        dist_mapped, hops, ops = v18.solve(source_mapped)
        stats['time_core'] = (time.perf_counter() - t_core_start) * 1000
        stats['hops'] = hops

        # --- PHASE 4: RECONSTRUCTION (Map back) ---
        # dist_final[original_id] = dist_mapped[perm_id]
        dist_final = np.zeros(self.n, dtype=np.float64)
        dist_final[layout.perm] = dist_mapped

        stats['total_time'] = (time.perf_counter() - t0) * 1000
        return dist_final, stats

# ==============================================================================
# BENCHMARK SUITE
# ==============================================================================
def run_comparison():
    # Setup Scale
    N = 1_000_000 # 1M nodes
    AVG_DEG = 3
    print(f"--- THERMO-DIFFERENTIAL V5.0 BENCHMARK (N={N/1e6}M) ---")

    # 1. Generate Graph (NumPy optimized)
    print("Synthesizing Graph...")
    m = N * AVG_DEG
    np.random.seed(42)
    sources = np.random.randint(0, N, m, dtype=np.int32)
    targets = np.random.randint(0, N, m, dtype=np.int32)
    weights = np.random.randint(1, 100, m, dtype=np.int32)

    # Remove self-loops
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]
    weights = weights[mask]

    edges_list = list(zip(sources, targets, weights))
    print(f"Edges: {len(edges_list)}")

    # --------------------------------------------------------------------------
    # COMPETITOR 1: V18 RAW (No Physics, Random Layout)
    # --------------------------------------------------------------------------
    print("\n>>> [1/2] RUNNING V18 RAW (Standard)...")
    from prototype import HFS_SSSP_V18_Symmetric # Import previous robust class

    v18_raw = HFS_SSSP_V18_Symmetric(N, edges_list, symmetric=True)
    dist_raw, time_raw, hops_raw = v18_raw.solve_exact(0)
    print(f"    V18 RAW Time : {time_raw:.2f} ms")
    print(f"    V18 RAW Hops : {hops_raw}")

    # --------------------------------------------------------------------------
    # COMPETITOR 2: THERMO-WDD (Exact CG Physics)
    # --------------------------------------------------------------------------
    print("\n>>> [2/3] RUNNING THERMO-WDD (Exact Physics - CG)...")
    god_solver = ThermoWDD_Solver(N, edges_list, symmetric=True)
    dist_god, stats = god_solver.solve(0, method='exact')

    print(f"    Physics Time : {stats['time_physics']:.2f} ms")
    print(f"    Layout Time  : {stats['time_layout']:.2f} ms")
    print(f"    Core Time    : {stats['time_core']:.2f} ms")
    print(f"    TOTAL Time   : {stats['total_time']:.2f} ms")

    # --------------------------------------------------------------------------
    # COMPETITOR 3: THERMO-WDD (Fast Jacobi Physics)
    # --------------------------------------------------------------------------
    print("\n>>> [3/3] RUNNING THERMO-WDD (Fast Physics - Jacobi)...")
    dist_god_fast, stats_fast = god_solver.solve(0, method='fast')

    print(f"    Physics Time : {stats_fast['time_physics']:.2f} ms")
    print(f"    Layout Time  : {stats_fast['time_layout']:.2f} ms")
    print(f"    Core Time    : {stats_fast['time_core']:.2f} ms")
    print(f"    TOTAL Time   : {stats_fast['total_time']:.2f} ms")

    # --------------------------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------------------------
    print("\n[FINAL ANALYSIS]")
    print(f"1. V18 Raw       : {time_raw:.2f} ms")
    print(f"2. Thermo (Exact): {stats['total_time']:.2f} ms")
    print(f"3. Thermo (Fast) : {stats_fast['total_time']:.2f} ms")

    best_time = min(time_raw, stats['total_time'], stats_fast['total_time'])

    if stats_fast['total_time'] < time_raw:
        print("\n🏆 PHYSICS (FAST) WINS! Approximation strategy successful.")
        print(f"   Speedup vs Raw: {time_raw / stats_fast['total_time']:.2f}x")
    elif stats['total_time'] < time_raw:
        print("\n🏆 PHYSICS (EXACT) WINS! (Unexpected for Random Graph)")
    else:
        print("\n🏆 RAW WINS. Even fast physics overhead was too high for this graph density.")

if __name__ == "__main__":
    run_comparison()
