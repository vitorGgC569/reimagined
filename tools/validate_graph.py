import numpy as np
import argparse
import sys

def check_symmetry(adj):
    """Checks if the matrix is symmetric."""
    if not np.allclose(adj, adj.T, atol=1e-6):
        print("[FAIL] Matrix is NOT symmetric. H-TTT physics requires A = A^T.")
        return False
    print("[PASS] Matrix is symmetric.")
    return True

def check_connectivity(adj):
    """Checks if the graph is connected (no isolated islands)."""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    queue = [0]
    visited[0] = True
    count = 0

    while queue:
        u = queue.pop(0)
        count += 1
        neighbors = np.where(adj[u] > 0)[0]
        for v in neighbors:
            if not visited[v]:
                visited[v] = True
                queue.append(v)

    if count != n:
        print(f"[FAIL] Graph is disconnected. Visited {count}/{n} nodes.")
        return False
    print(f"[PASS] Graph is connected ({n} nodes).")
    return True

def check_dead_ends(adj):
    """Checks for nodes with zero degree."""
    degrees = np.sum(np.abs(adj), axis=1)
    dead = np.where(degrees < 1e-6)[0]
    if len(dead) > 0:
        print(f"[FAIL] Found {len(dead)} dead-end nodes (Degree 0).")
        return False
    print("[PASS] No dead-end nodes.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Graph Doctor for OXTA")
    parser.add_argument("file", help="Path to .npy adjacency matrix")
    args = parser.parse_args()

    try:
        adj = np.load(args.file)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"Analyzing Graph: {adj.shape}")

    ok = True
    ok &= check_symmetry(adj)
    ok &= check_connectivity(adj)
    ok &= check_dead_ends(adj)

    if ok:
        print("\n[VERDICT] Graph is HEALTHY. Safe for ChrassLayer.")
        sys.exit(0)
    else:
        print("\n[VERDICT] Graph is SICK. Do not train.")
        sys.exit(1)

if __name__ == "__main__":
    main()
