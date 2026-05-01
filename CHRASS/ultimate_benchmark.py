"""
ULTIMATE BENCHMARK: V18 (C++) vs GOD V6 (C++) vs DUAN (C++)
===========================================================
The final showdown using strictly controlled environments.
"""

import subprocess
import os
import sys
import numpy as np
import tempfile
import time

def compile_binaries():
    print("--- COMPILING BINARIES ---")

    # 1. GOD ALGORITHM (V18/V6)
    cmd_god = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", "-fopenmp",
        "god_algorithm.cpp", "-o", "god_native"
    ]
    subprocess.run(cmd_god, check=True)
    print("✅ god_native compiled.")

    # 2. DUAN ALGORITHM
    # Requires cloning dependency if not present
    # Check for header file to ensure valid repo (handling empty submodule dir)
    repo_path = "bmssp_cpp"
    header_path = os.path.join(repo_path, "bmssp.hpp")

    if not os.path.exists(header_path):
        # Remove empty dir if exists
        if os.path.exists(repo_path):
            import shutil
            shutil.rmtree(repo_path)
        subprocess.run(["git", "clone", "https://github.com/lcs147/bmssp.git", repo_path], check=True)

    cmd_duan = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++20",
        "bridge.cpp", "-o", "duan_native"
    ]
    subprocess.run(cmd_duan, check=True)
    print("✅ duan_native compiled.")

    # 3. KIMERA ALGORITHM
    cmd_kimera = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", "-fopenmp",
        "kimera.cpp", "-o", "kimera_native"
    ]
    subprocess.run(cmd_kimera, check=True)
    print("✅ kimera_native compiled.")

def generate_graph_file(scenario, N):
    print(f"   Generating graph for {scenario} (N={N})...")
    avg_deg = 3
    m = N * avg_deg

    np.random.seed(42)

    if scenario == "CHAOS":
        sources = np.random.randint(0, N, m, dtype=np.int32)
        targets = np.random.randint(0, N, m, dtype=np.int32)
    elif scenario == "MAP":
        side = int(np.sqrt(N))
        print(f"   (Generating Perfect Grid {side}x{side}...)")
        # Deterministic Grid generation (matching prototype_v6.py logic)
        # Horizontal edges: (r, c) -> (r, c+1)
        # Vertical edges:   (r, c) -> (r+1, c)

        # Vectorized generation
        # Grid indices
        r, c = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
        u = r * side + c

        # Horizontal neighbors (excluding last col)
        u_h = u[:, :-1].flatten()
        v_h = u[:, 1:].flatten()

        # Vertical neighbors (excluding last row)
        u_v = u[:-1, :].flatten()
        v_v = u[1:, :].flatten()

        sources = np.concatenate([u_h, u_v]).astype(np.int32)
        targets = np.concatenate([v_h, v_v]).astype(np.int32)

        # Recalculate m to be exact
        m = len(sources)

        # Weights 1-100 (Hard Mode - Requested by User)
        # Testing limits of SPFA convergence on deep graphs
        weights = np.random.randint(1, 100, m, dtype=np.int32)

        # Make edges symmetric manually here because load_graph_from_file handles it?
        # No, load_graph_from_file handles symmetry if flag passed.
        # But this generator makes directed edges (u->v).
        # prototype_v6 generated symmetric grid.
        # Let's rely on the loader to symmetrize if needed, BUT
        # ultimate_benchmark writes N M ...
        # The C++ loader reads M edges.
        # We should write standard edges.
    elif scenario == "DAG":
        # Layered DAG logic
        sources = np.random.randint(0, N-1, m, dtype=np.int32)
        # Target always > Source (DAG property)
        targets = sources + np.random.randint(1, 1000, m, dtype=np.int32)
        targets = np.clip(targets, 0, N-1)

    if scenario != "MAP":
        weights = np.random.randint(1, 100, m, dtype=np.int32)

    # Write to temp file
    fd, path = tempfile.mkstemp(text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(f"{N} {m} 0\n")
        # Vectorized write?
        # Converting to big string is memory heavy. Iterate.
        for i in range(m):
            f.write(f"{sources[i]} {targets[i]} {weights[i]}\n")

    return path

def run_benchmark():
    compile_binaries()

    N = 1_000_000
    scenarios = ["CHAOS", "MAP", "DAG"]

    results = {}

    for sc in scenarios:
        print(f"\n=== BENCHMARKING SCENARIO: {sc} ===")
        graph_file = generate_graph_file(sc, N)

        # 1. DUAN (C++)
        print("   Running Duan (C++)...")
        try:
            res = subprocess.run(["./duan_native", graph_file], capture_output=True, text=True)
            t_duan = float(res.stdout.splitlines()[0])
        except: t_duan = float('inf')

        # 2. V18 RAW (C++)
        print("   Running V18 RAW (C++)...")
        try:
            res = subprocess.run(["./god_native", graph_file, "RAW"], capture_output=True, text=True)
            t_v18 = float(res.stdout.strip())
        except: t_v18 = float('inf')

        # 3. GOD V6 (C++)
        print("   Running GOD V6 (C++)...")
        try:
            res = subprocess.run(["./god_native", graph_file, "GOD"], capture_output=True, text=True)
            t_god = float(res.stdout.strip())
        except: t_god = float('inf')

        # 4. KIMERA (C++)
        print("   Running KIMERA (C++)...")
        try:
            res = subprocess.run(["./kimera_native", graph_file], capture_output=True, text=True)
            t_kimera = float(res.stdout.strip())
        except: t_kimera = float('inf')

        results[sc] = (t_duan, t_v18, t_god, t_kimera)

        os.remove(graph_file)

    print("\n\n" + "="*80)
    print(f"{'SCENARIO':<10} | {'DUAN':<10} | {'V18 (Raw)':<10} | {'GOD V6':<10} | {'KIMERA':<10} | {'WINNER':<10}")
    print("-" * 80)

    for sc, (duan, v18, god, kimera) in results.items():
        times = {"DUAN": duan, "V18": v18, "GOD V6": god, "KIMERA": kimera}
        winner = min(times, key=times.get)

        print(f"{sc:<10} | {duan:<10.2f} | {v18:<10.2f} | {god:<10.2f} | {kimera:<10.2f} | {winner:<10}")

    print("="*60)

if __name__ == "__main__":
    run_benchmark()
