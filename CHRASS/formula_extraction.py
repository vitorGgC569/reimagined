import subprocess
import os
import numpy as np
from scipy import stats

def compile_formula():
    print("--- COMPILING FORMULA EXTRACTOR ---")
    cmd = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", "chrass_formula.cpp", "-o", "chrass_formula"
    ]
    subprocess.run(cmd, check=True)
    print("✅ Compiled.")

def run_point(N, deg, bit_load):
    try:
        res = subprocess.run(
            ["./chrass_formula", str(N), str(deg), str(bit_load)],
            capture_output=True, text=True, timeout=120
        )
        # Output: N, M, BitLoad, RelaxTime, RelaxCount, PushTime, PushCount, PopTime, PopCount
        return list(map(float, res.stdout.strip().split(',')))
    except Exception as e:
        print(f"Error: {e}")
        return None

def analyze():
    compile_formula()

    # Data Collection
    data = []

    # 1. Vary M (Edges) -> Alpha
    print(">> Collecting Alpha (Edge Scaling)...")
    for deg in [2, 5, 10, 20, 50]:
        row = run_point(100000, deg, 0)
        if row: data.append(row)

    # 2. Vary N (Nodes) -> Beta
    print(">> Collecting Beta (Node Scaling)...")
    for n in [10000, 50000, 100000, 200000]:
        row = run_point(n, 5, 0) # Fixed deg
        if row: data.append(row)

    # 3. Vary BitLoad -> Gamma/Delta
    print(">> Collecting Gamma (Bit Scaling)...")
    for b in [0, 10, 50, 100, 200]: # Abstract load units
        row = run_point(100000, 5, b)
        if row: data.append(row)

    # Regression Analysis
    # T_total = RelaxTime + PushTime + PopTime
    # Model: T ~ alpha * M + beta * N + delta * B * M

    # We focus on RelaxTime vs (M, B) and PQ Time vs (N, B)

    M_vals = np.array([r[1] for r in data])
    N_vals = np.array([r[0] for r in data])
    B_vals = np.array([r[2] for r in data])

    RelaxTime = np.array([r[3] for r in data]) # ns
    PQTime = np.array([r[5] + r[7] for r in data]) # ns

    # Relax Time per Edge ~ alpha + delta * B
    # y = RelaxTime / M
    # x = B
    y_relax_per_edge = RelaxTime / M_vals
    res_relax = stats.linregress(B_vals, y_relax_per_edge)
    alpha = res_relax.intercept
    delta = res_relax.slope

    print("\n=== EMPIRICAL FORMULA EXTRACTED ===")
    print(f"Relaxation Time per Edge: {alpha:.2f} ns + {delta:.2f} ns * (BitLoad)")
    print(f"R-squared: {res_relax.rvalue**2:.4f}")

    # PQ Time per Node ~ beta + gamma * B
    # Approx pop count ~ N (for connected graph)
    # y = PQTime / N
    y_pq_per_node = PQTime / N_vals
    res_pq = stats.linregress(B_vals, y_pq_per_node)
    beta = res_pq.intercept
    gamma = res_pq.slope

    print(f"PQ Time per Node:         {beta:.2f} ns + {gamma:.2f} ns * (BitLoad)")
    print(f"R-squared: {res_pq.rvalue**2:.4f}")

    print("\n=== FINAL CHRASS MODEL ===")
    print(f"T(n, m, b) = ({alpha:.1f} m) + ({beta:.1f} n) + ({delta:.1f} m * b) + ({gamma:.1f} n * b) nanoseconds")

    if os.path.exists("chrass_formula"): os.remove("chrass_formula")

if __name__ == "__main__":
    analyze()
