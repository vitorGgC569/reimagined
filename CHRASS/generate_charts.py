import subprocess
import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

def compile_stress():
    print("--- COMPILING KIMERA STRESS TEST FOR CHARTS ---")
    cmd_compile = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", "-fopenmp",
        "kimera_stress.cpp", "-o", "kimera_stress"
    ]
    subprocess.run(cmd_compile, check=True)
    print("✅ Compiled successfully.\n")

def run_point(N, mode):
    try:
        res = subprocess.run(
            ["./kimera_stress", str(N), mode],
            capture_output=True,
            text=True,
            timeout=120
        )
        if res.returncode != 0:
            return None

        t1 = float(re.search(r"Step1\(Chebyshev\): ([\d\.]+) ms", res.stdout).group(1))
        t2 = float(re.search(r"Step2\(Layout\): ([\d\.]+) ms", res.stdout).group(1))
        t3 = float(re.search(r"Step3\(Core\): ([\d\.]+) ms", res.stdout).group(1))
        return (t1, t2, t3)
    except Exception as e:
        print(f"Error N={N} Mode={mode}: {e}")
        return None

def generate_charts():
    compile_stress()

    modes = ["GRID", "CHAOS"]
    # Up to 5M nodes. 10M is risky for plotting script timeouts.
    points_N = [100000, 500000, 1000000, 2000000, 5000000]

    data_grid = {"N": [], "Cheby": [], "Layout": [], "Core": []}
    data_chaos = {"N": [], "Cheby": [], "Layout": [], "Core": []}

    for mode in modes:
        print(f"Collecting data for {mode}...")
        for N in points_N:
            vals = run_point(N, mode)
            if vals:
                t1, t2, t3 = vals
                if mode == "GRID":
                    data_grid["N"].append(N)
                    data_grid["Cheby"].append(t1)
                    data_grid["Layout"].append(t2)
                    data_grid["Core"].append(t3)
                else:
                    data_chaos["N"].append(N)
                    data_chaos["Cheby"].append(t1)
                    data_chaos["Layout"].append(t2)
                    data_chaos["Core"].append(t3)

    # Chart 1: Stacked Bar Chart (Grid)
    # Showing how Setup dominates Core
    if data_grid["N"]:
        labels = [f"{n/1000000:.1f}M" for n in data_grid["N"]]
        setup = np.array(data_grid["Cheby"]) + np.array(data_grid["Layout"])
        core = np.array(data_grid["Core"])

        plt.figure(figsize=(10, 6))
        plt.bar(labels, setup, label='Setup (Cheby + Layout)', color='#1f77b4') # Blue
        plt.bar(labels, core, bottom=setup, label='Core (Search)', color='#d62728') # Red

        plt.title('Kimera Time Breakdown: Grid (Sparse/Hard)')
        plt.ylabel('Time (ms)')
        plt.xlabel('Nodes (Millions)')
        plt.legend()
        plt.savefig('kimera_grid_breakdown.png')
        print("Saved kimera_grid_breakdown.png")

    # Chart 2: Stacked Bar Chart (Chaos)
    # Showing Balanced Growth
    if data_chaos["N"]:
        labels = [f"{n/1000000:.1f}M" for n in data_chaos["N"]]
        setup = np.array(data_chaos["Cheby"]) + np.array(data_chaos["Layout"])
        core = np.array(data_chaos["Core"])

        plt.figure(figsize=(10, 6))
        plt.bar(labels, setup, label='Setup (Cheby + Layout)', color='#1f77b4')
        plt.bar(labels, core, bottom=setup, label='Core (Search)', color='#d62728')

        plt.title('Kimera Time Breakdown: Chaos (Dense/Random)')
        plt.ylabel('Time (ms)')
        plt.xlabel('Nodes (Millions)')
        plt.legend()
        plt.savefig('kimera_chaos_breakdown.png')
        print("Saved kimera_chaos_breakdown.png")

    # Chart 3: Scalability (Total Time)
    plt.figure(figsize=(10, 6))
    if data_grid["N"]:
        total_grid = np.array(data_grid["Cheby"]) + np.array(data_grid["Layout"]) + np.array(data_grid["Core"])
        plt.plot(data_grid["N"], total_grid, marker='o', label='Grid Total Time')

    if data_chaos["N"]:
        total_chaos = np.array(data_chaos["Cheby"]) + np.array(data_chaos["Layout"]) + np.array(data_chaos["Core"])
        plt.plot(data_chaos["N"], total_chaos, marker='s', label='Chaos Total Time')

    plt.title('Kimera Scalability: Total Time vs Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()
    plt.savefig('kimera_scalability.png')
    print("Saved kimera_scalability.png")

    # Cleanup
    if os.path.exists("kimera_stress"):
        os.remove("kimera_stress")

if __name__ == "__main__":
    generate_charts()
