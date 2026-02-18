import subprocess
import os
import sys
import re
import math

def compile_stress():
    print("--- COMPILING KIMERA STRESS TEST FOR COMPLEXITY ANALYSIS ---")
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
            timeout=60
        )
        if res.returncode != 0:
            return None

        # Parse output
        t1 = float(re.search(r"Step1\(Chebyshev\): ([\d\.]+) ms", res.stdout).group(1))
        t2 = float(re.search(r"Step2\(Layout\): ([\d\.]+) ms", res.stdout).group(1))
        t3 = float(re.search(r"Step3\(Core\): ([\d\.]+) ms", res.stdout).group(1))
        return (t1, t2, t3)
    except Exception as e:
        return None

def analyze():
    compile_stress()

    modes = ["GRID", "CHAOS"]
    # We want a range where N log N deviates from N.
    # 100k to 5M is a good range.
    points_N = [100000, 500000, 1000000, 2000000, 5000000]

    print(f"{'MODE':<8} | {'N':<10} | {'Cheby (ms)':<10} | {'Layout (ms)':<10} | {'Core (ms)':<10} | {'Core/N':<10}")
    print("-" * 80)

    for mode in modes:
        for N in points_N:
            data = run_point(N, mode)
            if data:
                t1, t2, t3 = data
                ratio = t3 / N * 1000.0 # microseconds per node
                print(f"{mode:<8} | {N:<10} | {t1:<10.2f} | {t2:<10.2f} | {t3:<10.2f} | {ratio:<10.4f}")
            else:
                print(f"{mode:<8} | {N:<10} | {'FAIL':<10} | {'FAIL':<10} | {'FAIL':<10} | {'FAIL':<10}")

    # Remove binary
    if os.path.exists("kimera_stress"):
        os.remove("kimera_stress")

if __name__ == "__main__":
    analyze()
