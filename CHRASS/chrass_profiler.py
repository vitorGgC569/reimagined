import subprocess
import os
import sys

def compile_chrass():
    print("--- COMPILING CHRASS ---")
    cmd = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", "-fopenmp", "chrass.cpp", "-o", "chrass"
    ]
    subprocess.run(cmd, check=True)
    print("✅ Compiled.")

def run_test(N, mode, deg):
    try:
        res = subprocess.run(["./chrass", str(N), mode, str(deg)], capture_output=True, text=True, timeout=120)
        return res.stdout.strip()
    except Exception as e:
        return f"ERR: {e}"

def main():
    compile_all = True
    if compile_all: compile_chrass()

    print("\n--- TEST 1: SETUP SCALING (Vary N, Grid) ---")
    print("N,M,SetupTime,CoreTime,Ops")
    # Linear growth test for Setup
    for n in [10000, 50000, 100000, 500000, 1000000, 2000000]:
        print(run_test(n, "GRID", 0))

    print("\n--- TEST 2: CORE SCALING (Vary M, Chaos) ---")
    print("N,M,SetupTime,CoreTime,Ops")
    # Linear growth test for Core (Ops vs M)
    # Fixed N=100k, varying Deg -> varying M
    n = 100000
    for deg in [2, 5, 10, 20, 50, 100]:
        print(run_test(n, "CHAOS", deg))

    if os.path.exists("chrass"): os.remove("chrass")

if __name__ == "__main__":
    main()
