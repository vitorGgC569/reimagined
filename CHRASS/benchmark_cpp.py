import subprocess
import os
import sys

def run_cpp_benchmark():
    src = "god_algorithm.cpp"
    bin_name = "god_algo"

    print(f"--- COMPILING GOD ALGORITHM (C++ NATIVE) ---")
    # Flags agressivas para "Potência Máxima"
    cmd = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", "-fopenmp", # Enable OpenMP for physics
        src, "-o", bin_name
    ]

    try:
        subprocess.run(cmd, check=True)
        print("Compilation successful.")
    except subprocess.CalledProcessError:
        print("Compilation failed.")
        return

    print(f"\n--- RUNNING BENCHMARK ---")
    try:
        subprocess.run([f"./{bin_name}"], check=True)
    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        if os.path.exists(bin_name):
            os.remove(bin_name)

if __name__ == "__main__":
    run_cpp_benchmark()
