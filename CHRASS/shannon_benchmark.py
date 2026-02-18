import subprocess
import os

def run_shannon():
    print("--- COMPILING KIMERA SHANNON ---")
    cmd_compile = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17",
        "kimera_shannon.cpp", "-o", "kimera_shannon"
    ]
    subprocess.run(cmd_compile, check=True)
    print("✅ Compiled successfully.\n")

    print(">> Running 10^120 Simulation...")
    res = subprocess.run(["./kimera_shannon"], capture_output=True, text=True)
    print(res.stdout)

    if os.path.exists("kimera_shannon"):
        os.remove("kimera_shannon")

if __name__ == "__main__":
    run_shannon()
