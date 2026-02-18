import subprocess
import os

def run_galactic():
    print("--- COMPILING KIMERA GALACTIC ---")
    cmd_compile = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17",
        "kimera_galactic.cpp", "-o", "kimera_galactic"
    ]
    subprocess.run(cmd_compile, check=True)
    print("✅ Compiled successfully.\n")

    print(">> Running 10^40 Simulation...")
    res = subprocess.run(["./kimera_galactic"], capture_output=True, text=True)
    print(res.stdout)

    if os.path.exists("kimera_galactic"):
        os.remove("kimera_galactic")

if __name__ == "__main__":
    run_galactic()
