import subprocess
import os

def run_vlsi():
    print("--- COMPILING KIMERA VLSI (CHIP ROUTER) ---")
    cmd = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", "-fopenmp",
        "kimera_vlsi.cpp", "-o", "kimera_vlsi"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("✅ Compiled.")

        print("\n>> Running VLSI Simulation...")
        res = subprocess.run(["./kimera_vlsi"], capture_output=True, text=True, timeout=300)
        print(res.stdout)

    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        if os.path.exists("kimera_vlsi"): os.remove("kimera_vlsi")

if __name__ == "__main__":
    run_vlsi()
