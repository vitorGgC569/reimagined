import subprocess
import os

def run_tsp():
    print("--- COMPILING CHRASS TSP ---")
    cmd = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", "chrass_tsp.cpp", "-o", "chrass_tsp"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("✅ Compiled.")

        print("\n>> Running TSP Heuristic Comparison...")
        res = subprocess.run(["./chrass_tsp"], capture_output=True, text=True, timeout=60)
        print(res.stdout)

    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        if os.path.exists("chrass_tsp"): os.remove("chrass_tsp")

if __name__ == "__main__":
    run_tsp()
