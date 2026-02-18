import subprocess
import os

def run_riemann():
    print("--- COMPILING CHRASS RIEMANN ---")
    cmd = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", "-fopenmp", "-ffast-math",
        "chrass_riemann.cpp", "-o", "chrass_riemann"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("✅ Compiled.")

        print("\n>> Running Zeta Engine...")
        res = subprocess.run(["./chrass_riemann"], capture_output=True, text=True, timeout=60)
        print(res.stdout)

    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        if os.path.exists("chrass_riemann"): os.remove("chrass_riemann")

if __name__ == "__main__":
    run_riemann()
