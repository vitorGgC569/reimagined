import subprocess
import os

def run_odlyzko():
    print("--- COMPILING ODLYZKO TEST ---")
    # Need -lquadmath for __float128 support
    cmd = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", "-fopenmp", "rierass_odlyzko.cpp", "-o", "rierass_odlyzko", "-lquadmath"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("✅ Compiled.")

        print("\n>> Running High-Altitude Test...")
        res = subprocess.run(["./rierass_odlyzko"], capture_output=True, text=True, timeout=120)
        print(res.stdout)

    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        if os.path.exists("rierass_odlyzko"): os.remove("rierass_odlyzko")

if __name__ == "__main__":
    run_odlyzko()
