import subprocess
import os

def run_rierass():
    print("--- COMPILING RIERASS (V20) ---")
    cmd = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", "rierass_core.cpp", "-o", "rierass_core"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("✅ Compiled.")

        print("\n>> Running Next-Gen Riemann Engine...")
        res = subprocess.run(["./rierass_core"], capture_output=True, text=True, timeout=60)
        print(res.stdout)

    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        if os.path.exists("rierass_core"): os.remove("rierass_core")

if __name__ == "__main__":
    run_rierass()
