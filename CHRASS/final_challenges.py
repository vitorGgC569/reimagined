import subprocess
import os

def compile_and_run(name, file):
    print(f"--- {name} ---")
    cmd = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17", file, "-o", "bench_bin"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("✅ Compiled.")
        res = subprocess.run(["./bench_bin"], capture_output=True, text=True, timeout=60)
        print(res.stdout)
    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        if os.path.exists("bench_bin"): os.remove("bench_bin")

if __name__ == "__main__":
    compile_and_run("RSA-4096", "kimera_rsa4096.cpp")
    compile_and_run("FACTORIAL-1000", "kimera_factorial.cpp")
    compile_and_run("POINCARE", "kimera_poincare.cpp")
