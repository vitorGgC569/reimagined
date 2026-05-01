import subprocess
import os

def run_rsa():
    print("--- COMPILING KIMERA RSA ---")
    cmd_compile = [
        "g++", "-O3", "-march=native", "-funroll-loops",
        "-std=c++17",
        "kimera_rsa.cpp", "-o", "kimera_rsa"
    ]
    subprocess.run(cmd_compile, check=True)
    print("✅ Compiled successfully.\n")

    print(">> Running RSA-2048 Simulation...")
    res = subprocess.run(["./kimera_rsa"], capture_output=True, text=True)
    print(res.stdout)

    if os.path.exists("kimera_rsa"):
        os.remove("kimera_rsa")

if __name__ == "__main__":
    run_rsa()
