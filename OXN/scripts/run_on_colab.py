import os
import sys
import subprocess

def setup_colab():
    print(">>> Setting up NSOS on Google Colab (T4)...")

    # 1. Install Dependencies
    print("[1/4] Installing Build Tools...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "cmake", "ninja-build"], check=True)

    # 2. Build
    print("[2/4] Building NSOS with CUDA...")
    os.makedirs("build", exist_ok=True)
    os.chdir("build")

    # Get Python Info
    import sysconfig
    py_include = sysconfig.get_path('include')
    py_lib = sysconfig.get_path('stdlib')

    cmd = [
        "cmake", "..",
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DUSE_CUDA=ON", # Force CUDA on Colab
        f"-DPYTHON_INCLUDE_DIR={py_include}",
        f"-DPYTHON_LIBRARY={py_lib}"
    ]
    subprocess.run(cmd, check=True)
    subprocess.run(["cmake", "--build", "."], check=True)

    os.chdir("..")

    # 3. Generate Data
    print("[3/4] Generating Curriculum Data...")
    subprocess.run(["python3", "scripts/generate_datasets.py"], check=True)

    # 4. Run Training
    print("[4/4] Running Real Training Protocol...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + "/build"
    subprocess.run(["python3", "-u", "scripts/train_nsos_curriculum.py"], check=True, env=env)

if __name__ == "__main__":
    setup_colab()
