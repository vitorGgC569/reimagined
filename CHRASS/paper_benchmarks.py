import subprocess
import os
import time
import numpy as np
import tempfile

def compile_all():
    print("--- COMPILING SUITE ---")
    # Kimera Paper (AVX2)
    subprocess.run(["g++", "-O3", "-march=native", "-funroll-loops", "-std=c++17", "-fopenmp", "kimera_paper.cpp", "-o", "kimera_paper"], check=True)
    # Kimera Galactic/RSA
    subprocess.run(["g++", "-O3", "-march=native", "-funroll-loops", "-std=c++17", "kimera_galactic.cpp", "-o", "kimera_galactic"], check=True)
    subprocess.run(["g++", "-O3", "-march=native", "-funroll-loops", "-std=c++17", "kimera_shannon.cpp", "-o", "kimera_shannon"], check=True)
    subprocess.run(["g++", "-O3", "-march=native", "-funroll-loops", "-std=c++17", "kimera_rsa.cpp", "-o", "kimera_rsa"], check=True)
    subprocess.run(["g++", "-O3", "-march=native", "-funroll-loops", "-std=c++17", "kimera_rsa4096.cpp", "-o", "kimera_rsa4096"], check=True)

    # Duan & V18 (Using existing ultimate_benchmark compilation logic if binary absent, but we assume environment)
    # Re-compiling god_native just in case
    subprocess.run(["g++", "-O3", "-march=native", "-funroll-loops", "-std=c++17", "-fopenmp", "god_algorithm.cpp", "-o", "god_native"], check=True)

    print("✅ Compilation Done.")

def generate_graph(type, N):
    m = N * 3
    sources = np.random.randint(0, N, m, dtype=np.int32)
    targets = np.random.randint(0, N, m, dtype=np.int32)
    weights = np.random.randint(1, 100, m, dtype=np.int32)

    if type == "MAP":
        side = int(np.sqrt(N))
        r, c = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
        u = r * side + c
        u_h = u[:, :-1].flatten(); v_h = u[:, 1:].flatten()
        u_v = u[:-1, :].flatten(); v_v = u[1:, :].flatten()
        sources = np.concatenate([u_h, u_v])
        targets = np.concatenate([v_h, v_v])
        m = len(sources)
        weights = np.random.randint(1, 100, m, dtype=np.int32)

    fd, path = tempfile.mkstemp(text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(f"{N} {m} 0\n")
        for i in range(m):
            f.write(f"{sources[i]} {targets[i]} {weights[i]}\n")
    return path

def run_axis_1():
    print("\n=== AXIS 1: ALGORITHMIC COMPARISON ===")
    N = 1_000_000
    scenarios = ["CHAOS", "MAP"]
    print(f"{'SCENARIO':<10} | {'DUAN (ms)':<10} | {'V18 (ms)':<10} | {'KIMERA (ms)':<10}")

    for sc in scenarios:
        path = generate_graph(sc, N)

        # Duan
        try:
            res = subprocess.run(["./duan_native", path], capture_output=True, text=True, timeout=60)
            t_duan = float(res.stdout.splitlines()[0])
        except: t_duan = 99999.9

        # V18
        try:
            res = subprocess.run(["./god_native", path, "RAW"], capture_output=True, text=True, timeout=60)
            t_v18 = float(res.stdout.strip())
        except: t_v18 = 99999.9

        # Kimera
        try:
            res = subprocess.run(["./kimera_paper", path, "15", "1"], capture_output=True, text=True, timeout=60)
            t_kimera = float(res.stdout.split()[0])
        except: t_kimera = 99999.9

        print(f"{sc:<10} | {t_duan:<10.2f} | {t_v18:<10.2f} | {t_kimera:<10.2f}")
        os.remove(path)

def run_axis_2():
    print("\n=== AXIS 2: SPECTRAL VALIDATION (On Grid) ===")
    path = generate_graph("MAP", 1_000_000)

    # Without Spectral
    res_no = subprocess.run(["./kimera_paper", path, "0", "1"], capture_output=True, text=True)
    parts_no = res_no.stdout.split()
    t_no = float(parts_no[0])
    ops_no = int(parts_no[2])

    # With Spectral
    res_yes = subprocess.run(["./kimera_paper", path, "15", "1"], capture_output=True, text=True)
    parts_yes = res_yes.stdout.split()
    t_yes = float(parts_yes[0])
    ops_yes = int(parts_yes[2])

    print(f"Metric      | No Spectral | With Spectral")
    print(f"Time (ms)   | {t_no:<11.2f} | {t_yes:<11.2f}")
    print(f"Relaxations | {ops_no:<11} | {ops_yes:<11}")
    print(f"Speedup     | 1.00x       | {t_no/t_yes:.2f}x")
    os.remove(path)

def run_axis_3():
    print("\n=== AXIS 3: BIT-COMPLEXITY SCALING ===")
    # Running binaries directly which have embedded graph generators (small N)
    # Output format: Time: X ms

    def parse_time(cmd):
        res = subprocess.run([cmd], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if "Time:" in line:
                return float(line.split()[1])
        return 0.0

    t_192 = parse_time("./kimera_galactic")
    t_512 = parse_time("./kimera_shannon")
    t_2048 = parse_time("./kimera_rsa")
    t_4096 = parse_time("./kimera_rsa4096")

    print(f"{'Bits':<6} | {'Time (ms)':<10}")
    print(f"{192:<6} | {t_192:<10.2f}")
    print(f"{512:<6} | {t_512:<10.2f}")
    print(f"{2048:<6} | {t_2048:<10.2f}")
    print(f"{4096:<6} | {t_4096:<10.2f}")

def run_axis_4():
    print("\n=== AXIS 4: HARDWARE SENSITIVITY (AVX vs Scalar) ===")
    path = generate_graph("CHAOS", 1_000_000)

    # Scalar (Force flag 0)
    res_s = subprocess.run(["./kimera_paper", path, "15", "0"], capture_output=True, text=True)
    t_scalar = float(res_s.stdout.split()[1]) # Core time only

    # AVX (Force flag 1)
    res_a = subprocess.run(["./kimera_paper", path, "15", "1"], capture_output=True, text=True)
    t_avx = float(res_a.stdout.split()[1]) # Core time only

    print(f"Mode   | Core Time (ms) | Speedup")
    print(f"Scalar | {t_scalar:<14.2f} | 1.00x")
    print(f"AVX2   | {t_avx:<14.2f} | {t_scalar/t_avx:.2f}x")
    os.remove(path)

if __name__ == "__main__":
    compile_all()
    run_axis_1()
    run_axis_2()
    run_axis_3()
    run_axis_4()
