import subprocess
import os

def run_suite():
    print("--- COMPILING NEW FRONTIERS ---")

    # Collatz
    subprocess.run(["g++", "-O3", "-march=native", "chrass_collatz.cpp", "-o", "chrass_collatz"], check=True)

    # Mersenne
    subprocess.run(["g++", "-O3", "-march=native", "chrass_mersenne.cpp", "-o", "chrass_mersenne"], check=True)

    # Navier (Requires quadmath)
    try:
        subprocess.run(["g++", "-O3", "chrass_navier.cpp", "-o", "chrass_navier", "-lquadmath"], check=True)
    except:
        print("Warning: Quadmath not found, skipping Navier.")

    # Coloring
    subprocess.run(["g++", "-O3", "chrass_coloring.cpp", "-o", "chrass_coloring"], check=True)

    print("\n>>> RUNNING BENCHMARKS <<<")

    print("\n[1] Collatz Throughput:")
    subprocess.run(["./chrass_collatz"], check=True)

    print("\n[2] Mersenne Primes:")
    subprocess.run(["./chrass_mersenne"], check=True)

    if os.path.exists("chrass_navier"):
        print("\n[3] Navier-Stokes Precision:")
        subprocess.run(["./chrass_navier"], check=True)

    print("\n[4] Graph Coloring:")
    subprocess.run(["./chrass_coloring"], check=True)

    # Cleanup
    for bin in ["chrass_collatz", "chrass_mersenne", "chrass_navier", "chrass_coloring"]:
        if os.path.exists(bin): os.remove(bin)

if __name__ == "__main__":
    run_suite()
