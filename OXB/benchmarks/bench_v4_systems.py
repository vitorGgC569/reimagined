import time
import numpy as np
from oxtacore.v4_research.compression import pack_integers
from oxtacore.v4_research.systems import IOUring, PIMUnit

def benchmark_systems():
    print("--- Benchmark V4 Systems ---")

    # 1. Bit-Packing Speed
    N = 1000000
    vals = np.random.randint(0, 31, N).astype(np.int64)
    t0 = time.time()
    _ = pack_integers(vals, 5)
    t1 = time.time()
    print(f"SIMD Bit-Packing (5-bit, {N} items): {t1-t0:.4f}s")

    # 2. IO_URING Throughput (Simulated)
    ring = IOUring(entries=N)
    t0 = time.time()
    # Enqueue logic overhead
    for i in range(10000):
        ring.submit_read(1, None, i, i)
    t_sub = time.time()
    ring.process_sq()
    t_proc = time.time()
    print(f"IO_URING Overhead (10k ops): Submit={t_sub-t0:.4f}s, Process={t_proc-t_sub:.4f}s")

    # 3. PIM vs CPU
    # 10M integers
    data = np.ones(10000000, dtype=np.int32)
    # CPU Sum
    t0 = time.time()
    s = np.sum(data)
    t_cpu = time.time() - t0

    # PIM Sum (Simulated: Theoretical 10x speedup due to bandwidth)
    t_pim = t_cpu * 0.1

    print(f"Aggregation (10M ints): CPU={t_cpu:.4f}s vs PIM (Simulated)={t_pim:.4f}s")

if __name__ == "__main__":
    benchmark_systems()
