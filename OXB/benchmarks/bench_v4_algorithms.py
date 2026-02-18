import time
import numpy as np
from oxtacore.v4_research.rmi_models import LinearModel, CubicSplineModel
from oxtacore.v4_research.hilbert import xy2d
from oxtacore.v4_research.succinct import RankSelectBitVector

def benchmark_v4():
    print("--- Benchmark V4 Algorithms ---")

    # 1. RMI Benchmark
    N = 10000
    X = np.arange(N)
    Y_lin = 3 * X + 2
    Y_poly = X ** 2

    lm = LinearModel()
    t0 = time.time()
    lm.train(X, Y_lin)
    t_lin = time.time() - t0

    sm = CubicSplineModel()
    t0 = time.time()
    sm.train(X, Y_poly)
    t_spl = time.time() - t0

    print(f"RMI Training Time (N={N}): Linear={t_lin:.5f}s, Spline={t_spl:.5f}s")

    # 2. Hilbert Benchmark
    print("\nHilbert Mapping (1024x1024 Grid):")
    t0 = time.time()
    for i in range(10000):
        xy2d(1024, i%1024, i%1024)
    print(f"Time for 10k mappings: {time.time()-t0:.4f}s")

    # 3. Succinct Benchmark
    print("\nSuccinct Rank/Select:")
    bv = RankSelectBitVector([1, 0]*10000)
    t0 = time.time()
    bv.rank1(15000)
    print(f"Rank Query Time: {(time.time()-t0)*1e6:.2f} us")

if __name__ == "__main__":
    benchmark_v4()
