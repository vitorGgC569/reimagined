import numpy as np
import uhk

def main():
    print("[Python SDK] Initializing Universal Heterogeneous Kernel Runtime...")
    runtime = uhk.UniversalKernelRuntime()

    print("[Python SDK] Preparing Data (BitNet)...")
    M, N, K = 128, 128, 128

    # Weights (Packed 2-bit): K/4 columns * M rows
    # Random bytes
    A = np.random.randint(0, 255, size=M * (K // 4), dtype=np.uint8)

    # Activations (INT8): K rows * N columns (assuming column major or similar layout check)
    B = np.random.randint(-127, 127, size=K * N, dtype=np.int8)

    # Output (FP32)
    C = np.zeros(M * N, dtype=np.float32)

    print(f"[Python SDK] Submitting Task GEMM_BITNET ({M}x{N}x{K})...")
    runtime.submit_bitnet_gemm(M, N, K, A, B, C)

    # Note: Execution is async.
    # In a real scenario, we might wait for a callback or check a status buffer.

    print("[Python SDK] Shutting down...")
    runtime.shutdown()
    print("[Python SDK] Done.")

if __name__ == "__main__":
    main()
