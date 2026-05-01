import time
import threading
import sys
import random
import numpy as np

# Try importing the SDK
try:
    import uhk.uhk_core as uhk_core
except ImportError:
    print("[ERROR] UHK SDK not found. Build it first.")
    sys.exit(1)

def run_benchmark():
    print("Initializing UHK Runtime...")
    runtime = uhk_core.UniversalKernelRuntime()

    # Wait for initialization
    time.sleep(1)

    print("\n--- Starting BitNet GEMM Benchmark (Simulation Mode) ---")

    # Start a producer thread to feed the ring buffer
    def producer():
        M, N, K = 128, 128, 128
        A = np.random.randint(0, 3, M*K, dtype=np.uint8) # Fake ternary packed
        B = np.random.randint(-127, 127, K*N, dtype=np.int8)
        C = np.zeros(M*N, dtype=np.float32)

        while keep_running:
            runtime.submit_bitnet_gemm(M, N, K, A, B, C)
            # Sleep slightly to not overflow the ring buffer instantly in simulation
            time.sleep(0.0001)

    keep_running = True
    t = threading.Thread(target=producer)
    t.start()

    # Monitor Metrics
    start_time = time.time()
    try:
        while time.time() - start_time < 10: # Run for 10 seconds
            throughput = runtime.get_throughput()
            latency = runtime.get_latency()
            active_sms = runtime.get_active_sms()
            usage = runtime.get_ring_buffer_usage()
            processed = runtime.get_tasks_processed()

            print(f"\r[STATUS] Throughput: {throughput:.2f} TOPS | Latency: {latency:.3f} us | SMs: {active_sms} | RingBuffer: {usage} | Tasks: {processed}", end="")
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        keep_running = False
        t.join()
        print("\n\nBenchmark Finished.")
        runtime.shutdown()

if __name__ == "__main__":
    run_benchmark()
