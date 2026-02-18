import sys
import os
import time
import psutil

# FIX: Add CUDA bin to PATH/DLL Directory for Windows (Crucial for GPU)
if os.name == 'nt':
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)
        os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']

# Path to built extension
build_path = os.path.join(os.getcwd(), 'OXN/nsos/build')
sys.path.append(build_path)

try:
    import nsos_ext
except ImportError as e:
    print(f"CRITICAL: Failed to import nsos_ext. {e}")
    sys.exit(1)

def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_device(device_name, device_enum):
    print(f"\n--- Benchmarking on {device_name} ---")
    
    try:
        # Config (Reduced for stability check)
        batch_size = 1
        seq_len = 128
        d_model = 512 # Scaled up to 512
        vocab_size = 1000
        layers = 4 # Restored to 4
        
        print(f"Model Config: L={layers}, D={d_model}, Vocab={vocab_size}")
        
        # Init
        start_mem = measure_memory()
        model = nsos_ext.JambaModel(layers, d_model, vocab_size, device_enum)
        end_mem = measure_memory()
        print(f"Model Memory Overhead: {end_mem - start_mem:.2f} MB")
        
        # Input
        input_ids = [1] * seq_len
        # Batching for training
        
        # --- INFERENCE LATENCY (Batch 1) ---
        print("Running Inference Benchmark (Batch 1)...")
        ctx = nsos_ext.Context()
        
        # Warmup
        model.forward_ids(input_ids, ctx)
        
        iterations = 10 # Reduced from 50
        start_t = time.perf_counter()
        for _ in range(iterations):
            model.forward_ids(input_ids, ctx)
        end_t = time.perf_counter()
        
        avg_latency = (end_t - start_t) / iterations * 1000 # ms
        tokens_per_sec = (seq_len * iterations) / (end_t - start_t)
        
        print(f"Latency (B=1, L={seq_len}): {avg_latency:.2f} ms")
        print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
        
        # --- TRAINING THROUGHPUT (Batch 32 simulation) ---
        # Note: Tensor creation might be bottleneck on CPU if not careful
        # We simulate training load by running forward on larger tensor
        
        print("Running Training Load Benchmark (Tensor Forward)...")
        batch_train = 32 # Restored to 32
        x = nsos_ext.Tensor.random([batch_train, seq_len, d_model], device_enum)
        
        # Warmup
        model.forward(x, ctx)
        
        start_t = time.perf_counter()
        for _ in range(5): # Reduced from 10
            out = model.forward(x, ctx)
            # backward would be here
        end_t = time.perf_counter()
        
        train_tokens_sec = (batch_train * seq_len * 5) / (end_t - start_t)
        print(f"Training Throughput: {train_tokens_sec:.2f} tokens/sec")

    except Exception as e:
        print(f"FAILED on {device_name}: {e}")

if __name__ == "__main__":
    print(f"NSOS Extension Loaded. Validating devices...")
    
    # Test CPU (Light check only)
    print("\n--- Skipping Heavy CPU Benchmark (Run GPU focus) ---")
    # benchmark_device("CPU", nsos_ext.Device.CPU)
    
    # Test GPU (if compiled)
    # Check if GPU is available (simple try)
    print("\nAttempting GPU Benchmark...")
    try:
        t = nsos_ext.Tensor.zeros([1], nsos_ext.Device.GPU)
        benchmark_device("GPU", nsos_ext.Device.GPU)
    except Exception as e:
        print(f"GPU Not Available or Failed: {e}")
