import sys
import os
import time
import numpy as np

# Robust DLL and Path setup for Windows
script_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
build_release = os.path.join(root_dir, "build/Release")

sys.path.append(root_dir)
sys.path.append(build_release)

if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    if os.path.exists(build_release):
        os.add_dll_directory(os.path.abspath(build_release))
    # Standard CUDA paths
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
    ]
    for p in cuda_paths:
        if os.path.exists(p):
            os.add_dll_directory(p)

import nsos_ext

def run_benchmark(num_layers=8, d_model=256, seq_len=128, batch_size=4, num_steps=50):
    print(f"=== NSOS PERFORMANCE BENCHMARK (HAM ENABLED) ===")
    print(f"Config: Layers={num_layers}, D_Model={d_model}, Seq={seq_len}, Batch={batch_size}")
    
    device = nsos_ext.Device.GPU
    model = nsos_ext.JambaModel(num_layers, d_model, 1000, device)
    
    # Dummy input
    x_data = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    cpu_x = nsos_ext.Tensor.from_blob(x_data.ctypes.data, [batch_size, seq_len, d_model], nsos_ext.Device.CPU)
    x = cpu_x.to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        model.forward(x)
    
    print(f"Running {num_steps} iterations...")
    start_time = time.time()
    
    for i in range(num_steps):
        model.forward(x)
        if (i+1) % 10 == 0:
            print(f"Step {i+1}/{num_steps}...")
            
    end_time = time.time()
    total_time = end_time - start_time
    
    total_tokens = batch_size * seq_len * num_steps
    tokens_per_sec = total_tokens / total_time
    ms_per_token = (total_time / total_tokens) * 1000
    
    print("\n=== RESULTS ===")
    print(f"Total Time: {total_time:.2f} s")
    print(f"Total Tokens: {total_tokens}")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print(f"Latency: {ms_per_token:.4f} ms/token")
    
    # Check if we can access ham property now
    try:
        current_mem = model.ham.get_current_size()
        print(f"Memory Matrix Size: {current_mem} concepts")
    except Exception as e:
        print(f"Memory stats check failed: {e}")
    
if __name__ == "__main__":
    run_benchmark()
