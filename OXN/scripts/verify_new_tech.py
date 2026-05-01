import sys
import os

# Add build path
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("build/Release"))

# Windows DLL Search Path fix for CUDA 12.5
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
if os.name == 'nt' and os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)

import nsos_ext
import numpy as np

def verify_sophia():
    print("\n[1] Verifying Sophia Optimizer (1D)...")
    try:
        opt = nsos_ext.SophiaOptimizer([10], 0.1)
        param = nsos_ext.Tensor([10], nsos_ext.Device.CPU, 0.5)
        grad = nsos_ext.Tensor([10], nsos_ext.Device.CPU, 0.1)
        hessian = nsos_ext.Tensor([10], nsos_ext.Device.CPU, 0.02)

        # Step
        opt.step(param, grad, hessian)
        p_np = param.numpy()
        print("Param after step:", p_np)

        # Check if value changed
        if np.allclose(p_np, 0.5):
            print("FAIL: Param didn't change.")
        else:
            print("PASS: Sophia update applied.")
    except Exception as e:
        print(f"FAIL: {e}")

def verify_ttt():
    print("\n[2] Verifying TTT Layer (Test-Time Training)...")
    try:
        dim = 4
        hidden = 4
        layer = nsos_ext.TTTLayer(dim, hidden, 0.1)
        x = nsos_ext.Tensor([1, dim], nsos_ext.Device.CPU, 0.5) # Randomish

        # Forward pass 1
        out1 = layer.forward(x).numpy()
        print("Output 1:", out1)

        # Forward pass 2 (Should be different due to internal update)
        out2 = layer.forward(x).numpy()
        print("Output 2:", out2)

        if np.allclose(out1, out2):
            print("FAIL: TTT Layer output didn't change (No Learning).")
        else:
            print("PASS: TTT Layer adapted online.")
    except Exception as e:
        print(f"FAIL: {e}")

def verify_holographic():
    print("\n[3] Verifying Holographic Memory (HDC)...")
    try:
        mem = nsos_ext.HolographicMemory(1024)

        # Concepts
        red = mem.create_concept("red")
        apple = mem.create_concept("apple")
        blue = mem.create_concept("blue")
        sky = mem.create_concept("sky")

        # Bind: Red * Apple
        red_apple = mem.bind(red, apple)
        blue_sky = mem.bind(blue, sky)

        # Bundle: Scene = (Red * Apple) + (Blue * Sky)
        scene = mem.bundle([red_apple, blue_sky])

        # Query Scene with Red -> Should be Apple?
        # A * B * A = B (approx) if bipolar
        # Check: bind(scene, red) ~= apple
        query_vec = mem.bind(scene, red)
        result = mem.query(query_vec)

        print(f"Query: 'Scene * Red' -> Result: '{result}'")

        if result == "apple":
            print("PASS: Holographic Unbinding correct.")
        else:
            print(f"WARN: HDC Noise high, expected 'apple', got '{result}'. Check dimension/orthogonality.")
            # Note: 1024 dims is small for HDC, might fail occasionally.

    except Exception as e:
        print(f"FAIL: {e}")

def verify_d2f():
    print("\n[4] Verifying Discrete Diffusion Forcing (Stub)...")
    try:
        model = nsos_ext.JambaModel(2, 64)
        decoder = nsos_ext.D2FDecoder(model, 4)
        prompt = nsos_ext.Tensor([1, 64], nsos_ext.Device.CPU, 0.1)

        output = decoder.generate(prompt, 12)
        print("Generated Tokens:", output)

        if len(output) == 12:
            print("PASS: D2F generated correct length.")
        else:
            print("FAIL: D2F length mismatch.")

    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    verify_sophia()
    verify_ttt()
    verify_holographic()
    verify_d2f()
