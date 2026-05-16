import sys
import os
import numpy as np

# Add OXN dir to path
sys.path.append(r"c:\Users\VitorGGc\Desktop\Pantheon-Oxtav1-15338770387506525964\OXN")

try:
    import nsos_ext
except ImportError:
    print("Erro: nsos_ext não encontrado.")
    sys.exit(1)

d_model = 16
vocab_size = 128

t = nsos_ext.Tensor([d_model, vocab_size], nsos_ext.Device.CPU, 0.0)
print(f"Tensor shape from property: {t.shape}")
print(f"Tensor size from property: {t.size}")

arr = np.asarray(t)
print(f"NumPy array shape: {arr.shape}")
print(f"NumPy array ndim: {arr.ndim}")
print(f"NumPy array dtype: {arr.dtype}")

# Try to modify
arr.fill(1.0)
print(f"First element: {arr.flat[0]}")
