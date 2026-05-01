import sys
import os

# Setup environment
sys.path.append(os.path.abspath("OXN/build/Release"))
print(f"Path: {sys.path[-1]}")

import nsos_ext as nsos

print("🧪 Testing Parameter Update...")
try:
    model = nsos.JambaModel(2, 64, 10, nsos.Device.CPU)
    params = model.parameters()
    p = params[0]
    
    initial_norm = p.data.norm()
    print(f"Initial Norm: {initial_norm}")
    
    # Try distinct update methods
    print("Method 1: p.data = new_data")
    new_data = p.data.mul(0.5)
    p.data = new_data
    print(f"Norm after M1: {p.data.norm()}")
    
    # Re-fetch params to see if model noticed
    params_refetched = model.parameters()
    p_refetched = params_refetched[0]
    print(f"Model Norm after M1: {p_refetched.data.norm()}")
    
    if abs(p_refetched.data.norm() - initial_norm) < 1e-4:
        print("❌ Method 1 failed (expected behavior if distinct objects)")
    else:
        print("✅ Method 1 worked!")

    print("\nMethod 2: p.data.copy_from(new_data)")
    target_data = p.data.mul(0.5)
    p.data.copy_from(target_data)
    print(f"Norm after M2: {p.data.norm()}")
    
    params_refetched2 = model.parameters()
    p_refetched2 = params_refetched2[0]
    print(f"Model Norm after M2: {p_refetched2.data.norm()}")
    
    if abs(p_refetched2.data.norm() - p.data.norm()) < 1e-4 and abs(p.data.norm() - initial_norm) > 1e-4:
         print("✅ Method 2 worked!")
    else:
         print("❌ Method 2 failed!")

except Exception as e:
    print(e)
