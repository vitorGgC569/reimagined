
import sys
import os
import traceback

sys.path.append(os.path.abspath("build"))

print("--- DEBUGGING IMPORTS ---")

print("\n1. Testing AION Core...")
try:
    import aion_core
    print("SUCCESS: AION Core Loaded.")
except ImportError:
    print("FAILURE: AION Core Import Failed.")
    traceback.print_exc()

print("\n2. Testing NSOS Ext...")
try:
    import nsos_ext
    print("SUCCESS: NSOS Ext Loaded.")
except ImportError:
    print("FAILURE: NSOS Ext Import Failed.")
    traceback.print_exc()

print("\n3. Testing UHK Graph...")
try:
    import uhk_graph
    print("SUCCESS: UHK Graph Loaded.")
except ImportError:
    print("FAILURE: UHK Graph Import Failed.")
    traceback.print_exc()
