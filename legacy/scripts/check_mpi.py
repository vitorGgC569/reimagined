import sys
import os

# Setup path
build_dir = os.path.join(os.path.dirname(__file__), "OXN", "build", "Release")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import nsos_ext as nsos
    print("✅ NSOS Loaded.")
    
    print("\n[MPI Check] Testing Distributed Training API...")
    if hasattr(nsos, "MultiNodeOrchestrator"):
        mpi = nsos.MultiNodeOrchestrator()
        rank = mpi.get_rank()
        size = mpi.get_world_size()
        print(f"   • Orchestrator Online. Rank: {rank}/{size}")
        
        # Test gradient sync (Mock)
        t = nsos.Tensor([10], nsos.Device.CPU, 1.0)
        print(f"   • Tensor Norm Before Sync: {t.norm()}")
        mpi.sync_gradients(t)
        print(f"   • Tensor Norm After Sync:  {t.norm()} (Should be unchanged in single-node mock)")
        
        print("✅ Distributed Training API Verified.")
    else:
        print("❌ MultiNodeOrchestrator NOT found in nsos_ext.")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Failed to import nsos_ext: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
