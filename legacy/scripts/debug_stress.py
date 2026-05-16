import sys
import os
import numpy as np
import time

# Add possible paths for the extension
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'build'))

try:
    import nsos_ext
except ImportError:
    print("Could not import nsos_ext. Attempting to build or locate it...")
    sys.exit(1)

def test_layer_count():
    print("\n--- Test 1: Layer Count ---")
    model = nsos_ext.JambaModel(1, 128, 256) # 1 Layer, 128 Dim, 256 Vocab
    params = model.parameters()
    print(f"Number of parameters objects: {len(params)}")
    names = []
    # Since Parameter is a C++ struct exposed, we might not get names easily if not stored in Python list
    # But JambaModel::parameters returns list of pointers.
    # In bindings, we didn't see 'name' exposed on Parameter, but the C++ code sets it.
    # Let's see if we can infer from count.

    # 1 Layer (TTT only per logic? or Mamba?)
    # If 1 layer, index 0. is_ttt = (0 == 0) -> True.
    # JambaBlock(is_ttt=True).
    # Creates TTTLayer.
    # TTTLayer has W_hidden, b_hidden (maybe), velocity (state, not param).
    # JambaModel::parameters collects:
    # Embedding weight.
    # Loop layers:
    # If TTT: TTTLayer is NOT added to parameters() in the C++ code I read!
    # Wait, read JambaModel::parameters again.

    # "if (layer->mamba_layer) ... if (layer->attn_layer) ... if (layer->ffn) ..."
    # It does NOT check layer->ttt_layer!
    # So TTT parameters are MISSING from parameters()!
    # That explains why "Kernel ignores parameters" maybe? Or why it seems empty.

    print("Checking if TTT params are present...")
    # If len(params) == 1 (Embedding only), then TTT is ignored.
    if len(params) == 1:
        print("FAIL: Only embedding found. TTT Layer parameters are missing from JambaModel::parameters()!")
    else:
        print(f"Pass? Found {len(params)} params.")

def test_ttt_stability():
    print("\n--- Test 2: TTT Stability (NaN Explosion) ---")
    D = 64
    ttt = nsos_ext.TTTLayer(D, D, 0.01) # lr = 0.01

    # Input: [Batch=16, Seq=10, Dim=64] -> Flat [160, 64]
    # Large inputs
    x = nsos_ext.Tensor([16, 10, D], nsos_ext.Device.CPU)
    x_np = np.random.normal(0, 2.0, (16, 10, D)).astype(np.float32) # Variance 4
    # Fill tensor
    # We rely on buffer protocol or manual fill
    # Extension has numpy() support?
    # .def("numpy", ...)

    # Write via numpy view
    x_view = np.array(x.numpy(), copy=False)
    x_view[:] = x_np

    print("Running TTT steps...")
    for i in range(50):
        out = ttt.forward(x)
        norm = out.norm()
        if np.isnan(norm) or norm > 1e6:
            print(f"FAIL: TTT Exploded at step {i}. Norm: {norm}")
            return
    print("TTT Stability Check Passed (or didn't explode in 50 steps).")

def test_memory_heap():
    print("\n--- Test 3: Memory/Heap Stress ---")
    model = nsos_ext.JambaModel(2, 64, 128)

    # Random input loop
    for i in range(20):
        # Varying sequence length
        seq = 10 + (i % 10)
        input_ids = [np.random.randint(0, 128) for _ in range(seq)]

        # We need to pass Tensor to forward?
        # JambaModel::forward takes Tensor.
        # But JambaModel expects embeddings?
        # "forward_embedding" calls "embedding->forward(indices)"?
        # No, JambaModel::forward(x) calls forward_embedding(x).
        # JambaModel::forward_embedding(x) takes Tensor x (embeddings).
        # So we must manually embed first?
        # "embedding" property exposed.

        emb_layer = model.embedding
        # Embedding::forward takes std::vector<int>

        # Bindings: .def("forward", &Embedding::forward)
        emb_out = emb_layer.forward(input_ids)

        # Now pass to model
        try:
            ctx = nsos_ext.Context()
            out = model.forward(emb_out, ctx)
        except Exception as e:
            print(f"FAIL: Crash at step {i}: {e}")
            return

    print("Heap Stress Passed.")

def test_kan_bounds():
    print("\n--- Test 4: KAN Bounds ---")
    kan = nsos_ext.BitFastKANLayer(10, 10, 5)

    # Input size mismatch
    # Expected [B, 10]. Pass [B, 12].
    try:
        x = nsos_ext.Tensor([5, 12], nsos_ext.Device.CPU) # 12 features
        out = kan.forward(x)
        print("FAIL: KAN accepted wrong input dimension!")
    except Exception as e:
        print(f"Pass: KAN caught mismatch: {e}")
        # Note: If it's a segfault, python won't catch it easily without a wrapper, but let's try.

if __name__ == "__main__":
    test_layer_count()
    test_ttt_stability()
    test_memory_heap()
    test_kan_bounds()
