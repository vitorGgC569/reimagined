
import nsos_ext as nsos
import numpy as np

def diag():
    print("--- NSOS Layer-by-Layer Diagnostics ---")
    
    dim = 128
    vocab = 256
    seq = 64
    batch = 1
    
    # 1. Test Embedding
    print("\n[Layer 1] Embedding")
    emb = nsos.Embedding(vocab, dim)
    import random
    ids = [random.randint(0, 255) for _ in range(seq)]
    x = emb.forward(ids)
    print(f"  Output shape: {x.shape} (Expected: [64, 128])")
    
    # 2. Test BitLinear
    print("\n[Layer 2] BitLinear (Projection)")
    lin = nsos.BitLinear(dim, dim)
    y = lin.forward(x)
    print(f"  Output shape: {y.shape} (Expected: [64, 128])")
    
    # 3. Test Attention
    print("\n[Layer 3] Attention")
    attn = nsos.Attention(dim, 8)
    # Attention expects 3D [B, S, D]
    x_3d = x.reshape([1, seq, dim])
    try:
        y_attn = attn.forward(x_3d, None)
        print(f"  Output shape: {y_attn.shape} (Expected: [1, 64, 128])")
    except Exception as e:
        print(f"  Attention Forward FAILED: {e}")
        return

    # 4. Test Mamba2SSD
    print("\n[Layer 4] Mamba2SSD")
    mamba = nsos.Mamba2SSD(dim, 16, 8)
    try:
        y_mamba = mamba.forward(x_3d, None)
        print(f"  Output shape: {y_mamba.shape} (Expected: [1, 64, 128])")
    except Exception as e:
        print(f"  Mamba2 Forward FAILED: {e}")
        return

    # 5. Test Backward
    print("\n[Gradients] Backward Trace")
    ctx = nsos.Context()
    model = nsos.JambaModel(2, dim, vocab, nsos.Device.CPU)
    
    # Run full path
    output = model.forward_ids(ids, ctx)
    print(f"  Model output shape: {output.shape}")
    
    # Simulate loss
    targets = [random.randint(0, 255) for _ in range(seq)]
    loss, grad = output.cross_entropy(targets)
    print(f"  Initial Loss: {loss}")
    
    # Backward
    try:
        model.backward(grad, ctx)
        print("  Backward pass COMPLETED.")
    except Exception as e:
        print(f"  Backward pass FAILED: {e}")
        return
        
    # Check Gradients
    params = model.parameters()
    has_grad = 0
    total_grad_norm = 0
    for i, p in enumerate(params):
        if p.grad is not None:
            has_grad += 1
            total_grad_norm += p.grad.norm()
            if i < 3: # Print first few
                print(f"  Param {i} ({p.name}) Grad Norm: {p.grad.norm()}")
                
    print(f"\nSummary: {has_grad}/{len(params)} parameters have gradients.")
    print(f"Total Grad Norm: {total_grad_norm}")

if __name__ == "__main__":
    diag()
