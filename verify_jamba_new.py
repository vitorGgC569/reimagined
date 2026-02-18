import nsos
import torch
import numpy as np

def test_jamba_backward():
    print("Testing Jamba Attention & MoE Backward...")
    d_model = 128
    vocab_size = 100
    model = nsos.JambaModel(layers=2, d_model=d_model, vocab_size=vocab_size)
    
    # Simular entrada
    batch_size = 2
    seq_len = 8
    x_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).int().cpu().numpy().tolist()
    
    # Forward pass com contexto para salvar ativações
    ctx = nsos.Context()
    # Note: JambaModel::forward_ids usually creates its own context or uses provided one
    # If the bindings support it:
    logits = model.forward_ids(x_ids, ctx)
    
    print(f"Forward pass successful. Output shape: {logits.shape}")
    
    # Simular gradiente
    dy = nsos.Tensor.random(logits.shape)
    
    # Backward pass
    try:
        dx = model.backward(dy, ctx)
        print("Backward pass successful!")
        
        # Verificar se os parâmetros receberam gradientes
        params = model.parameters()
        has_grads = any(p.grad.norm() > 0 for p in params if p.grad is not None)
        if has_grads:
            print("Verified: Parameters have non-zero gradients.")
        else:
            print("Warning: All parameter gradients are zero!")
            
    except Exception as e:
        print(f"Backward pass failed: {e}")

def test_moe_efficiency():
    print("\nTesting MoE Routing Efficiency...")
    d_model = 128
    router = nsos.MoERouter(d_model=d_model, n=16, k=2)
    
    # Grande número de tokens para testar O(N log K)
    tokens = 1024
    x = nsos.Tensor.random([tokens, d_model])
    
    import time
    start = time.time()
    indices, weights = router.forward(x)
    end = time.time()
    
    print(f"MoE Forward (1024 tokens) took: {1000*(end-start):.2f}ms")
    print(f"Indices shape: {indices.shape}")

if __name__ == "__main__":
    test_jamba_backward()
    test_moe_efficiency()
    print("\nJamba Verification Complete.")
