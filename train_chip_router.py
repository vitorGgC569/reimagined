import sys
import os
import time
import random
import numpy as np
from tqdm import tqdm
from chip_env import ChipEnv

# Setup NSOS path
build_dir = os.path.join(os.path.dirname(__file__), "OXN", "build", "Release")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)
import nsos_ext as nsos

# CONFIG
CONFIG = {
    "dim": 128, 
    "layers": 3, # 3 camadas é o sweet spot para este kernel sem RMSNorm nativo
    "grid_window": 7,
    "actions": 4, 
    "lr": 0.001,
    "steps": 100 
}

def train_router():
    print("--- OXN-EDA Chip Router Training (Deep Stable Mode) ---")
    ctx_global = nsos.Context()
    env = ChipEnv(size=16)
    
    model = nsos.JambaModel(CONFIG['layers'], CONFIG['dim'], 128)
    head = nsos.BitFastKANLayer(CONFIG['dim'], CONFIG['actions'])
    
    # Forçar inicialização ULTRA-CONSERVADORA
    params = list(model.parameters()) + [head.base_weight, head.rbf_weight]
    for p in params:
        if p.data is not None:
            # 1e-5 para garantir que a soma das residuais não estoure
            p.data.copy_from(p.data.mul(1e-5)) 
    
    total_loss = 0
    step_i = 0
    pbar = tqdm(total=CONFIG['steps'], desc="Routing Learning")
    
    for episode in range(CONFIG['steps']):
        obs = env.reset() 
        done = False
        step_count = 0
        
        while not done and step_count < 32:
            r, c = env.current_pos
            tr, tc = env.end_pos
            
            if abs(tr - r) > abs(tc - c):
                best_action = 0 if tr < r else 1
            else:
                best_action = 2 if tc < c else 3
            
            input_tensor = nsos.Tensor.zeros([1, 1, CONFIG['dim']])
            input_data = input_tensor.cpu().numpy()
            for idx, val in enumerate(obs):
                if idx < CONFIG['dim']:
                    input_data[0, 0, idx] = float(val) / 32.0 
                
            input_tensor.copy_from(nsos.Tensor.from_blob(input_data.ctypes.data, [1, 1, CONFIG['dim']], nsos.Device.CPU))
            
            ctx = nsos.Context()
            h = model.forward(input_tensor, ctx)
            
            # Normalização robusta do hidden state
            h_np = h.cpu().numpy()
            norm = np.linalg.norm(h_np) + 1e-8
            h_np = h_np / (norm / np.sqrt(CONFIG['dim']))
            h.copy_from(nsos.Tensor.from_blob(np.clip(h_np, -1.0, 1.0).astype(np.float32).ctypes.data, h.shape, nsos.Device.CPU))
            
            logits = head.forward(h.reshape([1, CONFIG['dim']]))
            
            loss_val, d_logits = logits.cross_entropy([best_action])
            if np.isnan(float(loss_val)) or np.isinf(float(loss_val)) or float(loss_val) > 10.0:
                break 
                
            total_loss += loss_val
            d_h = head.backward(d_logits).reshape([1, 1, CONFIG['dim']])
            model.backward_external(d_h, ctx)
            
            for p in params:
                if p.grad is not None and p.grad.size > 0:
                    gnorm = p.grad.norm()
                    if gnorm > 0.05: # Clipping severo
                        p.grad.copy_from(p.grad.mul(0.05 / (gnorm + 1e-6)))
                    
                    new_data = p.data.sub(p.grad.mul(CONFIG['lr']))
                    p.data.copy_from(new_data)
                    p.grad = nsos.Tensor.zeros(p.grad.shape, p.grad.device)
            
            obs, reward, done = env.step(best_action)
            step_count += 1
            step_i += 1
            
            if step_i % 10 == 0:
                logits_np = logits.cpu().numpy()[0]
                l_max = np.max(logits_np)
                pbar.set_postfix({
                    'ep': episode,
                    'loss': f'{float(loss_val):.4f}', 
                    'l_max': f'{l_max:.1f}'
                })
        pbar.update(1)

    print("\n✅ Training Complete. Model internalizing routing heuristics.")
    
    # SALVAR CEREBRO
    print("\n💾 Salvando 'Cérebro' do Roteador...")
    model.save("router_trunk.bin")
    np.save("head_base.npy", head.base_weight.data.cpu().numpy())
    np.save("head_rbf.npy", head.rbf_weight.data.cpu().numpy())
    print("✅ Pesos salvos: router_trunk.bin, head_base.npy, head_rbf.npy")
    
    # Teste de Inferência
    print("\n🚀 Verificando Roteamento Autônomo...")
    obs = env.reset()
    env.render()
    
    for _ in range(10):
        input_tensor = nsos.Tensor.zeros([1, 1, CONFIG['dim']])
        input_data = input_tensor.cpu().numpy()
        for idx, val in enumerate(obs):
            if idx < CONFIG['dim']:
                input_data[0, 0, idx] = float(val)
        
        input_tensor.copy_from(nsos.Tensor.from_blob(input_data.ctypes.data, [1, 1, CONFIG['dim']], nsos.Device.CPU))
        
        h = model.forward(input_tensor, None)
        logits = head.forward(h.reshape([1, CONFIG['dim']]))
        action = int(np.argmax(logits.cpu().numpy()[0]))
        
        print(f"Agente decidiu mover: {action}")
        obs, _, done = env.step(action)
        if done: break
    
    env.render()

if __name__ == "__main__":
    train_router()
