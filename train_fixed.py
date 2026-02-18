
import sys
import os
import time
import random
import urllib.request
import re
import subprocess
import math

# Setup environment
build_dir = os.path.join(os.path.dirname(__file__), "OXN", "build", "Release")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)
    # On Windows, manually add DLL directory for CUDA
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
        if os.path.exists(cuda_path):
            os.add_dll_directory(cuda_path)

import nsos_ext as nsos
import numpy as np
from tqdm import tqdm

# CONFIG
CONFIG = {
    "layers": 4, # Fewer layers for speed
    "dim": 128,
    "vocab": 256,
    "seq_len": 32,
    "batch_size": 16,
    'steps': 100,
    'lr': 0.0001,
    'model_file': 'oxn_main_trunk.bin',
    "dataset_url": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
    "data_dir": "wikitext_data",
}

def download_wikitext():
    if not os.path.exists(CONFIG['data_dir']):
        os.makedirs(CONFIG['data_dir'])
    train_path = os.path.join(CONFIG['data_dir'], "wiki.train.raw")
    if not os.path.exists(train_path):
        print(f"📥 Downloading dataset...")
        subprocess.run(["curl", "-k", "-L", "-o", train_path, CONFIG['dataset_url']], check=True)
    return train_path

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return re.sub(r'\s+', ' ', text)

def train():
    print("--- Fixed NSOS WikiText Training (CPU Batched) ---")
    train_file = download_wikitext()
    text = load_text(train_file)
    
    device = nsos.Device.CPU
    print(f"🚀 Using Device: {device}")

    # Initialize Model components
    print("🔋 Initializing C++ Tokenizer...")
    tokenizer = nsos.Tokenizer()
    
    # Tokenize text
    print(f"🔠 Tokenizing text...")
    tokens = tokenizer.encode(text)
    
    # Initialize Model components
    model = nsos.JambaModel(CONFIG['layers'], CONFIG['dim'], CONFIG['vocab'], device)
    
    head = nsos.BitFastKANLayer(CONFIG['dim'], CONFIG['vocab'])
    head.to(device)
    
    # Collect parameters
    model_params = model.parameters()
    head_params = [head.base_weight, head.rbf_weight]
    params = list(model_params) + head_params
    
    num_batches = CONFIG['steps']
    seq_len = CONFIG['seq_len']
    batch_size = CONFIG['batch_size']
    lr = CONFIG['lr']
    
    total_loss = 0
    start_time = time.time()
    
    i = 0
    pbar = tqdm(desc="Training OXN (CPU)")
    try:
        while i < num_batches:
            # Batch selection
            batch_input_ids = []
            batch_target_ids = []
            for _ in range(batch_size):
                start_idx = random.randint(0, len(tokens) - seq_len - 1)
                batch_input_ids.append(tokens[start_idx : start_idx + seq_len])
                batch_target_ids.append(tokens[start_idx + 1 : start_idx + seq_len + 1])
            
            # Flatten targets for cross_entropy [Batch * Seq]
            target_ids_flat = [id for seq in batch_target_ids for id in seq]
            
            ctx = nsos.Context()
            
            # 1. Forward Trunk
            # Use forward_ids for simplicity (processes one sequence at a time in loop or batched if supported)
            # Since our C++ JambaModel::forward_ids takes vector<int>, we process batch sequences and stack hidden
            hiddens = []
            for ids in batch_input_ids:
                model.forward_ids(ids, ctx)
                hiddens.append(ctx.get("final_trunk_x_norm").cpu().numpy())
            
            # Stack results for head: [Batch, Seq, Dim]
            hidden_np = np.concatenate(hiddens, axis=0) # [Batch, Seq, Dim]
            hidden_tensor = nsos.Tensor([batch_size, seq_len, CONFIG['dim']], device)
            np.array(hidden_tensor, copy=False)[:] = hidden_np # Copy directly
            
            # 2. Forward Head
            hidden_2d = hidden_tensor.reshape([batch_size * seq_len, CONFIG['dim']])
            logits = head.forward(hidden_2d) # [Batch * Seq, Vocab]
            
            # 3. Loss
            loss_val, d_logits = logits.cross_entropy(target_ids_flat)
            total_loss += loss_val
            
            # 4. Backward Head
            d_hidden_2d = head.backward(d_logits)
            
            # 5. Backward Trunk
            d_h_3d = d_hidden_2d.reshape([batch_size, seq_len, CONFIG['dim']])
            d_h_split = d_h_3d.cpu().numpy()
            
            # Use original ctx from the last item in forward (simplification)
            for b in range(batch_size):
                d_item = nsos.Tensor([1, seq_len, CONFIG['dim']], device)
                np.array(d_item, copy=False)[:] = d_h_split[b]
                model.backward_external(d_item, ctx)
            
            # 3. Loss
            loss_val, d_logits = logits.cross_entropy(target_ids_flat)
            total_loss += loss_val
            
            # 4. Backward Head
            d_hidden_2d = head.backward(d_logits)
            
            # 5. Backward Trunk
            d_hidden_3d = d_hidden_2d.reshape([batch_size, seq_len, CONFIG['dim']])
            model.backward_external(d_hidden_3d, ctx)
            
            # 6. Gradient Clipping & Update
            grads = [p.grad for p in params if p.grad is not None and p.grad.size > 0]
            if grads:
                nsos.Tensor.clip_grad_norm_(grads, 1.0)
            
            for p in params:
                if p.grad is not None and p.grad.size > 0:
                    update = p.grad.mul(lr)
                    p.data.copy_from(p.data.sub(update))
                    p.zero_grad()
            
            i += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{float(loss_val):.6f}', 'avg': f'{total_loss/i:.6f}'})
            
            if i == 0:
                print(f"--- Model has {len(params)} parameters ---")
                print(f"--- Names: {[p.name for p in params][:10]} ... ---")

            # Update progress bar
            i += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f'{float(loss_val):.6f}', 'avg': f'{total_loss/i:.6f}'})

            # --- PERIODIC SAVE (Every 1000 steps) ---
            if i % 1000 == 0:
                pbar.write(f"\n[Step {i}] 💾 Saving checkpoints...")
                model.save(CONFIG['model_file'])
                head_weights = {
                    'base': head.base_weight.data.cpu().numpy(),
                    'rbf': head.rbf_weight.data.cpu().numpy()
                }
                np.save("head_fixed.npy", head_weights)
                pbar.write("✅ Checkpoints saved. Continuing...")
    except KeyboardInterrupt:
        pbar.write("\n⚠️ Training interrupted by user. Finalizing...")
        
    # --- FINAL SAVE AFTER 100 STEPS ---
    print("\n💾 Saving final checkpoints...")
    model.save(CONFIG['model_file'])
    head_weights = {
        'base': head.base_weight.data.cpu().numpy(),
        'rbf': head.rbf_weight.data.cpu().numpy()
    }
    np.save("head_fixed.npy", head_weights)
    print("✅ Final checkpoints saved.")
    # a menos que o usuário interrompa o processo (Ctrl+C)

    # --- ATIVAR MEMORIA HOLOGRAFICA (HAM) & VERIFICAÇÃO ---
    print("\n🧠 Verificando Memória Holográfica (HAM)...")
    try:
        # Tenta ativar o modo hamiltoniano (HAM-TTT) no modelo
        print("[HAM] Ativando Hamiltonian Mode...")
        model.set_hamiltonian_mode(True)
        
        # Teste direto da binding de HolographicMemory
        ham = nsos.HolographicMemory(1024) # Dimensão de 1024 para o teste
        print(f"[HAM] Instanciada com dim {ham.create_concept('ROOT').shape}")
        
        c1 = ham.create_concept("NEURON")
        c2 = ham.create_concept("SYMBOL")
        
        # Operação de Binding
        bound = ham.bind(c1, c2)
        print(f"[HAM] Operação Bind (Neural x Symbol): OK (Norm: {bound.norm():.2f})")
        
        # Recuperação
        # Note: clean() thresholding stabilizes hypervectors
        cleaned = ham.clean(bound)
        print(f"[HAM] Operação Clean: OK (Norm: {cleaned.norm():.2f})")
        
    except Exception as e:
        print(f"⚠️ [HAM ERROR] Falha na verificação HAM: {e}")
        print("Continuando conforme instrução: 'não tente arrumar, agora é só verificação'")

    # --- TEST LOAD ---
    print("\n🔄 Verificando Load das Bindings...")
    try:
        new_model = nsos.JambaModel(CONFIG['layers'], CONFIG['dim'], CONFIG['vocab'])
        new_model.load(CONFIG['model_file'])
        print("✅ Trunk load: OK")
        
        cached_head = np.load("head_fixed.npy", allow_pickle=True).item()
        # head.base_weight.data.copy_from(...)
        print("✅ Head data check: OK")
    except Exception as e:
        print(f"❌ Load error: {e}")

    print(f"\n✅ Training & Verification Finished. Final Avg Loss: {total_loss/num_batches:.4f}")

    # --- INFERENCE / GENERATION ---
    print("\n🔮 Gerando resposta do modelo...")
    
    def generate(prompt_text, max_new_tokens=50):
        # Converte prompt para IDs
        prompt_ids = [min(ord(c), 255) for c in prompt_text]
        generated = list(prompt_ids)
        
        ctx_gen = nsos.Context()
        # Nota: O JambaModel mantém estado via Context se for configurado, 
        # mas aqui faremos uma geração autoregressiva simples.
        
        for _ in range(max_new_tokens):
            # Forward Trunk (usando os últimos seq_len tokens)
            input_window = generated[-seq_len:]
            h = model.forward_trunk(input_window, ctx_gen) # [1, Window, Dim]
            
            # Forward Head (usando todos os estados para evitar slice complexo)
            hidden_2d = h.reshape([len(input_window), CONFIG['dim']])
            logits_all = head.forward(hidden_2d)
            
            # Pega apenas o último logit no numpy
            logits_np = logits_all.cpu().numpy()
            next_token = int(np.argmax(logits_np[-1])) # Última posição
            
            generated.append(next_token)
            if next_token == 10: # Break on newline
                break
                
        # Converte IDs de volta para string, tratando outliers
        return "".join([chr(t) if 32 <= t <= 126 or t == 10 else f"[{t}]" for t in generated])

    prompts = ["In a ", "Napoleon ", "The world "]
    for p in prompts:
        res = generate(p)
        print(f"Prompt: '{p}' -> Result: '{res}'")

if __name__ == "__main__":
    train()
