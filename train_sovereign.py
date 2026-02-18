import os
import sys
import time

# Absolute path to the LATEST compiled extension
BUILD_DIR = r"C:\Users\VitorGGc\Desktop\Pantheon-Oxtav1-15338770387506525964\OXN\build\Release"
sys.path.insert(0, BUILD_DIR)

# Add DLL directory for CUDA/MSVC components
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
if sys.platform == "win32":
    if os.path.exists(cuda_path):
        os.add_dll_directory(cuda_path)
    os.add_dll_directory(BUILD_DIR)

import nsos_ext as nsos
print(f"✅ Loaded nsos_ext (Circuit Engine SOTA) from: {nsos.__file__}")
from tqdm import tqdm

def train_circuit_mvp_sovereign():
    print("="*80)
    print("⚡ NSOS CIRCUIT-OXN MVP TRAINING - TOKEN-DIRECT SOTA")
    print("="*80)
    
    CONFIG = {
        "layers": 6,
        "dim": 128,
        "vocab": 256,
        "batch_size": 1,
        "seq_len": 64,
        "epochs": 10,
        "lr": 0.0005,
        "dataset": "circuit_data/train.raw"
    }
    
    device = nsos.Device.CPU
    
    # 1. Initialize Model
    model = nsos.JambaModel(CONFIG['layers'], CONFIG['dim'], CONFIG['vocab'], device)
    
    # 2. Tokenize Dataset in Python (Maximum Stability)
    print(f"📥 Loading and Tokenizing {CONFIG['dataset']}...")
    with open(CONFIG['dataset'], 'r') as f:
        text = f.read()
    
    # Byte-level encoding (SOTA for circuits)
    tokens = [ord(c) for c in text if ord(c) < 256]
    print(f"✅ Total Tokens: {len(tokens)}")
    
    # 3. Initialize Industrial Trainer
    trainer = nsos.Trainer(model, CONFIG['lr'])
    
    pbar = tqdm(total=5000, desc="SOTA Training")
    
    def on_step(step, loss):
        # The C++ engine calls this every 10 steps.
        # We update the bar by 10 to show real 'circuits' processed.
        pbar.update(10)
        pbar.set_postfix({'loss': f'{loss:.4f}'})

    try:
        # Note: The stop logic will be handled by the trainer loop completion or manual Ctrl+C
        trainer.train_loop(tokens, CONFIG['epochs'], CONFIG['batch_size'], CONFIG['seq_len'], on_step)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted. Saving brain...")
    
    # 4. Finalize
    model.save("circuit_brain_mvp_sota.bin")
    print("✅ MVP Brain Saved.")

if __name__ == "__main__":
    train_circuit_mvp_sovereign()
