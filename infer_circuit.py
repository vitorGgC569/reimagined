import os
import sys
import numpy as np

# Setup
BUILD_DIR = r"C:\Users\VitorGGc\Desktop\Pantheon-Oxtav1-15338770387506525964\OXN\build\Release"
sys.path.insert(0, BUILD_DIR)
import nsos_ext as nsos

def infer():
    print("="*80)
    print("🔮 CIRCUIT-OXN INFERENCE ENGINE")
    print("="*80)
    
    # Configuration MUST match training
    CONFIG = {"layers": 6, "dim": 128, "vocab": 256}
    model_file = "circuit_brain_mvp_sota.bin"
    
    if not os.path.exists(model_file):
        # Fallback to checkpoint if main file not saved yet
        model_file = "circuit_brain_checkpoint.bin"
        if not os.path.exists(model_file):
            print("❌ No brain file found! Wait for training to save at least one checkpoint.")
            return

    # 1. Load Model
    print(f"🧠 Loading brain from {model_file}...")
    model = nsos.JambaModel(CONFIG['layers'], CONFIG['dim'], CONFIG['vocab'], nsos.Device.CPU)
    model.load(model_file)
    
    # 2. Prompts for the MVP
    prompts = [
        "REQ:voltage_divider_10_20|NET:",
        "REQ:rc_filter_5k_10n|NET:",
        "REQ:led_driver_9v|NET:"
    ]
    
    def generate(prompt_text, max_tokens=150):
        # Encode (Byte-level)
        ids = [ord(c) for c in prompt_text]
        generated = list(ids)
        
        ctx = nsos.Context()
        
        for _ in range(max_tokens):
            # Forward pass
            # Note: Using the last N tokens if needed, but Jamba/Mamba handles long sequences
            logits = model.forward_ids(generated, ctx)
            
            # Simple Greedy Search (SOTA for precision)
            # Logits are [1, Seq, Vocab]
            logits_np = logits.cpu().numpy()
            next_id = int(np.argmax(logits_np[0, -1, :]))
            
            if next_id == ord('
') or next_id == 0:
                break
                
            generated.append(next_id)
            
        return "".join([chr(i) for i in generated])

    print("
🚀 Testing Knowledge:")
    for p in prompts:
        print(f"
Prompt: {p}")
        result = generate(p)
        print(f"Result: {result}")

if __name__ == "__main__":
    infer()
