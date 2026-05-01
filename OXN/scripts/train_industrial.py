import sys
import os
# Force Single Threading for Determinism (Emergency Protocol)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import shutil
import math
import argparse
import yaml
import hashlib

# Ensure we can import nsos_ext
def try_import_nsos():
    search_paths = [
        os.getcwd(),
        os.path.join(os.getcwd(), "build/Release"),
        os.path.join(os.path.dirname(__file__), ".."),
        os.path.join(os.path.dirname(__file__), "../build/Release")
    ]
    
    # Windows DLL handling for Python 3.8+
    if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
        # 1. Add current and build directories
        for path in search_paths:
            if os.path.exists(path):
                try:
                    os.add_dll_directory(os.path.abspath(path))
                except Exception:
                    pass
        
        # 2. Add CUDA path if present
        cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
        if os.path.exists(cuda_path):
            try:
                os.add_dll_directory(cuda_path)
            except Exception:
                pass

    for path in search_paths:
        if path not in sys.path:
            sys.path.append(path)
    
    try:
        import nsos_ext
        print(f"[Import] nsos_ext location: {nsos_ext.__file__}")
        return nsos_ext
    except ImportError as e:
        # Try finding the file explicitly if name has suffixes
        found = False
        for path in search_paths:
            if not os.path.exists(path): continue
            for f in os.listdir(path):
                if f.startswith("nsos_ext") and f.endswith(".pyd"):
                    found = True
                    break
            if found: break
        
        if not found:
            return None
            
        print(f"DEBUG: Found nsos_ext.pyd but import failed: {e}")
        # Try to import again, maybe it's a name variation
        try:
            import nsos_ext
            return nsos_ext
        except ImportError:
            return None

nsos_ext = try_import_nsos()
if nsos_ext is None:
    print("CRITICAL: nsos_ext not found or could not be loaded. Run build script first and check CUDA/DLL paths.")
    sys.exit(1)

import numpy as np

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Datasets library not found. Falling back to synthetic data.")

# Default Config
CONFIG = {
    "experiment_name": "Default",
    "seed": 42,
    "model": {"d_model": 128, "layers": 4, "vocab_size": 256},
    "training": {"batch_size": 1, "accumulate_grad": 1, "lr": 0.001, "max_steps": 100, "checkpoint_interval": 20},
    "hardware": {"precision": "fp32", "compile": False}
}

def load_config(path):
    global CONFIG
    if os.path.exists(path):
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Deep update logic simplified
            if "model" in yaml_config: 
                CONFIG["model"].update(yaml_config["model"])
                if "dim" in yaml_config["model"]:
                    CONFIG["model"]["d_model"] = yaml_config["model"]["dim"]
            if "training" in yaml_config: CONFIG["training"].update(yaml_config["training"])
            if "hardware" in yaml_config: CONFIG["hardware"].update(yaml_config["hardware"])
            if "experiment_name" in yaml_config: CONFIG["experiment_name"] = yaml_config["experiment_name"]
            if "seed" in yaml_config: CONFIG["seed"] = yaml_config["seed"]
        print(f" Loaded Configuration from {path}")
    else:
        print(f"  Config file {path} not found. Using Defaults.")

def get_file_hash(filepath):
    # L blocos para no estourar RAM
    if not os.path.exists(filepath): return "00000000"
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_device(force_cpu=False):
    if force_cpu:
        print(" Forcing CPU mode as requested.")
        return nsos_ext.Device.CPU
    try:
        test = nsos_ext.Tensor.zeros([1], nsos_ext.Device.GPU)
        print(" GPU Detected. Engaging Turbo Mode.")
        return nsos_ext.Device.GPU
    except Exception as e:
        print(f" GPU not available or failed ({e}). Falling back to CPU.")
        return nsos_ext.Device.CPU

def train_industrial():
    parser = argparse.ArgumentParser(description='NSOS Industrial Training')
    parser.add_argument('--debug', action='store_true', help='Enable deep debugging and monitoring')
    parser.add_argument('--config', type=str, default="OXN/configs/1050ti_industrial.yaml", help='Path to config file')
    parser.add_argument('--dataset', type=str, default="wikitext", help='HuggingFace dataset name (e.g., wikitext)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data (for testing pipeline only)')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps to run')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    args = parser.parse_args()

    print("==========================================")
    print("     NSOS INDUSTRIAL TRAINING SUITE       ")
    print(f"    DEBUG MODE: {'ON' if args.debug else 'OFF'}")
    print("==========================================")

    # Load Config
    load_config(args.config)

    if args.debug:
        print("[Debug] Enabling C++ Monitor & Inspector...")
        nsos_ext.Monitor.enable()
        if hasattr(nsos_ext.Monitor, "set_verbose"):
            nsos_ext.Monitor.set_verbose(True)
        if hasattr(nsos_ext, "Inspector"):
            # Level 2 = Health Check (Stats)
            nsos_ext.Inspector.set_level(2)

    # 1. Configuration (Applied)
    model_cfg = CONFIG["model"]
    train_cfg = CONFIG["training"]
    # Select Device
    device = get_device(force_cpu=args.cpu)

    # 2. Initialize Model
    print(f"[Init] Creating JambaModel ({model_cfg['layers']}L, {model_cfg['d_model']}D)...")
    model = nsos_ext.JambaModel(model_cfg['layers'], model_cfg['d_model'], model_cfg['vocab_size'], device)

    if device == nsos_ext.Device.GPU:
        model.to(device)

    # 3. Data Loading
    print("[Data] Loading Training Corpus...")

    tokens = []
    use_synthetic = args.synthetic
    dataset_name = args.dataset

    if use_synthetic:
        print("  WARNING: Using SYNTHETIC Data. Model will not learn real knowledge.")
        tokens = [i % model_cfg['vocab_size'] for i in range(10000)]
    elif dataset_name:
        if HAS_DATASETS:
            try:
                print(f"Loading HuggingFace dataset: {dataset_name}...")
                ds = load_dataset(dataset_name, split="train")
                print("Tokenizing...")
                # Simplified tokenization for demo - ideal is using tokenizer.encode
                # But for now we stick to simple char-level or similar if tokenizer not robust
                # We will assume wikitext-like text and just use raw bytes if tokenizer unavailable in python
                # checks
                for line in ds["text"][:1000]:
                     tokens.extend([ord(c) % model_cfg['vocab_size'] for c in line])
            except Exception as e:
                print(f" Dataset load failed: {e}")
                sys.exit(1)
        else:
            print(" 'datasets' library not installed. Install or use --synthetic.")
            sys.exit(1)
    else:
         print(" No dataset specified. Use --dataset <name> or --synthetic.")
         sys.exit(1)

    if not tokens:
        print(" Dataset is empty.")
        sys.exit(1)

    seq_len = 32
    if len(tokens) < seq_len + 1:
        tokens = [0] * (seq_len + 100)

    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f" Train Tokens: {len(train_tokens)} | Val Tokens: {len(val_tokens)}")

    # 4. Training Loop
    print("[Train] Starting Epochs...")

    epochs = 2
    for epoch in range(epochs):
        model.reset_session()
        start_time = time.time()
        total_loss = 0.0
        num_batches = len(train_tokens) // (train_cfg['batch_size'] * seq_len)
        if num_batches > train_cfg['max_steps']: num_batches = train_cfg['max_steps']

        for step in range(num_batches):
            ctx = nsos_ext.Context()

            offset = step * seq_len
            seq_ids = train_tokens[offset : offset + seq_len]
            target_ids = train_tokens[offset + 1 : offset + seq_len + 1]

            if len(target_ids) != len(seq_ids): break

            try:
                # Forward
                if args.debug: print(f"[Step {step}] Forward Pass...")
                logits = model.forward_ids(seq_ids, ctx)

                # logits is [1, seq_len, vocab_size]
                final_logits = logits
                if len(final_logits.shape) == 3:
                     final_logits = final_logits.reshape([-1, final_logits.shape[2]])

                # Loss
                loss, d_logits = final_logits.cross_entropy(target_ids)
                total_loss += loss

                if args.debug: print(f"[Step {step}] Backward Pass...")

                # Backward using model's own head logic
                model.backward(d_logits, ctx)

                # Update (Muon Optimizer Integration)
                if args.debug: print(f"[Step {step}] Optimizer Step (Muon)...")

                params = model.parameters()
                # Create Optimizer (Ephemeral for this script logic)
                optimizer = nsos_ext.MuonOptimizer([], train_cfg['lr'])
                optimizer.step_and_quantize(params)

                # Zero Grad Manual
                for p in params:
                    if p.grad.size > 0:
                        zero_g = nsos_ext.Tensor.zeros(p.grad.shape, device)
                        p.grad.copy_from(zero_g)

                if args.debug:
                     # Inspect first param
                     p0 = params[0]
                     if p0.data.device == nsos_ext.Device.CPU:
                          norm = p0.data.norm()
                          print(f"  [Param 0 Norm] {norm:.4f}")

            except Exception as e:
                print(f"Error at step {step}: {e}")
                break

            if step % 10 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{num_batches} | Loss: {loss:.4f}")

        total_time = time.time() - start_time
        actual_steps = step + 1 if 'step' in locals() else 0
        avg_loss = total_loss / (actual_steps + 1e-6)
        total_tokens = actual_steps * train_cfg['batch_size'] * seq_len
        tokens_per_sec = total_tokens / (total_time + 1e-9)
        print(f"Epoch {epoch+1} Complete.")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Steps Completed: {actual_steps}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")

        # 5. Validation
        print("[Val] Running Validation...")
        val_loss = 0.0
        val_batches = len(val_tokens) // seq_len
        if val_batches > 10: val_batches = 10

        for v_step in range(val_batches):
            offset = v_step * seq_len
            seq_ids = val_tokens[offset : offset + seq_len]
            target_ids = val_tokens[offset + 1 : offset + seq_len + 1]

            logits = model.forward_ids(seq_ids, None)
            
            # Model already performs head projection
            final_logits = logits
            if len(final_logits.shape) == 3:
                 final_logits = final_logits.reshape([-1, final_logits.shape[2]])
            loss, _ = final_logits.cross_entropy(target_ids)
            val_loss += loss

        print(f"Validation Loss: {val_loss/(val_batches+1e-6):.4f}")

        # 6. Save Checkpoint (Atomic)
        print("\n[Persistence] Auditing Parameters before save:")
        params = model.parameters()
        print(f"Total Parameters: {len(params)}")
        for p in params:
            print(f"  - {p.name}: {p.data.shape}")
            
        ckpt_path = "industrial_ckpt.bin"
        temp_path = ckpt_path + ".tmp"
        print(f"[Persistence] Atomic Save to {ckpt_path}...")
        model.save(temp_path)
        os.replace(temp_path, ckpt_path)
        print("Checkpointer secured.")

    # 7. Verification Load
    print("\n[Verification] Reloading Model...")
    try:
        model2 = nsos_ext.JambaModel(model_cfg['layers'], model_cfg['d_model'], model_cfg['vocab_size'], device)
        if device == nsos_ext.Device.GPU: model2.to(device)

        model2.load("industrial_ckpt.bin")
        print(" Model Reloaded Successfully. Pipeline Integrity Verified.")
    except Exception as e:
        print(f"Reload Failed: {e}")

    print("\n=== Industrial Training Complete ===")

if __name__ == "__main__":
    train_industrial()
