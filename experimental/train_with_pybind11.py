"""
Experimental PyBind11 Native C++ Training Script
Target: Train NSOS JambaModel natively using compiled C++ pybind11 bindings (nsos_ext).

WARNING: This script is entirely self-contained inside experimental/ and does NOT touch or modify any existing project files.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add compiled C++ binaries directory to Python search path
release_build_path = str(Path(__file__).parent.parent / "OXN" / "build" / "Release")
if release_build_path not in sys.path:
    sys.path.insert(0, release_build_path)

import nsos_ext

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def run_pybind11_native_training():
    print("=" * 80)
    print("🚀 NSOS NATIVE C++ PYBIND11 TRAINING TRIAL")
    print(f"📦 Loaded C++ Module: {nsos_ext.__file__}")
    print("=" * 80)

    # 1. Instantiate C++ JambaModel with 4 layers, d_model=128, vocab=256
    print("\n[1/3] Instantiating C++ JambaModel(num_layers=4, d_model=128, vocab_size=256)...")
    model = nsos_ext.JambaModel(4, 128, 256)
    model.set_training_mode(True)

    # 2. Instantiate C++ Trainer with learning rate 0.003
    print("[2/3] Instantiating C++ Trainer (lr=0.003)...")
    trainer = nsos_ext.Trainer(model, 0.003)

    # 3. Create synthetic sequence of tokens for training loop
    print("[3/3] Executing C++ Native Training Loop via PyBind11...\n")
    tokens = [i % 256 for i in range(1024)]
    
    print(f"{'Step':<6} | {'C++ Loss':<12} | {'Step Time (ms)':<15} | {'Global Step':<12}")
    print("-" * 55)

    start_time = time.time()
    for step in range(1, 21):
        step_start = time.time()
        
        # Call native C++ train_step on tokens
        prompt = tokens[step : step + 16]
        target = tokens[step + 1 : step + 17]
        loss = trainer.train_step(prompt, target)
        
        step_ms = (time.time() - step_start) * 1000.0
        print(f"{step:<6d} | {loss:<12.4f} | {step_ms:<15.2f} | {trainer.global_step_count:<12d}")

    total_time = time.time() - start_time
    print("-" * 55)
    print(f"✅ Native PyBind11 Training Completed in {total_time:.3f}s!")
    print(f"📊 Final C++ Loss: {loss:.4f} | Total C++ Global Steps: {trainer.global_step_count}")
    print("=" * 80)


if __name__ == "__main__":
    run_pybind11_native_training()
