"""
Experimental Real PyBind11 Training Script
Target: Train using native C++ pybind11 module (nsos_ext.JambaModel + nsos_ext.Trainer).

WARNING: This script is entirely self-contained inside experimental/ and does NOT touch or modify any existing project files.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add compiled C++ binaries directory to Python search path
release_path = str(Path(__file__).parent.parent / "OXN" / "build" / "Release")
if release_path not in sys.path:
    sys.path.insert(0, release_path)

import nsos_ext

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def execute_real_cpp_training():
    print("=" * 80)
    print("🚀 EXECUÇÃO DE TREINO REAL USANDO BINDINGS NATIVOS C++ PYBIND11 (nsos_ext)")
    print("=" * 80)

    # 1. Instancia o modelo Jamba C++ nativo
    print("\n[1] Instanciando JambaModel C++ nativo (layers=4, d_model=128, vocab=256)...")
    model = nsos_ext.JambaModel(4, 128, 256)
    model.set_training_mode(True)

    # 2. Instancia o Trainer C++ nativo com otimizador 4-bit
    print("[2] Instanciando Trainer C++ com AdamW em 4-bit (optimizer_state_bits=4)...")
    trainer = nsos_ext.Trainer(model, 0.003)
    trainer.optimizer_state_bits = 4
    trainer.warmup_steps = 20

    # 3. Executa o loop de treinamento supervisionado C++
    print("[3] Executando passos de treinamento supervisionado em C++...\n")
    
    prompt_tokens = [10, 15, 20, 25, 30]
    answer_tokens = [50, 55, 60, 65, 70]

    print(f"{'Passo':<8} | {'Loss C++ Nativo':<18} | {'Passos Globais C++':<20} | {'Tempo (ms)':<12}")
    print("-" * 65)

    start_time = time.time()
    for step in range(1, 21):
        t0 = time.time()
        loss = trainer.train_supervised(prompt_tokens, answer_tokens)
        dt = (time.time() - t0) * 1000.0
        
        print(f"{step:<8d} | {loss:<18.4f} | {trainer.global_step_count:<20d} | {dt:<12.2f}")

    total_dt = time.time() - start_time
    print("-" * 65)
    print(f"✅ Treino C++ via PyBind11 finalizado em {total_dt:.3f}s!")
    print(f"💾 Salvando Checkpoint do Modelo C++...")
    model.save("experimental/meu_modelo_treinado.bin")
    print("=" * 80)


if __name__ == "__main__":
    execute_real_cpp_training()
