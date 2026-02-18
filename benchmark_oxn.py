#!/usr/bin/env python3
"""
==============================================================================
NSOS/OXN BENCHMARK & VERIFICATION
==============================================================================
Carrega o modelo treinado e executa testes de performance e integridade.
"""

import sys
import os
import time
import random

# Setup
build_dir = os.path.join(os.path.dirname(__file__), "OXN", "build", "Release")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import nsos_ext as nsos

MODEL_FILE = "oxn_trained_model.bin"
CONFIG = {"layers": 8, "dim": 128, "vocab": 256}

print("=" * 80)
print("📊 NSOS/OXN BENCHMARK SUITE")
print("=" * 80)

# 1. Carregar Modelo
print(f"\n1. Carregando modelo: {MODEL_FILE}...")
if not os.path.exists(MODEL_FILE):
    print("❌ Erro: Arquivo do modelo não encontrado!")
    sys.exit(1)

try:
    model = nsos.JambaModel(CONFIG['layers'], CONFIG['dim'], CONFIG['vocab'], nsos.Device.CPU)
    model.load(MODEL_FILE)
    print(f"   ✅ Modelo carregado ({os.path.getsize(MODEL_FILE)/1024/1024:.2f} MB)")
except Exception as e:
    print(f"   ❌ Erro ao carregar: {e}")
    sys.exit(1)

# 2. Benchmark de Inferência
print("\n2. Executando Benchmark de Performance...")

def run_benchmark(model, num_tokens=500):
    ctx = nsos.Context()
    tokens = [random.randint(1, 200) for _ in range(num_tokens)]
    
    start = time.time()
    # Batch process para evitar overhead excessivo do loop Python
    chunk_size = 50
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size]
        model.forward_ids(chunk, ctx)
    
    dt = time.time() - start
    return num_tokens / dt, (dt/num_tokens)*1000

try:
    # Warmup
    print("   🔥 Aquecendo (Warmup)...")
    run_benchmark(model, 50)
    
    # Teste Real
    print("   ⏱️ Medindo (1000 tokens)...")
    tps, lat = run_benchmark(model, 1000)
    
    print(f"""
   ╔══════════════════════════════════════════════════════════╗
   ║  RESULTADOS DE PERFORMANCE                               ║
   ╠══════════════════════════════════════════════════════════╣
   ║  Throughput:    {tps:>10.2f} tokens/sec                   ║
   ║  Latency:       {lat:>10.2f} ms/token                     ║
   ║  Est. Memory:   {os.path.getsize(MODEL_FILE)/1024/1024:>10.2f} MB (Model Size)           ║
   ╚══════════════════════════════════════════════════════════╝
    """)
    
except Exception as e:
    print(f"   ⚠️ Erro no benchmark: {e}")

# 3. Teste Funcional (Reasoning)
print("\n3. Teste Funcional (System 2 Reasoning)...")
try:
    test_input = [1, 2, 3, 4]
    ctx = nsos.Context()
    
    print(f"   Input: {test_input}")
    output = model.forward_ids(test_input, ctx)
    print(f"   Output Norm: {output.norm():.4f}")
    
    # Executa Reasoning
    print("   🧠 Running Reasoning Loop...")
    thought = model.run_reasoning_loop(output, 2)
    print(f"   Reasoning ok. Thought shape: {list(thought.shape)}")
    print("   ✅ Teste Funcional Passou!")

except Exception as e:
    print(f"   ❌ Erro no teste funcional: {e}")

print("\n" + "=" * 80)
print("🎉 VERIFICAÇÃO COMPLETA")
print("=" * 80)
