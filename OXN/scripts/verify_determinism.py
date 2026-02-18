#!/usr/bin/env python3
# OXN/scripts/verify_determinism.py - Verificação de consistência de seeds

import hashlib
import numpy as np
import sys
import os

# Adicionar caminho do projeto para importar nsos_ext se necessário
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import nsos_ext
except ImportError:
    print("❌ Erro: nsos_ext não encontrado. Certifique-se de que o projeto foi compilado.")
    sys.exit(1)

def _hash_result(result):
    if hasattr(result, 'numpy'):
        arr = result.numpy()
        return hashlib.sha256(arr.tobytes()).hexdigest()
    return hashlib.sha256(str(result).encode()).hexdigest()

def verify_determinism():
    print("--- Verificando Determinismo do NSOS ---")
    
    seeds = [42, 123, 999]
    all_passed = True
    
    for seed in seeds:
        print(f"Testando seed: {seed}...", end=" ")
        
        # Primeira execução
        nsos_ext.set_seed(seed)
        # Mock de operação (ex: criar tensor aleatório)
        t1 = nsos_ext.Tensor([10, 10], nsos_ext.Device.CPU)
        t1 = nsos_ext.Tensor.random([10, 10]) 
        h1 = _hash_result(t1)
        
        # Segunda execução com mesma seed
        nsos_ext.set_seed(seed)
        t2 = nsos_ext.Tensor.random([10, 10])
        h2 = _hash_result(t2)
        
        if h1 == h2:
            print("✅ OK")
        else:
            print("❌ FALHA (Inconsistente)")
            all_passed = False
            
    if all_passed:
        print("\n🎉 Todos os testes de determinismo passaram!")
    else:
        print("\n🚨 Problemas de determinismo detectados!")
        sys.exit(1)

if __name__ == "__main__":
    verify_determinism()
