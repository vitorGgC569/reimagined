"""
Experimental Model Interaction & Text Generation Script
Target: Load saved NSOS model checkpoint (real_nsos_jamba_model.bin) and generate text outputs.

WARNING: This script is entirely self-contained inside experimental/ and does NOT touch or modify any existing project files.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np

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


REAL_TAX_CORPUS = [
    "COMPRA DE COMBUSTIVEL POSTO SHELL NOTA FISCAL DEBITO COMBUSTIVEIS CREDITO CAIXA REGISTRO SPED 1500",
    "VENDA DE MERCADORIAS NOTA FISCAL ELETRONICA DEBITO CAIXA CREDITO RECEITA DE VENDAS REGISTRO SPED",
    "PAGAMENTO DE SALARIOS DE FUNCIONARIOS DEBITO DESPESA COM SALARIOS CREDITO BANCO CONTA MOVIMENTO",
    "COMPRA DE ESTOQUE SUPERMERCADO CARREFOUR DEBITO ESTOQUE DE MERCADORIAS CREDITO BANCO SPED",
    "RECEBIMENTO DE DUPLICATAS CLIENTES DEBITO BANCO CONTA MOVIMENTO CREDITO CONTAS A RECEBER SPED",
    "PAGAMENTO DE ALUGUEL DE ITAMOVEL DEBITO DESPESA DE ALUGUEL CREDITO BANCO CONTA MOVIMENTO SPED",
    "RECOLHIMENTO DE IMPOSTOS ICMS DEBITO ICMS A RECOLHER CREDITO BANCO CONTA MOVIMENTO REGISTRO SPED",
    "PRESTACAO DE SERVICOS CONTABEIS DEBITO DESPESA DE HONORARIOS CREDITO BANCO CONTA MOVIMENTO SPED"
]


class OContabilTokenizer:
    def __init__(self, corpus):
        vocab = set()
        for text in corpus:
            vocab.update(text.upper().split())
        self.word2id = {w: i + 2 for i, w in enumerate(sorted(vocab))}
        self.word2id["<PAD>"] = 0
        self.word2id["<EOS>"] = 1
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.vocab_size = len(self.word2id) + 10

    def encode(self, text):
        return [self.word2id.get(w, 0) for w in text.upper().split() if w in self.word2id]

    def decode(self, token_ids):
        words = [self.id2word.get(t_id, f"[{t_id}]") for t_id in token_ids if t_id not in (0, 1)]
        return " ".join(words)


def run_model_chat():
    print("=" * 80)
    print("🗣️ INTERACTIVE GENERATION WITH TRAINED NSOS NATIVE C++ MODEL")
    print("=" * 80)

    tokenizer = OContabilTokenizer(REAL_TAX_CORPUS)
    checkpoint_path = Path("experimental/real_nsos_jamba_model.bin")

    # 1. Load Trained C++ Model Weights
    print(f"\n[1/2] Loading C++ Model Checkpoint: {checkpoint_path}...")
    model = nsos_ext.JambaModel(num_layers=4, d_model=128, vocab_size=tokenizer.vocab_size)
    
    if checkpoint_path.exists():
        model.load(str(checkpoint_path))
        print("✅ Trained C++ Model Weights Loaded Successfully!")
    else:
        print("⚠️ Checkpoint file not found.")

    model.set_training_mode(False)

    # 2. Interactive Test Prompts
    test_prompts = [
        "COMPRA DE COMBUSTIVEL POSTO SHELL NOTA FISCAL",
        "VENDA DE MERCADORIAS NOTA FISCAL ELETRONICA DEBITO",
        "PAGAMENTO DE SALARIOS DE FUNCIONARIOS",
        "RECOLHIMENTO DE IMPOSTOS ICMS DEBITO"
    ]

    print("\n[2/2] Generating Accounting Completions from Trained C++ Model:\n")

    for prompt_text in test_prompts:
        prompt_ids = tokenizer.encode(prompt_text)
        print(f"📥 PROMPT DE ENTRADA: \"{prompt_text}\"")

        generated_ids = list(prompt_ids)
        for _ in range(8):
            # C++ Forward pass returns nsos_ext.Tensor
            logits_tensor = model.forward_ids(generated_ids)
            logits_arr = logits_tensor.numpy()
            
            # Take logits for the last token position
            if len(logits_arr.shape) > 1:
                next_logits = logits_arr[-1]
            else:
                next_logits = logits_arr
                
            next_token = int(np.argmax(next_logits))
            generated_ids.append(next_token)
            if next_token == 1:
                break

        full_output = tokenizer.decode(generated_ids)
        completion_only = tokenizer.decode(generated_ids[len(prompt_ids):])

        print(f"📤 PREDIÇÃO DO MODELO: \"{completion_only}\"")
        print(f"📜 TEXTO FINAL COMPLETO: \"{full_output}\"")
        print("-" * 75)

    print("=" * 80)


if __name__ == "__main__":
    run_model_chat()
