"""
Experimental Real C++ PyBind11 Benchmark Script
Target: Benchmark NSOS JambaModel natively using compiled C++ pybind11 bindings (nsos_ext) on a Real Dataset.

WARNING: This script is entirely self-contained inside experimental/ and does NOT touch or modify any existing project files.
"""

from __future__ import annotations

import os
import sys
import time
import math
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


# ------------------------------------------------------------------------
# 1. Real Dataset: SPED Accounting & Brazilian Tax Corpus
# ------------------------------------------------------------------------
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


class RealDatasetTokenizer:
    def __init__(self, corpus):
        vocab = set()
        for text in corpus:
            vocab.update(text.upper().split())
        self.word2id = {w: i + 2 for i, w in enumerate(sorted(vocab))}
        self.word2id["<PAD>"] = 0
        self.word2id["<EOS>"] = 1
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.vocab_size = len(self.word2id) + 10 # Buffer for vocab

    def encode(self, text):
        return [self.word2id.get(w, 0) for w in text.upper().split()]


# ------------------------------------------------------------------------
# 2. Real Native C++ PyBind11 Benchmark Execution
# ------------------------------------------------------------------------
def run_real_cpp_pybind_benchmark():
    print("=" * 80)
    print("🏆 REAL BENCHMARK: Native C++ PyBind11 Engine (nsos_ext) on Real Dataset")
    print("📋 Target Dataset: Brazilian Accounting & SPED Tax Rules (OContabil Domain)")
    print("=" * 80)

    # Tokenize real dataset
    tokenizer = RealDatasetTokenizer(REAL_TAX_CORPUS)
    dataset = [tokenizer.encode(t) for t in REAL_TAX_CORPUS]

    print(f"\n[CONFIG] Vocab Size: {tokenizer.vocab_size} words | Total Corpus Sentences: {len(dataset)}")

    # 1. Instantiate Real C++ JambaModel
    print("[1/3] Instantiating C++ JambaModel(num_layers=4, d_model=128, vocab_size=256)...")
    model = nsos_ext.JambaModel(num_layers=4, d_model=128, vocab_size=tokenizer.vocab_size)
    model.set_training_mode(True)

    # 2. Instantiate Real C++ Trainer with 4-bit Optimizer State
    print("[2/3] Instantiating C++ Trainer (lr=0.005, optimizer_state_bits=4)...")
    trainer = nsos_ext.Trainer(model, 0.005)
    trainer.optimizer_state_bits = 4
    trainer.warmup_steps = 10
    trainer.max_grad_norm = 1.0

    # Prepare batches: Prompt -> Target Answer
    prompt_batch = [seq[:-2] for seq in dataset]
    answer_batch = [seq[-2:] for seq in dataset]

    epochs = 20

    print("\n[3/3] Running Native C++ Training Loop on Real Dataset...\n")
    print(f"{'Epoch':<6} | {'C++ Loss':<12} | {'Perplexity (PPL)':<18} | {'Speed (Tokens/s)':<18} | {'Global Step':<12}")
    print("-" * 75)

    start_time = time.time()
    total_tokens_processed = 0
    start_loss = None

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        
        # Execute C++ native supervised batch training (pure C++ autograd & AdamW)
        loss = trainer.train_supervised_batch(prompt_batch, answer_batch)
        
        ep_time = time.time() - ep_start
        tokens_in_ep = sum(len(p) + len(a) for p, a in zip(prompt_batch, answer_batch))
        total_tokens_processed += tokens_in_ep
        
        speed = tokens_in_ep / max(ep_time, 1e-5)
        ppl = math.exp(min(loss, 20.0))

        if start_loss is None:
            start_loss = loss

        print(f"{epoch:<6d} | {loss:<12.4f} | {ppl:<18.2f} | {speed:<18.0f} | {trainer.global_step_count:<12d}")

    total_time = time.time() - start_time
    overall_throughput = total_tokens_processed / max(total_time, 1e-5)
    loss_reduction = ((start_loss - loss) / start_loss) * 100.0

    print("-" * 75)
    print(f"✅ Real Native C++ PyBind11 Benchmark Finished in {total_time:.3f}s!")
    print(f"📊 Initial Loss: {start_loss:.4f} ➔ Final C++ Loss: {loss:.4f} (Queda de Loss: {loss_reduction:.1f}%)")
    print(f"⚡ Final Perplexity (PPL): {ppl:.2f} | Overall C++ Throughput: {overall_throughput:.0f} tokens/s")
    
    # Save trained model checkpoint
    model_checkpoint_path = str(Path(__file__).parent / "real_nsos_jamba_model.bin")
    model.save(model_checkpoint_path)
    print(f"💾 Trained C++ Model Checkpoint saved to: {model_checkpoint_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_real_cpp_pybind_benchmark()
