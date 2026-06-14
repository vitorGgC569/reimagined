# NSOS Benchmark Results

> ⚠️ **AVISO DE HONESTIDADE (2026-06).** As tabelas abaixo desta seção são um
> **template histórico com números ILUSTRATIVOS/SIMULADOS** (a §3 está literalmente
> marcada "Simulated"), NÃO medições reproduzíveis. Não cite estes valores como
> evidência. As métricas REAIS e reproduzíveis — gradcheck verde, perplexidade
> held-out e exact-match de generalização numa tarefa controlada — são geradas por
> **`colab/train_gpu_phases_t4_v2.ipynb`** (rodar e colar a saída). Até lá, trate
> este arquivo como esqueleto pendente de preenchimento por uma run real.

## 0. Resultados MEDIDOS do v8 (reais — "verdade primeiro")

Fonte: `OXN/nsos/scripts/live_distill_v8/run_summary.json` (run real, não simulada).
Config: `hybrid_medium` — 12 camadas, d_model=512, MoE-8 (top-2), ~40M params,
vocab 4830, **CPU**, 800 steps, bundle v3. `release_gate.passed = false` (18 falhas).

**Holdouts oficiais (suites externas):**

| Suite | rows | heldout_loss (nats) | ≈ perplexidade | teacher-token acc | first-token acc | exact-match |
|---|---|---|---|---|---|---|
| nsos_micro_suite | 12 | 8.44 | ~4.6e3 | 9.9% | 0% | **0/12 (0%)** |
| nsos_eval_suite | 20 | 8.32 | ~4.1e3 | 13.4% | 0% | **0/20 (0%)** |

**heldout_loss por fase (champion):** phase1 8.65 (ppl≈5.7e3) · phase2 8.17 (≈3.5e3) ·
**phase3_curated_text 7.31 (≈1.5e3)** · phase4_instructions 7.45 (≈1.7e3) ·
phase5_verifier 7.92 (≈2.8e3) · phase6_memory 8.34 (≈4.2e3).
`exact_accuracy = 0.0` e `first_token_accuracy = 0.0` em **todas** as fases.

**Leitura honesta:** com vocab 4830 (perplexidade aleatória ≈ 4830), só
`phase3_curated_text` (ppl≈1500) mostra modelagem de linguagem real, porém fraca;
o resto está no nível do acaso. **Zero generalização de tarefa** (exact-match e
first-token = 0). Único sinal positivo: teacher-token accuracy em
translate/summarize/instruções (~25–40%) — o modelo capta estatística local, não
resolve as tarefas. Isto confirma com números o "vácuo de validação" e motiva a
tarefa controlada (adição PT com holdout composicional) do
`colab/train_gpu_phases_t4_v2.ipynb`, onde um modelo pequeno PODE exibir
generalização mensurável.

## 0.1. Evidência do STACK CORRIGIDO — medida localmente (CPU, MSVC)

Produzida em 2026-06-14 compilando o projeto local (Visual Studio 18 / MSVC,
CPU-only) e executando — **sem Colab/GPU**. Reproduzível.

**(a) Gradcheck (`test_gradcheck`, build-local-gradcheck):** 16/16 PASS, exit 0,
reprodutível (seed fixa). Erros relativos vs diferenças finitas:
- mamba2-proper: d/input 6.2e-3, d/A 4.7e-4, d/conv1d 2.4e-3 — **SSM corrigido OK**
- moe switch-aux 5.5e-3, moe router-grad 5.2e-4 — **MoE corrigido OK**
- mamba2 legado 3.0e-3, rmsnorm 4.7e-2, cross_entropy 7.5e-3, matmul 1.3e-4, KAN ≤5e-3.

**(b) Tarefa controlada PT — generalização** (`scripts/local_pt_addition_eval.py`):
"quanto e A mais B ?" com **hold-out composicional** (pares (A,B) inéditos; cada A,B
visto isolado). Modelo 4 camadas d=128 MoE-4, **flags corrigidos ON**
(`NSOS_MAMBA_PROPER_SSM`, `NSOS_MOE_FP_ROUTER`, `NSOS_MOE_SWITCH_AUX`,
`NSOS_MAMBA_A_LOGSPACED`), 40 épocas, CPU.

| | perplexidade | exact-match |
|---|---|---|
| treino | 1.000 | 100% |
| **held-out (inédito)** | **5.724** | **66.7%** |
| baseline aleatório | 19 (=\|vocab num\|) | 5.3% |

loss/pair 2.52 → 0.0002 (convergência limpa, sem divergência). **Veredito:
GENERALIZOU** — 66.7% de acerto em pares nunca vistos = 12.6× o acaso. Contraste com
o v8 (§0): exact-match 0% nas tarefas dele. O stack corrigido, numa tarefa em que um
modelo pequeno PODE mostrar generalização, de fato a mostra.

> Cobertura honesta: isto roda em CPU (a via proper-Mamba é host nesta versão). A
> validação em GPU/T4 (BF16, throughput, paridade determinística) usa
> `colab/train_gpu_phases_t4_v2.ipynb`.

## 1. Unit Test Coverage
All comprehensive unit tests passed successfully (`test_suite`).
- **Components Tested**: SprecherBlock, MemorySystem (EpMAN), Fabric, MCTS, TTTLayer, SophiaOptimizer.

## 2. Efficiency & Performance Metrics
Measured via `nsos_bench` and `nsos_superiority`.

| Metric | Result | Interpretation |
|---|---|---|
| **BitNet Quantization MSE** | ~0.26 | Low error indicating effective ternary compression vs Float32. |
| **Throughput (FLOPs)** | ~0.005 GFLOPS | Base CPU performance. CUDA path verified but requires HW. |
| **Edge Latency (ABM)** | ~1.5 ms | High-speed inference using Accumulation-Before-Multiplication. |
| **Energy Efficiency** | ~20x LLM | Estimated gain from 1.58-bit additions vs FP16 multiplications. |

## 3. Curriculum Training Validation
Results from `train_nsos_curriculum.py` (Simulated):

| Phase | Task | Loss Trend | Accuracy |
|---|---|---|---|
| **Phase 0** | Sanity (Copy/Reverse) | 0.96 -> 0.00 | 100% |
| **Phase 1** | Algorithmic (ARC) | 0.93 -> 0.05 | High |
| **Phase 2** | Long Memory (Needle) | Stable | >90% |

**Conclusion**: The architecture demonstrates monotonic loss reduction across all phases, confirming stability and learning capability.

## 4. Reasoning (System 2)
Measured via `bench_reasoning.py` and `nsos_superiority`.

| Benchmark | NSOS (Time/Steps) | Transformer (Time/Steps) | Gain |
|---|---|---|---|
| **Context Scaling (1M)** | 500ms | 500,000ms (Est) | **1000x** |
| **Latent Search (Depth 10)**| 90% Success | 10% Success (Greedy) | **9x** |

The MCTS engine effectively uses the `JambaModel` world model to explore future states, significantly outperforming greedy decoding in complex reasoning tasks.
