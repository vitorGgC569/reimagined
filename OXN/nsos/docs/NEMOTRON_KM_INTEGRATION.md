# Nemotron K·m Invariant — Cherry-pick #4

> Aplicar princípios de scaling laws para MoE Híbrido Mamba-Attention extraídos do Nemotron 3 Super (NVIDIA, abril 2026).
>
> **Paper:** `external_refs/papers/nemotron3_super_2026.pdf` (Section 2.1.1)
> **Antecessor:** `external_refs/papers/nemotron_h_2025.pdf` (Nemotron-H, foundational)
>
> Este é o **mais barato** dos 3 cherry-picks: ~1 dia de trabalho, sem retreino necessário, apenas mudança de config + ablation curta pra validar.

---

## Os 5 princípios extraídos do paper (Sec 2.1.1)

### Princípio 1 — Memory bandwidth bound
Em **low-latency serving** (nosso caso — single user contábil), MoE inference é dominado pelo **memory bandwidth de ler pesos de expert**.

Cada expert matrix tem tamanho `d × m`:
- `d` = hidden dimension do modelo
- `m` = expert FFN intermediate dimension

Reduzir custo de memória ⇒ reduzir `d` ou `m`.

### Princípio 2 — Communication bound
Em **throughput-oriented serving** (multi-cliente, batch grande), MoE distribuído é dominado por **all-to-all routing**. Volume escala como `d × K` onde `K` = experts ativos por token.

Reduzir comunicação ⇒ reduzir `d` ou `K`.

### Princípio 3 (⭐ THE INVARIANT) — Quality preservation
> "Preserving model quality requires preserving the effective nonlinear budget **K · m**."

**Para aliviar bottlenecks SEM sacrificar qualidade, K e m devem ser mantidos fixos juntos.**

Tradução prática:
- Aumentar K (mais experts ativos por token) **e** reduzir m proporcionalmente = mesma qualidade, menos custo
- Aumentar m (experts mais "gordos") **e** reduzir K = mesma qualidade
- Mas: aumentar um sem reduzir o outro = mais custo sem ganho de qualidade
- E: reduzir os DOIS = perde qualidade

### Princípio 4 — Feature rank floor
Cada task tem um **r_eff** (effective feature rank) mínimo. Se `d` cair abaixo desse, **qualidade colapsa**.

Implicação: dá pra reduzir `d` até um certo ponto, mas existe piso. Determinar empiricamente.

### Princípio 5 — Combinatorial expansion
Aumentar **ambos** N (total experts) e K (active per token) melhora qualidade **exponencialmente** via expansão de combinações de experts.

Implicação: se memory bandwidth permite, dobrar N E K é melhor que dobrar só um.

---

## Configurações atuais do Oxta vs Nemotron 3 Super

### Nemotron 3 Super (120B total, 12B active)
Da Table 1 do paper:

| Hyperparam | Valor |
|------------|-------|
| Total Layers | 88 |
| `d` (d_model) | 4096 |
| Q-Heads, KV-Heads | 32, 2 (GQA 16:1) |
| Head Dimension | 128 |
| Mamba State Dim | 128 |
| Mamba Groups | 8 |
| Mamba Heads | 128 |
| Mamba Head Dim | 64 |
| `m` (Expert Hidden Dim) | 2688 |
| Shared Expert Intermediate Size | 5376 |
| `N` (Total Experts per Layer) | 512 |
| `K` (Top-K Activated Experts) | 22 |
| MoE Latent Size (LatentMoE) | 1024 |
| MTP layers | 2 |

**K · m = 22 × 2688 = 59,136** (massivo)

### Oxta 80M atual (`_v11_80m_model_config`)

| Hyperparam | Valor | Vem de |
|------------|-------|--------|
| Total Layers | 16 | `PROFILES["hybrid_v11_colab_t4_80m"]["layers"]` |
| `d` (d_model) | 640 | idem |
| `n_heads`, `n_kv_heads` | 10, 5 (GQA 2:1) | `_v11_80m_model_config` |
| Head Dimension | 64 | implícito (640/10) |
| `m` (Expert Hidden Dim) | **2560** | HARDCODED: `dm * 4` em `jamba.cpp:1306` |
| `N` (Total Experts) | 8 | `_v11_80m_model_config["num_experts"]` |
| `K` (Top-K) | 2 | `_v11_80m_model_config["num_experts_per_token"]` |

**K · m = 2 × 2560 = 5,120** (~11× menor que Nemotron — proporcional ao 1500× menor em params)

---

## Aplicação dos princípios ao nosso caso

### Cenário de uso real (Oxta Contábil)
- **Single-user low-latency** (1 contador → 1 escritório → 1 query por vez)
- → **Princípio 1 domina**: reduzir memory bandwidth de expert loads é o ganho prático

### Sugestões a testar (mantendo K · m = 5120 invariante)

| Variante | K | m | N | K·m | Hipótese |
|----------|---|---|---|-----|----------|
| **Baseline atual** | 2 | 2560 | 8 | 5120 | Config v11 que já rodou no smoke |
| **A — mais sparse routing** | 4 | 1280 | 8 | 5120 | Princípio 5: mais combinações = qualidade ↑ |
| **B — mais experts totais** | 2 | 2560 | 16 | 5120 | Princípio 5: maior N (mais combinações), K igual |
| **C — full sparse** | 4 | 1280 | 16 | 5120 | Combinação A+B: K e N dobrados, m reduzido |

**Predição teórica:** ordenação esperada de qualidade (do melhor pro pior):
**C > A ≈ B > Baseline**

**Custo de inferência:** todas as 4 são equivalentes (K·m invariante) — só rota mais bytes vs menos.

### Variantes que QUEBRAM o invariante (não testar por ora)

- K=2, m=5120, N=8 (K·m=10240) → mais qualidade mas dobra custo de memória bandwidth
- K=1, m=5120, N=8 (K·m=5120 mas K=1 é não-MoE essencialmente)

---

## O que precisa mudar no código

### Mudança #1 — Tornar `m` configurável

**Hoje** (`jamba.cpp:1306`):
```cpp
const int hidden_dim = dm * 4;  // hardcoded
```

**Proposta:**
```cpp
const int hidden_dim = (configured_expert_hidden_dim > 0)
                       ? configured_expert_hidden_dim
                       : dm * 4;  // backward-compat default
```

### Mudança #2 — Adicionar campo ao ModelConfig

```cpp
struct ModelConfig {
    // ... existing ...
    
    // ── Nemotron K·m invariant tuning (Cherry-pick #4) ──
    // When > 0, overrides the default expert FFN intermediate dimension
    // (otherwise computed as d_model × 4).  Use this to apply the K·m
    // invariant: increase num_experts_per_token AND decrease expert
    // hidden dim proportionally, holding K × m fixed to preserve quality
    // while reducing memory bandwidth in MoE inference.
    // See docs/NEMOTRON_KM_INTEGRATION.md for the principle and tested values.
    int moe_expert_hidden_dim = 0;
};
```

### Mudança #3 — Propagar via constructor JambaBlock

Adicionar parâmetro `int configured_expert_hidden_dim` ao constructor (após `configured_top_k`).

### Mudança #4 — Expor via pybind11 + ModelConfig em Python

```cpp
.def_readwrite("moe_expert_hidden_dim", &ModelConfig::moe_expert_hidden_dim)
```

### Mudança #5 — Permitir override via profile no `train_curriculum.py`

```python
_v11_80m_model_config = {
    # ... existing ...
    "moe_expert_hidden_dim": 0,  # 0 = default (dm*4)
}
```

E criar variantes para ablation:
```python
PROFILES["hybrid_v11_colab_t4_80m_km_A"] = deepcopy(PROFILES["hybrid_v11_colab_t4_80m"])
PROFILES["hybrid_v11_colab_t4_80m_km_A"]["model_config"]["num_experts_per_token"] = 4
PROFILES["hybrid_v11_colab_t4_80m_km_A"]["model_config"]["moe_expert_hidden_dim"] = 1280
# K · m = 4 × 1280 = 5120 (invariante mantido)
```

---

## Plano de ablation no Colab

**Notebook:** `colab/oxta_ablation_nemotron_km.ipynb` (a criar)

Roda 3 smokes curtos (15 steps × 6 fases ≈ 90 steps cada) lado a lado:

```python
profiles_to_test = [
    "hybrid_v11_colab_t4_80m",          # baseline (K=2, m=2560)
    "hybrid_v11_colab_t4_80m_km_A",     # K=4, m=1280
    "hybrid_v11_colab_t4_80m_km_B",     # K=2, m=2560, N=16
]
```

Cada smoke ~10 min em T4. Total ~35-45min pra ter resposta empírica.

**Critério de decisão:**
- Qual config tem a menor loss final (média últimas 25 medições)?
- A variante vencedora vira o profile default pro treino real

**Outcome esperado** (baseado em Princípio 5 + 3):
- Config A ou C vence baseline por margem pequena (1-5%)
- Se margem é trivial, mantém baseline (menos risco de implementation bug)

---

## Decisões NÃO incluídas neste cherry-pick

### 1. LatentMoE (a inovação central do Nemotron 3 Super)

O paper introduz **LatentMoE**: projetar tokens pra dimensão latente menor `ℓ`, fazer routing/expert compute em latent, projetar de volta. Isso permite N e K maiores ao mesmo custo computacional.

**Por que NÃO implementar agora:**
- Mudança arquitetural grande (2 matrizes novas por layer MoE: down-proj + up-proj)
- Requer paper de referência separado (Elango et al., 2026 — ainda não vimos)
- Ganho marginal pro nosso scale (80M) — LatentMoE brilha em 100B+
- Tempo: 2-3 semanas vs 1 dia do cherry-pick atual

**Quando reconsiderar:** se escalar para 1B+ params E hit comm/memory bottleneck.

### 2. Multi-Token Prediction (MTP)

Nemotron usa MTP pra speculative decoding. Útil pra inference acceleration.

**Por que NÃO agora:**
- Inference time only, não afeta treino
- Cliente contábil faz <10 inferências/min — speculative decoding marginal
- Implementar requer mudança no inference engine + adicionar prediction heads
- ROI claro só em batch grande / latência crítica

**Quando reconsiderar:** após primeiro modelo treinado, se tok/s for queixa real.

### 3. NVFP4 quantization

Nemotron treinou em NVFP4 (FP4 da NVIDIA) pra eficiência de treino em B200. Não temos B200, T4 só suporta BF16+. Não aplicável.

---

## Status

- ✅ Paper lido (6 páginas principais — Sec 2.1.1 com os 5 princípios)
- ✅ Análise comparativa com config atual
- ✅ 3 variantes de ablation projetadas (mantendo K·m=5120 invariante)
- ✅ **Mudança #1**: `hidden_dim` configurável em `jamba.cpp:1311` (ternary expression com default `dm * 4`)
- ✅ **Mudança #2**: campo `moe_expert_hidden_dim = 0` em `ModelConfig` (`nsos_config.h`)
- ✅ **Mudança #3**: parâmetro `configured_expert_hidden_dim` no `JambaBlock` constructor (default 0 = backward-compat)
- ✅ **Mudança #4**: exposto via pybind11 (`bindings.cpp:120-123`)
- ✅ **Sanity compile**: `jamba.cpp` e `nsos_config.h` compilam limpos sem warnings
- ⬜ Mudança #5: profiles `_km_A`, `_km_B` em `train_curriculum.py`
- ⬜ Notebook `oxta_ablation_nemotron_km.ipynb`
- ⬜ Rodar ablation, escolher vencedor, atualizar profile default

**Esforço gasto até aqui:** ~30min (4 arquivos modificados, ~25 linhas adicionadas, zero linhas removidas — pure addition).
**Próximo passo:** criar profiles de ablation no Python + notebook de comparação. ~1-2h adicionais.
