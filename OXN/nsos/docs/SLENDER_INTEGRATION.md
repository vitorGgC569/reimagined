# Slender-Mamba Integration Plan

> Cherry-pick #2 (pre-training critical): aplicar quantização ternária 1.58-bit nas camadas de **embedding e output projection**, não apenas nas BitLinear intermediárias.
>
> **Paper:** Yu et al. 2025, COLING 2025. PDF em `external_refs/papers/slender_mamba_2025.pdf`.
> **Repo de referência:** `external_refs/repos/Slender-Mamba/`.

---

## Por que importa (resumo do paper)

Em Mamba-2 170M, a distribuição de parâmetros é:

| Componente | Params | % do total |
|------------|--------|------------|
| Linear layers (BitLinear hoje) | 90M | **53.7%** |
| Embedding | 38.7M | **23.1%** |
| Head (lm_head) | 38.7M | **23.1%** |
| Convolutional | 215k | 0.13% |
| Normalization | 36k | 0.02% |

BitNet original quantiza só linear → **48.4% de redução em bits**.
Slender-Mamba quantiza linear + embedding + head → **90% de redução em bits**.

**Resultado empírico (Table 2 do paper):**

| Model | HellaSwag | PIQA | ARC-E | Avg |
|-------|-----------|------|-------|-----|
| Mamba-2 baseline (FP) | 29.6 | 59.1 | 40.9 | 40.3 |
| Slender-Mamba (full) | 27.7 | 55.6 | 38.6 | **40.6** |

**Slender-Mamba full quantization MATCH o baseline FP** (40.6 vs 40.3) — degradação ~zero, redução de bits 1.86×.

---

## Fórmulas a implementar (Sec 3.3 do paper)

### Quantização do peso de embedding (treino)

```
W̃ = clip(Round(W / β), -1, 1)        # (eq 7)
β = max( (1/(V·D)) · Σ|W_vd|, ε )     # (eq 8)
```

Onde `V = vocab_size`, `D = embedding_dim`, `ε = 1e-5`.

Diferença do BitLinear: o denominador é `V·D` em vez de `D_in·D_out` (conceitualmente o mesmo: média absoluta dos pesos).

### Lookup + LayerNorm + activation quant (forward)

```
E = W̃[x]                                  # (eq 9) — lookup ternário
Ê = (E - μ(E)) / sqrt(Var(E) + ε)         # (eq 10) — LayerNorm
γ_i = max( max_j(|Ê_ij|), ε )             # (eq 11) — escala por token
Ẽ_i = clip(Round(Ê_i · Q_b / γ_i), -Q_b, Q_b - 1)  # (eq 12) — 8-bit, Q_b = 127
E_out = (Ẽ ⊙ γ · 1_D) · (β / Q_b)         # (eq 13) — dequantização
```

### Backward (straight-through estimator)

Mesmo padrão que BitLinear: gradientes fluem como se a quantização não existisse. Isso já está implementado no nosso `BitLinear` — vamos reusar a primitiva.

---

## Mapeamento no nosso código

### Arquivos a modificar

| Arquivo | Modificação | Esforço |
|---------|-------------|---------|
| `OXN/nsos/include/embedding.h` | Adicionar `set_slender_quantization(bool)`, `slender_quantization_` field | 5 min |
| `OXN/nsos/src/embedding.cpp` | Adicionar branches quantizados em `forward()` e `forward_batch()`. Usar straight-through estimator no backward. | 1-2 dias |
| `OXN/nsos/include/jamba.h` | Confirmar que `lm_head` usa `BitLinear` (não `Linear` regular). Se não, trocar. | 30 min |
| `OXN/nsos/src/jamba.cpp` | Se mudança no header, ajustar uso. | 30 min |
| `OXN/nsos/include/nsos_sdk.h` | Adicionar campo `use_slender_quantization` em `ModelConfig` | 5 min |
| `OXN/nsos/src/bindings.cpp` | Expor o novo campo via pybind11 | 5 min |
| `OXN/nsos/scripts/train_curriculum.py` | Aplicar `set_slender_quantization(True)` se config ativa | 15 min |
| `OXN/nsos/tests/test_bitlinear.cpp` (extend) | Testes de paridade Slender vs baseline | 2-3 horas |

### Novo arquivo a criar

| Arquivo | Conteúdo |
|---------|----------|
| `OXN/nsos/tests/test_slender_embedding.cpp` | Teste unitário do BitEmbedding: round-trip quantize/dequantize, gradient flow via STE, paridade com referência |

### Total: ~3-5 dias de trabalho focado

---

## Plano de implementação por fases

### Fase 1: Header changes + config (Dia 1, manhã)

1. Modificar `embedding.h` pra ter:
   ```cpp
   class Embedding {
     // ... existing ...
     void set_slender_quantization(bool enabled);
     bool slender_quantization_enabled() const;
   private:
     bool slender_quantization_ = false;
   };
   ```

2. Adicionar a `ModelConfig`:
   ```cpp
   struct ModelConfig {
     // ... existing ...
     bool use_slender_quantization = false;  // Cherry-pick #2 (Slender-Mamba head-to-toe)
   };
   ```

3. Expor via pybind11 em `bindings.cpp`:
   ```cpp
   .def_readwrite("use_slender_quantization", &ModelConfig::use_slender_quantization)
   ```

### Fase 2: Forward quantizado (Dia 1, tarde + Dia 2)

Em `embedding.cpp::forward_batch()`, adicionar branch:

```cpp
if (slender_quantization_) {
  // 1. Calcular β = mean(|W|) — pode cachear entre fwd se peso não mudou
  // 2. Quantizar W̃ = clip(round(W/β), -1, 1)
  // 3. Lookup ternário: E = W̃[indices]
  // 4. Dequantizar parcial: E_real = E * β
  // 5. LayerNorm: Ê = (E_real - μ) / sqrt(var + eps)
  // 6. Per-token quant: γ = max(|Ê|), Ẽ = clip(round(Ê * Qb / γ), -Qb, Qb-1)
  // 7. Output: E_out = Ẽ * γ * (β / Qb)
  // Return E_out
}
// else: existing FP32 path
```

### Fase 3: Backward via straight-through estimator (Dia 2-3)

Em `embedding.cpp::backward_batch()`:

```cpp
if (slender_quantization_) {
  // Straight-through: gradiente flui como se quantização não existisse
  // Apenas acumula grad_output nas linhas tocadas (mesmo que FP32 hoje)
  // Mas com escalonamento β/Qb pra compensar a dequantização do forward
}
```

### Fase 4: Output projection (lm_head) — Dia 3

Verificar `jamba.cpp` — provavelmente `lm_head` já é `BitLinear`. Se sim, nada a fazer aqui. Se não, trocar `Linear` → `BitLinear`.

### Fase 5: Tests (Dia 3-4)

`tests/test_slender_embedding.cpp`:

```cpp
TEST(SlenderEmbedding, RoundTripQuantization) {
  // Verifica que quantize → dequantize preserva valores até β/Qb
}

TEST(SlenderEmbedding, GradientFlowThrough) {
  // Verifica STE: gradiente de saída quantizada bate com gradiente FP32
  // dentro de tolerância pequena
}

TEST(SlenderEmbedding, ParityWithBaseline) {
  // Treina 100 steps com slender ON vs OFF
  // Verifica que loss desce em ambos os casos
  // Verifica que slender usa ~16× menos memória pro peso
}
```

### Fase 6: Integration teste em Colab (Dia 4-5)

1. Notebook `colab/oxta_ablation_slender.ipynb`:
   - Roda 500 steps com `use_slender_quantization = False` (baseline)
   - Roda 500 steps com `use_slender_quantization = True` (slender)
   - Compara loss curves lado a lado
   - Verifica disk size do pack final

2. Se loss slender é ~igual ao baseline → ship!
3. Se loss slender degrada >5% → debug (provavelmente issue de STE ou β computation)

---

## Risco e mitigação

| Risco | Probabilidade | Mitigação |
|-------|--------------|-----------|
| Loss diverge com embedding ternário | Baixa (paper provou que funciona) | Validar com ablation no Dia 4 antes de commitar |
| STE backward bugado | Média | Test unitário dedicado, gradient check |
| Slowdown no forward (math extra) | Baixa | Cachear `W̃` entre steps se `W` não mudou |
| Incompatibilidade com checkpoints antigos | Garantida | Tag de safety já criada: `oxta-pre-cherrypicks-2026-05-24` |

---

## Critério de "Done"

- ✅ Branch `feature/cherry-picks-3` tem todos os arquivos modificados
- ✅ `test_slender_embedding` passa (round-trip + STE + parity)
- ✅ Notebook `oxta_ablation_slender.ipynb` mostra loss curves lado a lado
- ✅ Disk size do pack final é ~60% menor com slender ativado
- ✅ Eval (HellaSwag, PPL Wikitext) com slender está dentro de 5% do baseline
- ✅ Documentação no `OXN/nsos/docs/RELEASE.md` menciona Slender como opt-in feature

Quando todos esses bullets estão check, merge `feature/cherry-picks-3` → `main` via PR com diff visível.

---

## Status atual

- ✅ Paper lido (6 páginas principais + apêndice)
- ✅ Código de referência inspecionado (`Bitembedding.py` em Slender-Mamba repo)
- ✅ Branch criada: `feature/cherry-picks-3`
- ✅ Tag de safety criada: `oxta-pre-cherrypicks-2026-05-24`
- ✅ **Header changes (Fase 1)** — `embedding.h` extendido com `set_slender_quantization(bool)` + estado privado. Compila limpo, zero impacto comportamental no path default.
- ⬜ Forward quantizado (Fase 2) — próximo passo
- ⬜ Backward STE (Fase 3)
- ⬜ Output projection (Fase 4)
- ⬜ Tests (Fase 5)
- ⬜ Ablation Colab (Fase 6)
- ⬜ Backward STE (Fase 3)
- ⬜ Output projection (Fase 4)
- ⬜ Tests (Fase 5)
- ⬜ Ablation Colab (Fase 6)
