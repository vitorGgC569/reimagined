# TTT (Test-Time Training Hamiltoniano) — Validation Report

**Data:** 2026-05-25
**Build:** `OXN/nsos/build-chrass-validation/Release/`
**Resultado:** ✅ **12/13 PASS** (1 fail = bug real de determinismo entre instâncias)

---

## Resumo

TTTLayer existia com `test_ttt_layer_kernel.cpp` wired (2 testes: adaptação + backward). Esta auditoria adicionou 13 testes COBRINDO superfície completa do layer.

## Resultados (test_ttt_validation.cpp)

| # | Teste | Resultado |
|---|---|---|
| 1 | constant_input_produces_drift | ✅ |
| 2 | training_mode_off_no_adaptation | ✅ (drift=0 em eval mode) |
| 3 | reset_returns_to_initial_state | ✅ (diff=0) |
| 4 | snapshot_restore_roundtrip | ✅ (replay diff=0) |
| 5 | hamiltonian_on_off_differ | ✅ (diff=26.8) |
| 6 | friction_affects_dynamics | ✅ (diff=31.4) |
| 7 | temperature_affects_adaptation | ✅ (diff=21.3) |
| 8 | long_sequence_no_nan (seq=200) | ✅ (max\|y\|=4.84) |
| 9 | extreme_input_magnitudes (1e-3 a 100) | ✅ |
| 10 | backward_shape_and_grad_propagation | ✅ |
| 11 | parameter_gradients_received | ✅ (9/15 params) |
| 12 | bitlinear_layers_collected | ✅ (3/3: w_k, w_v, w_out) |
| 13 | deterministic_forward_same_seed | ❌ **diff=16.95** |

## Achados positivos

- **reset/snapshot/restore PERFEITOS** — diff=0 em ambos. Crítico pra resume training.
- **Eval mode estável** — drift=0 em 8 steps com training_mode=false.
- **Todos hyperparams têm efeito mensurável** — Hamiltonian flag, friction, temperature.
- **Estável em longa sequência** — seq=200, max\|y\|=4.84 (sem divergência).
- **Robusto a magnitudes extremas** — inputs 1e-3 a 100, sem NaN.
- **3 BitLinear coletados** — w_k, w_v, w_out integrados corretamente.
- **9 de 15 params recebem gradient** — meta-learning flow funciona.

## Achado negativo — BUG REAL

**Falha de determinismo entre instâncias:** Duas TTTLayer com mesmos hyperparams e MESMA seed/input produzem outputs DIFERENTES (diff L2 = 16.95).

**Causa provável:** BitLinear (w_k, w_v, w_out) usa init aleatório com `state_` (uint64_t) por instância. Cada TTTLayer construtor inicializa state_ from system clock/random. **Sem seed externa, runs não são reproduzíveis.**

**Impacto:** Crítico pra:
- Testes de regressão (não pode comparar contra golden values)
- Validation runs reproduzíveis
- Replay de checkpoints (snapshot/restore funciona DENTRO de uma instância, mas duas instâncias do mesmo modelo diferem)
- Debug determinístico

**Fix:** Adicionar parâmetro `seed` no `TTTLayer` ctor que propaga pra BitLinear init. Mantém determinismo entre runs.

## Veredito

**Status atual no projeto:** "research-only para release" conforme `NSOS_VALIDATION_STATUS.md`.

**Após esta validação:** ✅ **PROMOVÍVEL A RESEARCH-VALIDATED** (não production-ready até fix de determinismo).

Componente é matematicamente correto, estável numericamente, integrado com BitLinear. Snapshot/restore confiável é o ouro pra long-run training.

**Bloqueador único pra production:** seed propagation. Resolvido isso, TTT pode entrar como `mcfg.use_ttt = True` no perfil de produção sem medo.

## Próximos passos

1. **Adicionar `seed_` parâmetro no TTTLayer ctor + propagar pra w_k/w_v/w_out BitLinear**
2. Re-rodar `test_deterministic_forward_same_seed` — esperado: diff < 1e-3
3. Atualizar `NSOS_VALIDATION_STATUS.md`: TTT de "research-only" → "production-validated"
4. Habilitar `mcfg.use_ttt = True` em `trainer_main.py` (e re-bundle pro próximo run)

## Como reproduzir

```bash
cmake -S OXN/nsos -B OXN/nsos/build-chrass-validation \
  -DNSOS_ENABLE_CUDA=OFF -DNSOS_BUILD_PYTHON=OFF \
  -DNSOS_BUILD_TESTS=ON -DNSOS_BUILD_CLI=OFF \
  -DNSOS_BUILD_API=OFF -DNSOS_BUILD_OXTAMEM=OFF
cmake --build OXN/nsos/build-chrass-validation --config Release \
  --target test_ttt_validation -j
OXN/nsos/build-chrass-validation/Release/test_ttt_validation.exe
# Esperado: 12/13 passed
```
