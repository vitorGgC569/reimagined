# RDNA3 WMMA no treino MoE — candidato opt-in

Este lote atende ao GEMM agrupado largo do objetivo integral; não implementa
Mamba-3, WMMA de todas as projeções, Adam esparso device ou uma release GPU.

## Implementação e contrato

- Novo provider em `src/cuda/moe_training_wmma.cu`: forward, dX e dW, operands
  BF16/FP16 e acumulador FP32. Pesos mestres/gradientes continuam FP32.
- CTA128 (quatro wave32), tile32x32/K32, fragments16x16/K16 rocWMMA. LDS8KiB,
  leituras contíguas; transposição de W/grad ocorre em LDS. Epílogo usa o store
  público do fragment, não suposições de mapping de registradores.
- Cada CTA consulta offsets no dispositivo. Experts vazios não leem pesos,
  nem escrevem gradientes. Caudas são preenchidas com zero; nenhuma barreira
  tem saída divergente. Magnitude/bias/squared-ReLU e STE QAT são preservados.
- `NSOS_MOE_WMMA_TRAINING=1` / `--moe-compute-policy wmma-v1`, default OFF.
  Seleção: linhas totais >=16, inputs/outputs >=32 e compute BF16/FP16.
  FP32 e formas curtas usam explicitamente o provider escalar, não fallback de
  um launch WMMA falho. Cada segmento pode ser curto, sem download para decidir.
- Requer headers rocWMMA e targets exclusivamente gfx1100/1101/1102 neste
  build. Binários HIP heterogêneos têm o provider desativado até compilação
  separada por arquitetura. GPU ativa deve ser compilada e wave32 RDNA3.
  CUDA/CPU não simulam suporte. Modelo CPU/misto e flags incompatíveis falham.
- Política WMMA e precisão são capturadas no tape; alteração antes do backward
  é rejeitada antes de registrar gradientes. Checkpoint distingue
  `moe.training_wmma_policy=rdna3_lowp_tile32_minrows16_mindim32_v1`.
  O campo base grouped v2 continua descrevendo preparação QAT/segmentação.
- Python verifica identidade nativa antes do treino. `grouped-v1` desativa WMMA
  explicitamente para controle escalar. `--wmma-moe` no probe implica grouped.
- Contador `grouped_moe_wmma_gemm` contabiliza tentativas de dispatch host,
  não instruções executadas nem trabalho útil de experts vazios.
- A leitura tardia dos offsets/registro dos parâmetros ativos permanece.
  Este lote não promete zero D2H, HIP Graph de treino ou ausência de CPU.

## Evidência e validação

Artefatos em `artifacts/gpu_moe_wmma_20260930`, incluindo falhas iniciais.
Validação e medições deste lote são realizadas após a implementação.

Suítes finais após as últimas correções: **CPU 60/60**, 18.05 s,
`cpu-full-policy-final.log`; **HIP 121/121**, 141.92 s,
`hip-full-policy-final.log`. Ambos os builds Release completos passaram.

- Paridade per-expert independente: FP32/BF16/FP16, referência/QAT, linear
  exato/RMS+magnitude, bias, padding, caudas e expert vazio. Tolerâncias
  existentes não foram afrouxadas. Contadores exigem exatamente seis GEMMs
  WMMA por forward/backward elegível; zero nos casos FP32/curtos.
- Oráculo double independente dos três produtos: D65/H67, 49+7 linhas ativas,
  expert vazio com NaN e destinos sentinela. Primeiro teste exigia igualdade
  bitwise indevida com double e observou diferença inicial de um ULP.
  O contrato foi corrigido para o limite de erro FP32
  `gamma_(2*paddedK) * sum(abs(a*b))`, com zeros/sentinelas exigidos exatamente.
  Máximo observado 1.19209e-7, 0.0141843 do limite, BF16 e FP16.
- Checkpoint MoE WMMA: 33 tokens/D64, quatro steps, loss/pesos/momentos bitwise
  entre repetições/retomada; rejeição de política scalar sem mutação. Este
  braço tem QAT OFF; não certifica checkpoint QAT por associação.
- Binário anterior real rejeitado antes de qualquer sample, native_verified
  false, ao solicitar WMMA. Nenhum request ignorado é chamado de aceleração.
- ISA extraída do objeto final: seis kernels com `v_wmma_*bf16`/`v_wmma_*f16`;
  metadata wave32, LDS8192, VGPR57–60, private segment zero.
- Testes de política Python: 14 casos. Erro inicial do fixture `num_heads`
  inexistente foi corrigido para o contrato nativo `n_heads`/`n_kv_heads`.
- Probe nativo que deliberadamente ignora o configurador Python confirma
  rejeição de WMMA=1/grouped=0, em `native-incompatible-policy.log`.

## Medição e limites

A campanha paired usa o mesmo binário, seeds, pesos e inputs. Probe usa IDs
sintéticos, duas camadas Mamba e uma MoE
E4/k2/H=D, vocab257, QAT OFF, sem tokenizer. Não é qualidade linguística nem
throughput de um modelo 71M. Medições GPU devem ocorrer sem testes concorrentes.

29 steps/processo, primeiro excluído; segunda série inverte a ordem dos três
braços. Todos os 18 probes executaram com parâmetros/gradientes/loss finitos e
mesmos hashes de pesos iniciais por geometria. Repetição de cada braço produz
mesmo hash final e diferença de loss zero em todos os 29 steps. Isso NÃO é
equivalência bitwise entre providers; hashes BF16 finais são distintos.
Na primeira série, diferença máxima de loss WMMA versus ordered/grouped:
B2 0.003159/0.001778; largo 0.011599/0.013611 ao longo dos 29 steps. Diferenças
de ordem aritmética propagam pela trajetória de treino; paridade local não
certifica convergência ou qualidade equivalente. A ablação de qualidade segue
obrigatória antes de promover esse provider.

| Geometria | Ordered série1 / série2 (ms) | Grouped scalar (ms) | Grouped WMMA (ms) |
|---|---:|---:|---:|
| D128/S32/B1 FP32 | 7.90 / 9.99 | 20.30 / 8.42 | 7.28 / 7.32 |
| D128/S256/B2 BF16 | 13.51 / 13.42 | 12.26 / 12.28 | 12.15 / 12.25 |
| D768/S512/B1 BF16 | 37.74 / 40.06 | 41.92 / 43.84 | 36.64 / 37.78 |

- Largo: WMMA reduz tempo aproximadamente 12.6%/13.8% versus grouped scalar
  e 2.9%/5.7% versus ordered. É ganho end-to-end deste probe, não speedup
  isolado de GEMM nem certificação de todo modelo/treino QAT.
- B2/D128: diferença WMMA versus grouped é pequena (menos de 1%); não promover
  um ganho universal com essa variabilidade.
- FP32: ZERO WMMA dispatches; grouped/flag-WMMA têm pesos finais idênticos.
  A grande variação da primeira série não pode ser atribuída à WMMA.
  Controle adicional de 49 steps, primeiro excluído: ordered8.30 ms,
  grouped7.70 ms, flag-WMMA7.67 ms, hashes grouped/flag iguais. Os três probes
  adicionais também terminaram finitos. Resultados ruins iniciais preservados.
- BF16: 174 tentativas WMMA =6 GEMMs x29 steps. Ordered/grouped: zero.
- D2H não mudou: ordered6 calls/44B; grouped e WMMA6 calls/48B. Stream fences
  B1 ordered7 versus grouped/WMMA8; B2 ordered5 versus grouped/WMMA6.

HIP final SHA256:
`551f3d7fadf009c693dae567cfeed2226f56d38b047ec133162ecf6a68737837`.
CPU final SHA256:
`c605621ed26e88eba2d86512c164dda6ff2fb4e14feefe635cbc343dde4ba761`.

```powershell
python OXN/nsos/scripts/audit_training_integrations.py --build-dir OXN/nsos/build-gm-hip --variant moe --redesign --wmma-moe --d-model 768 --seq-len 512 --batch-size 1 --precision bf16 --steps 29 --output <novo-arquivo.json>
```

Trocar `--wmma-moe` por `--grouped-moe` para controle escalar; omitir os dois
para ordered. Executar cada braço em processo separado, sem workload GPU
concorrente. Cada JSON contém identidade, hashes, samples e contadores.

Baseline preservado SHA256:
`996d66cb7b4b6d35fb11342d215d3c66acc55cba092ec8d2e6a9eccec4eb3329`.
Sua árvore `rocblas` é junction para o build; não é distribuição autocontida.

Mesmo quando uma geometria ganha, default segue OFF até campanha mais ampla.
Não alegar superação de SOTA, qualidade ou fechamento do objetivo integral.
