# MoE: acumulação de gradientes por banco no dispositivo

Este lote avança a remoção de desperdícios no backward agrupado. Não encerra
atividade/Adam device, a release ou o objetivo integral de camadas/Mamba-3.

## Implementação

O registro anterior fazia `slice -> add_grad` para cada weight/bias/magnitude
de cada expert ativo nos dois bancos. O slice materializava outro tensor;
a primeira contribuição ainda era clonada/copiada para o Parameter. Durante
acumulação havia também um launch de soma por parâmetro.

- Dois launches por backward com contribuições: um por banco up/down. Sem
  slices, temporários por parâmetro ou chamadas D2D para registrar gradientes.
- Cada destino tem sua própria política first-write versus add. Flags não
  derivam de norma/valores: contribuição zero permanece ativa.
- Primeiro write sobrescreve buffers retidos, inclusive não zerados. Writes
  posteriores somam na ordem dos microbatches, sem atomics entre experts.
- Offsets guardam segmentos vazios e status inválido antes de buscar scratch.
  Experts vazios não alocam gradientes nem passam a integrar o Adam.
- Parameters continuam donos dos destinos; scratch agrupado não vira alias
  de armazenamento público e não precisa sobreviver como gradiente por step.
- Descritores de destinos são reutilizados e reenviados somente quando
  endereços, membership ou política first/add mudam. Upload novo é bloqueante.
- Preflight de ambos os bancos antecede os writes/publicação. Destinos
  com shape/device inválidos ou endereços compartilhados são recusados.
  Isso não promete rollback de erros assíncronos do dispositivo.
- Contador `grouped_moe_gradient_commit` expõe tentativas host de dispatch;
  valores de saída/paridade, não só o contador, certificam execução.

Também foi corrigido `copy_gradient_activity_from`: copia o bit de contribuição
real, não o predicado legado de buffer presente. Um dense usado/zerado/clonado
e depois registrado sparse não pode reviver atividade. API dense de `.grad`
continua compatível antes de registro explícito.

## Limite de arquitetura preservado

A leitura tardia de offsets com sincronização continua para conservar o
registro host, momentos lazy e versões de experts inativos. Os novos kernels
não substituem esse requisito por gradientes zero de todos os experts.
O próximo fechamento continua sendo atividade/coorte/Adam no dispositivo,
incluindo regularização, finite gate, clipping, snapshots e versões/cache.

Contadores D2D representam chamadas de transferência, não toda leitura/write
DRAM. O kernel de acumulação também movimenta dados; não se afirma tráfego
físico zero nem ganho de cache a partir desses contadores.

## Evidência

Artefatos em `artifacts/gpu_moe_gradient_commit_20260930`. Binário anterior
preservado SHA256 `5c088f5b372529f18dc44970808d2a78be63a19afc350d16f481ceeccc9d8e3d`.
Seu rocBLAS é junction para o build; não é um pacote autocontido.

Novas verificações incluem oracle de acumulação independente com tails,
first/add por parâmetro, expert vazio NaN, bias opcional, status inválido,
união real de microbatches e estabilidade de endereços, sobrescrita de
buffers poisoned, cache de descritores e preflight/alias sem publicação.
Paridade por expert, WMMA/double oracle e checkpoint já existentes continuam
necessários e foram reexecutados. Após implementação:

- Builds Release CPU/HIP completos passaram (`cpu-build.log`, `hip-build.log`).
- CPU60/60 passou em18.08 s (`cpu-full.log`).
- HIP121/121 passou em148.52 s (`hip-full.log`), incluindo os braços grouped
  scalar/WMMA e continuação de checkpoint; directed subset2/2 em5.18 s.
- Python15/15 passou (`python-policy.log`).
- Nenhum código nativo mudou depois dessa campanha; controles adicionais
  abaixo executam os mesmos binários. Não houve treino de produção.

HIP SHA256 `d251673123ebc69f0c6ad189236c7c1ea22ed33d55a7856def1e1cb27c8b881f`.
CPU SHA256 `9602a257d783d3599077e6db9d3eb42092f6072b679dee3558c2b7b874db7b42`.

## Campanha paired e controle de variabilidade

RX7600/gfx1102, processos isolados, seed7301, precisão e política idênticas
antes/depois. Duas camadas Mamba + uma MoE E4/k2/H=D, vocab257, QAT OFF;
IDs sintéticos, sem tokenizer, não é um modelo PT-BR treinado. B2 usa SFT
heterogêneo com padding/máscara. Cada JSON inclui config, identidade, hashes,
samples, transferências e caminhos. Sem outro trabalho GPU nos probes.

16 steps/processo, primeiro excluído do p50. Série1 before->after; série2
after->before. Todos os12 probes finitos; hashes iniciais/finais iguais por
par e diferença de loss exatamente0 em todos os16 steps.

| Geometria/provider | Antes série1/2 (ms) | Depois série1/2 (ms) |
|---|---:|---:|
| D128/S256/B1 FP32 scalar | 9.950 / 10.271 | 10.825 / 9.875 |
| D768/S512/B1 BF16 WMMA | 39.060 / 39.786 | 38.916 / 39.026 |
| D128/S128/B2 BF16 WMMA | 10.893 / 11.346 | 12.152 / 11.060 |

Redução de chamadas D2D é consistente: B1 42->18, B2 40->16,24 chamadas
eliminadas/step. Bytes B1 D128 2777104->2244624; D768
45694992->26771472 (18.046875 MiB eliminados); B2 2618176->2085696.
H2D não aumentou em regime; D2H permaneceu6/step e fences8(B1)/6(B2).
32 dispatches de commit por processo =2x16, demonstrados também por valores
e hashes, não inferidos apenas do contador.

Como as formas curtas alternaram ganho e regressão, foi feito um controle
adicional definido antes da execução:64 steps/processo,8 primeiros excluídos
do p50, mesmas duas ordens. Todos os8 probes finitos e hashes finais iguais,
diferença de loss exatamente0 em todos os64 steps.

| Geometria | Antes série3/4 (ms) | Depois série3/4 (ms) |
|---|---:|---:|
| D128/S256/B1 FP32 scalar | 9.719 / 9.736 | 9.419 / 10.221 |
| D128/S128/B2 BF16 WMMA | 10.986 / 10.816 | 10.739 / 10.728 |

B2 controle reduz tempo2.25%/0.81%; largo16steps reduz0.37%/1.91%.
FP32 curto não sustenta melhora: controle+3.09%/-4.98% (redução de latência).
Os resultados ruins permanecem nos artefatos. Não promover speedup universal
nem default grouped/WMMA com esta evidência. O benefício fechado aqui é
remoção de slices/cópias/temporários e contrato de acumulação; ganho de sistema
continua limitado por outros buckets e pela fronteira host.

Reprodução:

```powershell
& OXN/nsos/artifacts/gpu_moe_gradient_commit_20260930/run_paired.ps1
& OXN/nsos/artifacts/gpu_moe_gradient_commit_20260930/summarize_paired.ps1
& OXN/nsos/artifacts/gpu_moe_gradient_commit_20260930/run_paired.ps1 -Rounds @(3,4) -Steps 64 -Cases @('short-fp32','batch-wmma')
& OXN/nsos/artifacts/gpu_moe_gradient_commit_20260930/summarize_paired.ps1 -Rounds @(3,4) -Skip 8 -Cases @('short-fp32','batch-wmma')
```

Os nomes originais recusam overwrite: reproduzir em novo diretório/rodadas.
Summaries preservados em `paired-summary.log`/`paired-control-summary.log`.
Nos controles, contadores de transferência excluem somente o primeiro step,
como definido pelo probe; o p50 acima exclui8. Não confundir as duas janelas.

Sem alegação de promoção GPU, inteligência PT-BR, superioridade SOTA ou
Mamba-3 implementado.
