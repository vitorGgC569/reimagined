# Engenharia de treinamento GPU — RX 7600 / gfx1102

Data: 29/09/2026. Escopo: motor nativo NSOS, treinamento e interligação das camadas. Este relatório distingue código corrigido, medições locais e propostas ainda não implementadas. Não certifica qualidade linguística nem superioridade ao estado da arte.

Atualização posterior: os quatro redesenhos prioritários receberam implementação
opt-in e validação própria em [GPU_TRAINING_REDESIGN_2026-09-29.md](GPU_TRAINING_REDESIGN_2026-09-29.md).
As medições e os itens "não implementados" abaixo descrevem esta etapa anterior;
consultar o relatório novo para o estado atual e as lacunas que continuam abertas.

## 1. Resultado executivo

O principal gargalo do piloto é o backward Mamba, não a transferência de grandes tensores à CPU. Na configuração medida, cerca de 69% do tempo fica no backward; o histórico de estados ocupa 192 MiB por camada. O conjunto das 16 camadas retém aproximadamente 3 GiB apenas nesse histórico. A hipótese de um modelo ternário de decode residente em 32 MiB de cache não descreve este treinamento.

Nesta etapa foram implementados:

1. Um profiler reproduzível de steps reais do Trainer, com identidade do binário, dados, tokenizer, políticas numéricas, memória e transferências.
2. Um ensaio independente de integração de seis topologias: Mamba, attention de substituição, composição paralela, MoE, KAN e TTT.
3. Correções no caminho MoE determinístico: acesso a ponteiros GPU somente por helpers de cópia apropriados e alinhamento de dispositivo do gradiente de roteamento antes do VJP.
4. Remoção de um clone redundante no backward de referência do BitLinear, sem mudar sua matemática: 17 cópias e 56 MiB eliminados por step do piloto.
5. Regressões de paridade CPU/GPU de forward e gradientes do MoE, além de preservação bit a bit do gradiente de entrada e acumulação do BitLinear, com/sem LoQA e com/sem modo linear exato.

As propostas de SSD matricial, attention de treino em tiles, MoE inteiramente no dispositivo e HIP Graphs de treinamento abaixo NÃO foram implementadas nesta etapa.

## 2. O que realmente foi medido

- Windows, TheRock em `C:/TheRock/build`, RX 7600, gfx1102, wave32; dispositivo discreto selecionado, não a iGPU gfx1103.
- 71.245.696 parâmetros treináveis; 71.419.264 elementos no registro, em 404 tensores.
- 16 camadas Mamba faithful, dimensão 768, estado 64, 24 heads internos de dimensão 64; sequência 512, batch 1, vocabulário 16.384 após os tokens especiais.
- Attention, MoE, KAN, TTT, CHRASS e Slender NÃO estão ativos nesse piloto de throughput.
- Pesos mestres e gradientes FP32. BF16 é precisão dos GEMMs, com acumulação FP32; não significa treinamento inteiro BF16 nem treinamento ternário. QAT desligado.
- hipBLAS clássico; Lt não promovido. Reduções determinísticas e execução GPU estrita mantidas.
- Mesmo seed 7301 e mesmas janelas consecutivas do primeiro shard educacional preparado. O tempo inclui retorno da loss após o commit do otimizador; aquecimento excluído.
- Os ensaios são descartáveis: executam atualizações reais, mas não salvam um modelo final de produção.

### Throughput: histórico completo, sem selecionar apenas o melhor resultado

Artefatos em `OXN/nsos/artifacts/gpu_engineering_20260929/` dentro deste workspace.

| Artefato JSON | Steps / aquecimento | Mediana tokens/s | Observação |
|---|---:|---:|---|
| `baseline.json` | 60 / 10 | 1.469,1 | Perfil de controle: forward chunked desligado, backward chunk 128 |
| `chunked-fp32.json` | 60 / 10 | 2.127,3 | Forward chunk 128, backward chunk 32 |
| `chunked-bf16.json` | 60 / 10 | 2.073,5 | Houve desaceleração durante o ensaio; mantido no registro |
| `bf16-checkpoint.json` | 60 / 10 | 1.781,5 | Recomputação seletiva do histórico |
| `chunked-fp32-repeat.json` | 120 / 20 | 2.120,6 | Repetição FP32 |
| `chunked-bf16-repeat.json` | 120 / 20 | 2.233,2 | Repetição BF16 |
| `bf16-final-binary.json` | 120 / 20 | 2.236,1 | Após correção MoE, antes da remoção do clone |
| `bf16-copy-final.json` | 120 / 20 | 2.410,8 | Primeiro ensaio após remoção do clone |
| `bf16-copy-ab-before.json` | 120 / 20 | 2.398,7 | Binário anterior preservado, executado novamente |
| `bf16-copy-ab-after.json` | 120 / 20 | 2.414,1 | Binário novo, imediatamente após o controle |

A comparação inicial de perfis indicou aproximadamente 45% de ganho com a geometria chunked FP32 e mais 5% com BF16 nas repetições. São mudanças de configuração de caminhos já existentes, não kernels inteiramente novos escritos nesta etapa. O baseline não representa necessariamente todos os defaults de outros executáveis do projeto.

A comparação temporal 2.236 → 2.411 NÃO demonstra ganho de 7,8% causado pela remoção da cópia: o controle antigo também ficou mais rápido na repetição. No A/B consecutivo, a diferença foi apenas +0,64%, pequena demais para atribuição firme sem repetições adicionais e controle de clocks/potência. Nenhuma causa para essa variação do ambiente foi comprovada.

O ganho estrutural da alteração é verificável independentemente do tempo: 155 → 99 MiB de D2D por step, ou -36,1% em bytes; 51 → 34 chamadas, ou -33,3%. As 120 losses retornadas foram idênticas entre os binários no A/B. Isso não prova identidade de todos os pesos nem generalização.

### Identidade dos artefatos

| Item | SHA-256 |
|---|---|
| Binário antes das correções MoE | `3de2df4510db61d9cba7e7170b11609246bcfc02b14087c061c9837d06de1388` |
| Binário após MoE / antes do clone | `922be83807d501f31642d5ab48dd1becd4ea95b0527985468bdcd4cb60eaa8a1` |
| Binário final após clone | `05ada6425d5917f284ea7cedc477713487a13d8920ec788df55e2b5405007242` |
| Shard usado | `5d84f01dd0b6551d0e2008d3eada22c5dbf7ca49e3e47e69eebec5912d3210c0` |
| Tokenizer usado | `6993b4ed5534dacc44623e483c70dc71b2db76ea184e97eac6d7f406ed986b35` |

Cada JSON também registra hashes dos scripts, configuração, variáveis efetivas e identidade de execução. O primeiro e o segundo binários foram preservados nos artefatos; o segundo também está no diretório isolado `ab-before-copy`. Nenhum binário do build foi substituído para executar o A/B.

## 3. Onde está o custo

Perfil por buckets antes da remoção da cópia, `bf16-buckets.json`, 40 steps / 10 de aquecimento:

| Etapa | Média ms/step | Fração aproximada |
|---|---:|---:|
| Preparação | 1,96 | 0,9% |
| Forward | 38,37 | 16,7% |
| Loss | 0,97 | 0,4% |
| Backward | 157,91 | 68,8% |
| Otimizador | 29,88 | 13,0% |
| Restante | 0,48 | 0,2% |
| Total | 229,56 | 100% |

Um ensaio separado, com eventos por camada (`bf16-stage-diagnostic.json` e `mamba-stages.log`), somou aproximadamente 71,28 ms no scan backward, 25,16 ms nas projeções agrupadas, 15,27 ms em norma/gate e 12,46 ms na convolução backward. Esses números são diagnósticos: os fences perturbam o tempo, não devem ser somados a medições sem instrumentação como se fossem da mesma execução.

Por Amdahl, dobrar apenas uma etapa não dobra o modelo inteiro. Mesmo dobrar todo o backward de 68,8% daria aproximadamente 1,52x no conjunto, assumindo que o restante não mudasse. Melhorias isoladas de GEMM, scan e launch não podem ter seus speedups multiplicados indiscriminadamente.

### Tráfego e memória

No binário final, por step medido: 6 H2D / aproximadamente 8,2 KiB; 4 D2H / 20 bytes; 34 D2D / 99 MiB; zero chamadas contabilizadas de sincronização global do dispositivo e 4 sincronizações de stream. Essas estatísticas cobrem os pontos instrumentados do runtime, não substituem um trace HIP completo, contagem de dispatches ou medição de banda física da VRAM.

Os pequenos retornos incluem norma/clipping, controle de segurança do otimizador e loss. O objetivo deve ser reduzir dependências host no caminho crítico mantendo os contratos de segurança, não apenas reduzir os 20 bytes.

Antes do clone: pico vivo de aproximadamente 5.389 MiB e reservado de 5.470 MiB; depois: 5.374 MiB e 5.438,5 MiB. Checkpointing no perfil anterior reduziu o pico vivo para aproximadamente 2.494 MiB e reservado para 2.590 MiB, com throughput cerca de 20% inferior à repetição BF16 sem checkpointing. Memória em cache do pool não equivale automaticamente a vazamento.

O cache de pesos de baixa precisão registrou 33 misses iniciais e 3.927 refreshes por versão ao longo de 120 steps, sem evicções ou bypass por orçamento. Os refreshes são esperados: o otimizador modifica os pesos. Reutilizar conversões antigas para fabricar hits seria incorreto.

## 4. Integrações: executável não significa plenamente otimizado

Os resultados abaixo são do binário final, arquivos `integration-*-copy-final.json`: modelos pequenos de 2 camadas, dimensão 128, sequência 32, vocabulário 257, 5 steps. O primeiro step é excluído dos contadores. A inspeção de pesos/gradientes na CPU ocorre depois da captura dos contadores. Os seis ensaios tiveram loss e parâmetros finitos.

| Topologia | D2H chamadas / bytes por step | D2D chamadas | Parâmetros com gradiente não zero / treináveis |
|---|---:|---:|---:|
| Mamba | 4 / 20 | 6 | 28 / 28 |
| Attention de substituição | 4 / 20 | 16 | 26 / 27 |
| Mamba + attention em paralelo | 4 / 20 | 18 | 43 / 44 |
| MoE | 12 / 49.704 | 296 | 54 / 54 |
| KAN | 8 / 36 | 6 | 34 / 34 |
| TTT | 36 / 148 | 149 | 30 / 30 |

O parâmetro sem gradiente nas duas variantes attention é `layers.0.attn.attn.ssa_wsel`, seletor da SSA desativada: não é uma conexão ativa quebrada. MoE também registrou 279 sincronizações de stream por step nessa pequena configuração. Não comparar seu throughput com o piloto de 71M nem extrapolar essas contagens para outros tamanhos.

O problema corrigido no MoE era real: o modo estrito recusava acesso host a Tensor GPU; após corrigir os ponteiros destinados ao helper de cópia, o teste expôs a incompatibilidade de dispositivo do VJP do router. Ambas foram corrigidas. A regressão nativa compara forward e gradientes CPU/GPU, em batch com comprimentos distintos, e exige gradientes de experts e router. Não foi desativado o modo estrito para contornar o erro.

Os arquivos anteriores `integration-attention.json` e `integration-attention-final.json` usavam a composição paralela default. Para distinguir corretamente as seis topologias, usar `*-final-v2.json` ou, preferencialmente, os novos `*-copy-final.json`.

## 5. Redesenho recomendado, por camada e prioridade

Esta seção é plano de engenharia ainda pendente, salvo as correções explicitamente descritas acima.

### P1 — Mamba: histórico por fronteira de chunk e SSD matricial

Pontos de entrada: `Mamba2SSD::forward_faithful`, `backward_faithful`, projeções agrupadas e kernels em `cuda/mamba_kernels.cu`.

Hoje já existem projeções agrupadas, convolução fused, decay pré-calculado, layouts state-major e scan backward em chunks. Os contadores confirmaram os caminhos GPU e ausência de fallback host no piloto. Entretanto, ainda se materializa o histórico completo `[B,S,H,P,N]`; o kernel chunked não elimina, por si só, esse custo nem equivale automaticamente a um SSD matricial otimizado.

Proposta: salvar estados nas fronteiras dos chunks, recomputar estados locais por tile durante o backward e expressar subproblemas SSD como operações matriciais. Com chunk 64, o exemplo de fronteiras a cada chunk usa 3 MiB por camada, contra 192 MiB do histórico completo; isso NÃO é estimativa da memória total do algoritmo. É necessário contabilizar fronteira inicial/final, gradientes, projeções e workspace de cada tile.

Manter FP32 onde a recorrência exige estabilidade, preservar discretização, clamp, softplus, causalidade e ordem de norma/gate. Tiles completos arbitrários não cabem nos 64 KiB de LDS; dividir N/P/T e verificar registradores, spills e ocupação. M=512 no treinamento não tem o desperdício de tile do GEMV M=1 de decode. Usar GEMMs BF16/FP32 ou WMMA nos subproblemas adequados precisa de medição específica na RDNA3.

Essa direção parte da decomposição algorítmica SSD em cálculo intra-chunk, estados, propagação entre chunks e saída. Não transfere automaticamente os ganhos publicados para a RX 7600. [Descrição original do algoritmo SSD](https://tridao.me/blog/2024/mamba2-part3-algorithm/).

Critérios: gradcheck em shapes pequenos; CPU/GPU forward e VJP de entrada/parâmetros; comprimentos não múltiplos do chunk; sequências longas; continuação de checkpoint; política numérica versionada; redução comprovada de memória e tempo no step completo.

### P1 — Otimizador: geometria da norma, clipping no dispositivo

Pontos: `clip_gradients`, `apply_optimizer_step`, `multi_tensor_sqsum_deterministic_partials_kernel`.

A redução determinística da norma usa um bloco de 256 threads por tensor, com loops internos proporcionais ao tamanho do tensor; um embedding grande e um bias pequeno recebem a mesma geometria inicial. A finalização soma parciais em ordem fixa. O update AdamW já possui chunking multi-tensor; não é correto dizer que ainda há um lançamento por peso em todo o caminho.

Proposta: dividir também a redução da norma em chunks de elementos, com redução hierárquica determinística; manter coeficiente de clipping no dispositivo; combinar os resultados de finitude, norma, status e loss em um retorno final pequeno. Primeiro decompor os 29,88 ms do bucket: nem todo esse tempo é da norma.

Não retirar verificações de NaN/Inf nem permitir commit parcial. Preservar cancelamento, counters, momentos, weight decay, acúmulo e retomada. Mudar a árvore de redução muda arredondamento: requer identidade explícita da política e critérios de compatibilidade, não uma promessa de identidade bit a bit com a política antiga.

### P1 — Fusões locais, embedding e temporários

O clone de `grad_pre` no backward de referência do BitLinear foi removido. Os clones de input salvos no forward foram preservados: protegem contra mutação in-place pelo chamador. Eliminar snapshots exige contrato de imutabilidade ou versionamento, não substituição cega por aliases.

Próximos candidatos:

- Fundir SiLU/gate, RMSNorm e gamma learned no Mamba, com backward correspondente e ordem matemática preservada.
- No embedding, a redução determinística CSR por IDs presentes ainda produz `d_w` denso zerado antes de acumular. Acumular diretamente por linha presente no gradiente existente evita um temporário `[vocab,d_model]` de 48 MiB neste modelo. Preservar limpeza entre steps, primeira contribuição, IDs repetidos e contribuição do LM head com pesos amarrados.
- Planejar lifetimes dos buffers de projeção e gradiente; reutilizar apenas após o último consumidor na stream correta.
- Avaliar LM head + cross-entropy em blocos de vocabulário. Hoje logits e gradiente de logits têm 32 MiB cada nessa configuração. O bucket da loss isolada é menor que 1 ms; o benefício relevante é memória e tráfego da projeção/backward, não uma promessa de grande ganho apenas acelerando a loss. [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009).

### P1 para modelos híbridos — Attention de treino exata em tiles

Pontos: `Attention::forward_exact_gpu`, `backward_exact_gpu` e `cuda/attention_train_kernels.cu`.

O treino já usa forward/backward de attention exatos, incluindo derivada do softmax. O limite atual é a materialização de intermediários quadráticos `[B,H,S,S]`, expansões GQA e transposes. O softmax de treino contém loops seriais por linha. O caminho de decode online não elimina esses custos de treino.

Proposta: attention tiled com softmax online/logsumexp e backward exato recomputando tiles, indexação nativa de heads GQA sem expandir K/V e máscaras causal, janela e padding compartilhadas entre forward/backward. Preservar RoPE e contratos de comprimentos variados. [FlashAttention-2](https://arxiv.org/abs/2307.08691) fundamenta tiling/particionamento; seus números em outras GPUs não são previsões locais.

### P1 para MoE — roteamento, dispatch e VJP no dispositivo

A correção atual restabelece a execução estrita, mas mantém seleção/ordenação no host, cópias por linha e VJP de routing calculado parcialmente na CPU. Mesmo o caminho GPU batched ainda consulta offsets no host e agenda experts individualmente.

Proposta: top-k estável com desempate explícito; histogramas/prefix sums; permutação CSR no dispositivo; grouped GEMM apenas dos experts ativos; combine e VJP do router ordenados na GPU. Reusar no backward a permutação e pesos de roteamento realmente usados no forward. Remover buffers de scatter densos onde a combinação segmentada substituir a mesma operação.

Validar experts vazios, top-k empatado, capacidade/overflow, todos os tokens em um expert, balanceamento, router task-gradient e reprodutibilidade. Não trocar pelo caminho atômico não determinístico silenciosamente.

### P2 — KAN

`BitFastKANLayer` ainda calcula escalas de quantização base/RBF como floats host via `tensor_abs_mean`; na configuração ensaiada isso acrescenta quatro D2H por step para duas camadas. A base RBF expandida é materializada e salva.

Proposta: escalas como tensores GPU, usando contratos de quantização já presentes no BitLinear; produção da base por tiles e fusão com projeção; recomputação seletiva no backward. LUT de RBF é aproximação, não otimização exata: precisa de flag de pesquisa e avaliação de erro de função, derivada e qualidade.

### P2 — TTT

`TTTLayer::forward` percorre os tokens no host, lê `row_error.norm()` e mantém histórico da matriz adaptativa por linha. A sequência 32 acrescentou 32 retornos escalares à CPU. Pequeno volume em bytes não elimina a dependência serial.

Proposta: norma/clipping e atualização causal no dispositivo, com tiles de estado e checkpointing por blocos. Garantir estado isolado por sessão e reset/continuação corretos. Não presumir que a matriz inteira cabe em registradores ou LDS.

Limite de inteligência/gradiente: o backward documentado é truncado; não diferencia a recorrência interna de adaptação através de todos os tokens anteriores. A paridade CPU/GPU valida essa semântica, não um meta-gradiente completo. Full BPTT seria mudança de algoritmo, separada da otimização de execução.

### P2 — Contexto explícito, concorrência e HIP Graphs

Já existe contexto de execução com stream, eventos e workspaces. Porém, vários scratch buffers de Mamba/MoE/Trainer são `thread_local`, não pertencem explicitamente a `(device, contexto, stream)`. Os fences atuais limitam conflitos entre escopos; isso não comprova uma race existente, mas restringe o desenho de concorrência.

Mover o ownership dos workspaces para o contexto, explicitar contratos de dtype/layout/alinhamento e lifetime de views. Testar modelos intercalados, streams distintas, devices distintos e atualização/validação concorrentes. `Parameter::zero_grad` e produtores de metadados precisam obedecer ao mesmo contrato de stream.

Graphs de decode não significam graphs de treino. Capturar primeiro subgrafos estáveis, com buffers pré-alocados, shapes em buckets, atualização de metadados e controle de commit definido. D2H escalares no meio do grafo e mudanças de ponteiros precisam ser eliminados ou mantidos fora da captura. [Documentação HIP Graphs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html).

Batching e acúmulo devem ser medidos por tokens úteis supervisionados, e não padding. Aumentar batch modifica a trajetória de otimização; comparar qualidade por token e por tempo, além de throughput. O piloto Python atual usa batch 1.

CHRASS e Slender continuam explicitamente fora do caminho GPU de treinamento do script; não foram certificados pelas seis integrações. JEV permanece adiado.

## 6. Inteligência por step e pesquisa recente

Ganhar tokens/s não demonstra ganhar raciocínio por step. Um scorecard útil precisa manter corpus/tokenizer e orçamento fixos e medir loss held-out por família, acurácia de tarefas verificáveis, recuperação de contexto, repetição/boilerplate, seguimento de formato e retenção entre etapas. Comparar também wall-time e energia quando houver telemetria confiável. Loss em 120 janelas consecutivas do treino não é esse scorecard.

Ablacionar primeiro Mamba puro versus híbrido com attention esparsa em profundidade; depois, separadamente, MoE, KAN e TTT. Mesmo número de parâmetros não assegura mesmo custo ativo. Registrar parâmetros ativos, tokens supervisionados, seed e distribuição de dados. Não habilitar todas as camadas extras simultaneamente e atribuir a melhoria ao conjunto sem controle.

Pesquisa de 2026: Mamba-3 propõe recorrência exponencial-trapezoidal, transições complexas e MIMO, além de mudanças de normalização. Os autores apontam custo maior de treino no MIMO mesmo quando o decode fica competitivo. Minha avaliação: é candidato a experimento de expressividade, não substituição automática do Mamba atual nem justificativa para retirar sua convolução isoladamente. O benchmark publicado em H100 não prova ganho nesta Radeon. [Mamba-3, autores](https://tridao.me/blog/2026/mamba3-part1/).

SSDi8, publicado em agosto de 2026, é quantização pós-treino de SSD, com caminho INT8 e avaliação também em Orin NX. É relevante para implantação futura; não prova que gradientes, estados e treinamento nativo possam ser convertidos para INT8 sem perda. [SSDi8](https://arxiv.org/abs/2608.21952).

Persistência global, speculative decoding, KV 2-bit e preditores de skip permanecem trilhas separadas. As duas primeiras não resolvem diretamente o custo do backward; KV compacto não reduz o histórico Mamba de treino; skip/quantização agressiva podem mudar a função aprendida. Um monokernel com barreira global não pode assumir que todos os blocos serão simultaneamente residentes: há risco de deadlock e pressão de registradores/LDS.

Para afirmar superioridade ao estado da arte faltam baselines externos equivalentes no mesmo hardware, qualidade held-out, custo total de treino, estabilidade prolongada e ensaios repetidos. Hoje a afirmação defensável é: motor nativo mensurável, com otimizações úteis e gargalos arquiteturais concretos ainda abertos.

## 7. Medição industrial e critérios de promoção

A instrumentação atual NÃO mediu ocupação, spills, cache hit, tráfego DRAM real, clocks/potência estáveis ou todos os dispatches. Portanto não há evidência para prometer TCC_HIT >85%, uma banda efetiva específica ou um launch por sequência.

`rocprofv3`, `rocprof`, `amd-smi` e `rocm-smi` não foram encontrados no PATH desta sessão Windows. A documentação atual da AMD distingue trace de instruções e contadores: RDNA não possui todos os recursos ATT/perfmon das Instinct; disponibilidade deve ser consultada por ferramenta/arquitetura e estado de potência apropriado. Isso não autoriza reutilizar nomes de contadores CDNA cegamente. [ROCprofiler-SDK, capacidades](https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/what-is-rocprofiler-sdk.html).

Antes de promover redesenhos grandes:

1. Testes locais de shapes, dtype/layout, overflow, strides, tail e máscaras.
2. Gradcheck pequeno, referência CPU e paridade GPU de entrada e parâmetros, incluindo caminhos realmente ativos.
3. Integração com pesos amarrados, acúmulo, múltiplas sessões e retomada.
4. Injeção de NaN/Inf antes do commit para garantir ausência de update parcial.
5. Benchmark A/B em processos novos, mais de uma ordem de execução, mesmas janelas, hash do binário, warmup separado e sem instrumentação perturbativa.
6. Piloto held-out: critérios explícitos de perda/qualidade, não apenas estabilidade numérica.

Uma lacuna de observabilidade adicional: o resumo Python `model_config_dict` ainda não inclui todos os campos híbridos nativos. O novo ensaio de integração registra composição, gates, experts e RoPE explicitamente. Recomenda-se uma única exportação canônica nativa da configuração; não foi constatado aqui um defeito equivalente na serialização nativa do checkpoint.

## 8. Validação concluída ao final das alterações

- HIP: 9/9 testes direcionados — gradcheck, training invariants, contrato do profiler, BitLinear, Jamba/MoE, decode runtime, TTT, continuação de checkpoint e precisão mista.
- CPU: 6/6 — BitLinear, MoE training, gradcheck, training invariants, contrato do profiler e Jamba.
- Seis topologias executadas novamente com o binário final, todas finitas.
- A/B de 120 steps por binário, 20 de warmup; nenhuma diferença nas losses retornadas.
- Logs finais: `validation-hip-copy-final.log` e `validation-cpu-copy-final.log`.

Esses são testes direcionados, não uma nova execução de toda a suíte do repositório nem certificado de qualidade de todos os caminhos. Os logs de falha anteriores foram preservados para rastreabilidade.

## 9. Reprodução e estado do treinamento

Na raiz do workspace, usando o Python configurado com as dependências do projeto:

```powershell
$env:NSOS_HIP_ROOT = 'C:/TheRock/build'
$env:PATH = 'C:/TheRock/build/bin;' + $env:PATH
$env:PYTHONUTF8 = '1'
python OXN/nsos/scripts/bench_training_engineering.py `
  --build-dir OXN/nsos/build-gm-hip `
  --shard OXN/nsos/artifacts/ptbr_verified_pilot_20260928/packs/base_train/base-train-00000.u16 `
  --tokenizer OXN/nsos/artifacts/ptbr_verified_pilot_20260928/tokenizer/tokenizer_16384.ox3 `
  --output OXN/nsos/artifacts/gpu_engineering_20260929/new-measurement.json `
  --profile chunked-bf16 --steps 120 --warmup 20
```

Executar um perfil GPU de cada vez, em processo novo. O script rejeita sobrescrita do resultado. `--native-timing` e `--layer-timing` são diagnósticos, não modos oficiais de comparação de throughput. Para integração, usar `audit_training_integrations.py --build-dir ... --variant moe --output <arquivo-novo.json>`; repetir separadamente para as demais variantes.

Não há treinamento de produção ativo. O job da receita completa falhou antes de iniciar o treino: o SFT estruturado tem somente 18.275 tokens de exemplos completos admissíveis, abaixo da cota de 50.000; parte dos candidatos não cabe na sequência 512. Os ensaios deste relatório usam um shard base válido já preparado e não corrigem essa insuficiência. Próxima ação de dados: admissão/reabastecimento sensível ao tokenizer e ao tamanho real do exemplo completo, sem cortar respostas nem afrouxar silenciosamente os filtros.

## 10. Arquivos desta etapa

- [Profiler nativo](C:/Users/vitor/OneDrive/Desktop/OGrandeOxta/reimagined/OXN/nsos/scripts/bench_training_engineering.py).
- [Ensaio de integrações](C:/Users/vitor/OneDrive/Desktop/OGrandeOxta/reimagined/OXN/nsos/scripts/audit_training_integrations.py).
- [Contrato do profiler](C:/Users/vitor/OneDrive/Desktop/OGrandeOxta/reimagined/OXN/nsos/tests/test_training_engineering_probe.py).
- [Correções MoE/router](C:/Users/vitor/OneDrive/Desktop/OGrandeOxta/reimagined/OXN/nsos/src/jamba.cpp:8430).
- [Cópia BitLinear removida](C:/Users/vitor/OneDrive/Desktop/OGrandeOxta/reimagined/OXN/nsos/src/bitlinear.cpp:1094).
- [Regressão MoE](C:/Users/vitor/OneDrive/Desktop/OGrandeOxta/reimagined/OXN/nsos/tests/gpu/test_gpu_parity_jamba.cpp).
- [Regressão BitLinear](C:/Users/vitor/OneDrive/Desktop/OGrandeOxta/reimagined/OXN/nsos/tests/gpu/test_gpu_parity_bitlinear.cpp).
- Registro do teste Python em `OXN/nsos/CMakeLists.txt`.

As demais alterações já presentes no worktree foram preservadas. Este relatório não atribui a esta etapa todo o diff acumulado do projeto.
