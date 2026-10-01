# Auditoria de gaps, arquitetura híbrida e desempenho AMD/HIP

Data: 2026-07-28  
Escopo: NSOS, com foco no perfil de produto Mamba + Attention + OxtaMem.  
Estado da campanha: encerrada; nenhum novo treino, build ou CTest foi iniciado nesta
etapa.

## Parecer executivo

O NSOS já tem um caminho HIP funcional e evidência real na RX 7600, mas ainda não
é correto descrevê-lo como totalmente robusto, auditável ou otimizado.

Os achados de maior impacto são:

1. O "híbrido" atual não preserva o Mamba e adiciona Attention. O agendador troca
   blocos Mamba inteiros por blocos Attention + FFN. Portanto, o perfil híbrido
   reduz a profundidade recorrente em vez de combinar dois mecanismos
   complementares.
2. Mamba e Attention não operam sob o mesmo contrato matemático. As projeções do
   Mamba faithful usam `BitLinear` em modo linear exato; as projeções de Attention
   e do FFN mantêm RMSNorm interno e magnitude aprendível.
3. Um único vetor LayerScale controla ao mesmo tempo a contribuição do mixer de
   Attention e a contribuição do FFN. Isso mistura duas hipóteses diferentes em
   um só peso e impede atribuição causal.
4. O comentário que apresenta a última camada Mamba como correção validada já
   não representa a evidência mais recente: no protocolo de 1.500 passos, a
   topologia com a guarda terminou em 50,3%, enquanto o Mamba-only terminou em
   99,9%.
5. A instrumentação atual não audita cada peso, gradiente e atualização. O
   `LayerAudit` observa ativações agregadas por bloco, e seu caminho de GPU tem
   problemas de contrato de memória e de medição assíncrona.
6. A integração HIP está lenta principalmente por decisões do trainer e dos
   kernels, não pelo fato de os arquivos terem extensão `.cu`: loss por amostra
   com sincronização D2H, concatenação de bias de convolução via CPU,
   `cudaMalloc/cudaFree` de IDs do embedding a cada chamada, projeções e
   convoluções não fundidas, scan temporal sequencial e uso intenso de atomics.

A orientação é não iniciar outra campanha longa antes de fechar observabilidade,
semântica do híbrido e os três gargalos P0 de sincronização/alocação.

## Evidência usada

- Implementação de blocos e agendamento: `src/jamba.cpp`.
- Mamba-2 faithful e caminho GPU: `src/mamba2.cpp` e
  `src/cuda/mamba_kernels.cu`.
- Tensores, reduções e cross-entropy: `src/tensor.cpp`.
- Trainer e AdamW fundido: `src/trainer.cpp`.
- Auditoria: `include/layer_audit.h` e `src/layer_audit.cpp`.
- Gates: `CMakeLists.txt`, `docs/PRODUCT.md`, `docs/TESTING.md` e
  `docs/NSOS_VALIDATION_STATUS.md`.
- Resultados já encerrados:
  - `artifacts/oxta_contabil_amd/benchmark/rx7600_product_architecture_ablation400.json`
  - `artifacts/oxta_contabil_amd/benchmark/rx7600_hybrid_last_guard_seed11_1500.json`
  - `artifacts/oxta_contabil_amd/benchmark/rx7600_mamba_oxtamem_seed11_1500.json`
- Inventário completo de parâmetros:
  `artifacts/oxta_contabil_amd/audit/product_parameter_manifest.json`.

O inventário tem SHA-256
`1FF16761C2FDEEF020786416516B21AAD22E933BD29D058886EE3115385C91A6`.
Ele contém nome absoluto, nome-base, índice no registro, shape, elementos, bytes
FP32, camada, componente, papel e classificação treinável/fixa de cada entrada.

## Topologia real

### Mamba-only

```text
embedding
  -> Mamba faithful
  -> Mamba faithful
  -> Mamba faithful
  -> Mamba faithful
  -> norm_f
  -> head ligado ao embedding
```

Cada bloco faithful usa RMSNorm aprendível + núcleo Mamba. O bloco não recebe o
FFN denso histórico.

### Híbrido com Attention terminal

```text
embedding
  -> Mamba faithful
  -> Attention + FFN denso
  -> Mamba faithful
  -> Attention + FFN denso
  -> norm_f
  -> head ligado ao embedding
```

### Híbrido com a guarda da última camada

```text
embedding
  -> Mamba faithful
  -> Attention + FFN denso
  -> Mamba faithful
  -> Mamba faithful
  -> norm_f
  -> head ligado ao embedding
```

Logo, "Mamba + Attention" hoje significa substituição heterogênea de camadas. Não
há um bloco no qual o mesmo estado receba Mamba e Attention com gates
independentes.

## Mapa de pesos

| Perfil | Topologia | Tensores no registro | Tensores treináveis | Elementos no registro | Elementos treináveis | Elementos legados fixos |
|---|---|---:|---:|---:|---:|---:|
| Mamba-only | M-M-M-M | 104 | 54 | 480.816 | 473.392 | 7.424 |
| Híbrido terminal | M-A-M-A | 108 | 62 | 613.144 | 605.208 | 7.936 |
| Híbrido com guarda | M-A-M-M | 106 | 58 | 546.980 | 539.300 | 7.680 |

Fingerprints do manifesto:

- Mamba-only:
  `9f0edc83e2d9d5346d83706984ba194ed14da2c2dce056c437a40ae0a75cb6ad`
- Híbrido terminal:
  `d4fd869fcac06a143808faa516e6c83c66c29551f1fff1afdeb49ff08c751d9d`
- Híbrido com guarda:
  `6935df8c7b9e340af1e145b6e951ac6362a317ec79b7d6791362b3614aa3952a`

### Composição de um bloco

Um bloco Mamba do perfil auditado contém:

- RMSNorm aprendível: 128 elementos.
- Cinco projeções independentes `x`, `z`, `B`, `C`, `dt`.
- Uma projeção de saída.
- Convolução depthwise, bias, norma interna, `A` e `D`.
- 119.180 elementos no registro, dos quais 117.388 treináveis; com a norma do
  wrapper, 117.516 treináveis.

Um bloco Attention do perfil auditado contém:

- Attention/LayerScale/projeções: 51.840 elementos no registro, 51.072
  treináveis.
- `ffn_gate_up` 128 -> 512: 66.816 no registro, 66.560 treináveis.
- `ffn_down` 512 -> 128: 66.816 no registro, 65.792 treináveis.
- Total: 185.472 no registro e 183.424 treináveis.

Trocar um Mamba por Attention + FFN adiciona 65.908 elementos treináveis, mas
remove uma recorrência Mamba completa. A comparação não é pareada por mecanismo,
profundidade ou orçamento.

### Contratos especiais do registro

- `flat_alpha` e `flat_beta` continuam no registro por compatibilidade de
  checkpoint, mas o trainer os filtra. O benchmark somou
  `model.parameters()` e, portanto, publicou elementos de registro como
  "parameters", não parâmetros treináveis.
- O head compartilha o peso com o embedding. Seu peso não aparece duas vezes; os
  dois itens restantes no grupo `value_head` são buffers legados fixos.
- O nome atual do embedding é `embedding.embedding.weight`, embora seu nome-base
  seja `embedding.weight`. É um prefixo duplicado. Alterá-lo sem migração
  quebraria checkpoints; deve existir uma versão explícita do schema antes da
  correção.

## Por que o híbrido não ajudou

### 1. Ele remove Mamba em vez de adicionar Attention

Na construção de `JambaBlock`, Attention e Mamba são mutuamente exclusivos. No
caso M-A-M-A, somente duas das quatro camadas mantêm estado recorrente. A
hipótese "Attention complementa Mamba" não foi testada por essa topologia.

### 2. Attention carrega um FFN que o Mamba faithful não carrega

O Mamba faithful é `core-only`. Um bloco Attention cria, além do mixer, um FFN
128 -> 512 -> 128 com squared-ReLU. Portanto, qualquer diferença atribuída a
Attention pode vir:

- do mixer de Attention;
- do FFN;
- da interação entre ambos;
- da remoção do Mamba;
- das normalizações adicionais.

Não existe ablação que isole esses fatores.

### 3. Projeções com contratos diferentes

As seis projeções do Mamba faithful chamam `set_exact_linear_mode(true)`. Isso
desliga o RMS interno do `BitLinear` e remove o parâmetro `magnitude`, fazendo a
projeção se comportar como `nn.Linear` em float.

As projeções `q`, `kv` e `out` de Attention não ativam esse modo. Os dois
`BitLinear` do FFN também não. Assim:

- o wrapper já aplica RMSNorm;
- `q`, `kv`, `out`, `ffn_gate_up` e `ffn_down` aplicam RMS adicional;
- cada projeção possui uma magnitude aprendível que não existe nas projeções
  Mamba faithful.

Isso não é uma comparação simétrica nem uma implementação fiel de Attention com
lineares comuns.

### 4. Um LayerScale para duas funções

`attn.layerscale` é um vetor por canal iniciado em 0,01. O mesmo vetor escala:

- a saída do mixer de Attention;
- a saída do FFN.

Seu gradiente soma as duas contribuições. Não há como saber se o mixer deveria
crescer enquanto o FFN permanece pequeno, ou o contrário. Também não há
telemetria da trajetória de gamma ou da norma de cada contribuição.

Há drift até nos comentários: o header ainda descreve inicialização 0,1, enquanto
o código usa 0,01.

### 5. Normalização assimétrica

O Mamba faithful recebe uma norma aprendível no wrapper. Attention recebe a
RMSNorm fixa do wrapper e depois RMSs internos dos `BitLinear`. O segundo RMS do
wrapper antes do FFN também não possui peso aprendível.

### 6. A guarda da última camada não é uma correção comprovada

O comentário do construtor afirma que Attention terminal trava e que Mamba
terminal aprende. A nova execução controlada contradiz a generalização dessa
regra:

- M-A-M-M com guarda, seed 11, 1.500 passos: 50,3%.
- M-M-M-M, seed 11, 1.500 passos: 99,9%.

A guarda é uma hipótese experimental codificada como padrão global via variável
de ambiente. Ela deve ser rebaixada a opção de arquitetura versionada até passar
uma matriz de seeds e tarefas.

### 7. O benchmark não exige a capacidade que Attention deveria adicionar

bAbI QA1 mede atualização e recuperação simples de localização. É favorável a
estado recorrente e não comprova vantagens esperadas de Attention em associação
multi-query, cópia, recuperação por conteúdo ou dependências longas. Uma única
seed com reduções não determinísticas também não separa arquitetura de variância.

## Arquitetura que realmente testa complementaridade

Uma variante mínima deve manter o bloco Mamba e sobrepor Attention, com pesos
separados:

```text
h0 = x
h1 = h0 + gamma_mamba * Mamba(RMS_mamba(h0))
h2 = h1 + gamma_attn  * Attention(RMS_attn(h1))
y  = h2 + gamma_ffn   * FFN(RMS_ffn(h2))
```

Requisitos:

- `gamma_mamba`, `gamma_attn` e `gamma_ffn` independentes;
- normas aprendíveis e contratos de projeção explícitos;
- opção Attention-core sem FFN para ablação;
- topologia e versão persistidas no checkpoint, não escondidas em ambiente;
- baseline pareada por número de blocos Mamba, parâmetros treináveis e FLOPs;
- log por passo das normas de cada ramo e do residual.

Uma alternativa paralela também pode ser avaliada:

```text
y = x + gamma_mamba * Mamba(RMS(x))
      + gamma_attn  * Attention(RMS(x))
```

Ela testa complementaridade diretamente, mas custa mais memória e compute. Deve
ser uma arquitetura separada, não uma mudança silenciosa no significado do
perfil atual.

## Gargalos AMD/HIP

### Evidência de desempenho controlado

No mesmo artefato de 400 passos, batch 32 e FP32:

| Braço | Exemplos/s | Acurácia |
|---|---:|---:|
| Attention-only | 273,16 | 48,9% |
| Híbrido | 46,83 | 50,5% |
| Mamba-only | 30,63 | 51,4% |

Nesse protocolo, Mamba-only treinou aproximadamente 8,9 vezes mais devagar que
Attention-only. O resultado de 1.500 passos melhorou para 53,17 exemplos/s, mas
continua sem um braço Attention de 1.500 passos medido na mesma execução.

### P0 — loss por amostra e sincronização D2H

`train_supervised_batch_impl` percorre cada item do batch e chama
`cross_entropy_weighted` separadamente. A função copia o escalar de loss da GPU
para o host em cada chamada.

No batch 32 do benchmark:

- 32 launches de cross-entropy por passo;
- uploads repetidos de target e row weight;
- 32 leituras D2H de escalar, cada uma criando uma barreira de dependência.

Correção necessária: loss batelada sobre `[batch, rows, vocab]`, uma máscara de
respostas no device e, no máximo, uma leitura de loss por passo de log. O
backward não precisa trazer a loss ao host.

### P0 — bias da convolução volta à CPU

No backward faithful, as três reduções de bias (`x`, `B`, `C`) são copiadas
individualmente para CPU, concatenadas e reenviadas para GPU. Com quatro blocos
Mamba, são doze cópias D2H sincronizantes por passo apenas para esse bias.

Correção necessária: escrever as três reduções diretamente em offsets de um
único tensor GPU ou usar um kernel de concatenação no device.

### P0 — alocação de IDs do embedding em toda chamada

`Embedding` cria um `CudaIntBuffer` com `cudaMalloc` e o destrói com `cudaFree`
em forward/backward. Essas APIs podem sincronizar o device.

Correção necessária: buffer persistente por modelo/thread, com crescimento por
capacidade, ou IDs já mantidos no device pelo trainer.

### P1 — Mamba fragmentado

O Mamba faithful faz:

- cinco GEMMs independentes para `x`, `z`, `B`, `C`, `dt`;
- três convoluções forward e três backward;
- SiLU/gates/RMS/multiplicações em kernels elementares separados;
- uma projeção de saída.

O desenho de referência usa uma projeção de entrada combinada. Um buffer
empacotado com views por offset preserva os nomes/checkpoints e reduz launches.
Da mesma forma, `x/B/C` podem compartilhar uma convolução sobre o tensor
concatenado.

### P1 — scan e atomics

Cada thread do scan faithful percorre a sequência inteira serialmente e mantém
um vetor de 64 estados. No backward, `gD`, `gB`, `gC`, `gDt` e `gA` usam
`atomicAdd`, com contenção especialmente em parâmetros compartilhados por
head/grupo.

É necessário:

- profiler por kernel e ocupação antes de alterar a matemática;
- backward em duas fases, com parciais por bloco e redução determinística;
- scan paralelo/chunked correto com carry entre chunks;
- variante especializada para `N=64`, `P=64` e `K=4` da RX 7600.

O arquivo já removeu uma tentativa chunked anterior porque o carry entre chunks
estava incorreto. Uma nova versão precisa de paridade e gradcheck antes de
promoção.

### P1 — histórico de estado

Para batch 32, sequência 83, H=4, P=64 e N=64, cada camada salva:

```text
32 * 83 * 4 * 64 * 64 = 43.515.904 floats ≈ 166 MiB
```

Quatro blocos Mamba retêm aproximadamente 664 MiB apenas em
`state_history`, sem contar outras ativações, gradientes e workspaces.
Checkpointing troca memória por recomputação, não resolve throughput.

Devem ser avaliados history compactado/chunked e backward recomputado por
janelas, sempre com comparação numérica.

### P1 — churn de tensores de gradiente

`Parameter::add_grad` faz `grad = grad.add(incoming)` na GPU. Isso cria um novo
tensor para cada acumulação. O pool reduz o custo de `cudaMalloc`, mas não elimina
kernel, troca de buffer e pressão de memória.

É necessário um `add_inplace` GPU e acumuladores estáveis por parâmetro.

### P1 — clip global ainda sincroniza

O AdamW multi-tensor já é uma melhoria real: dois kernels cobrem norma e update.
Ainda há uma leitura D2H de quatro bytes por passo para decidir o coeficiente de
clip no host. A decisão e o coeficiente podem permanecer no device.

### P2 — FP32 e tamanho pequeno

O benchmark forçou FP32 embora a RX 7600 reporte FP16/BF16. BF16 só deve ser
ativado depois de:

- paridade de forward/backward;
- tolerância de loss e acurácia;
- loss scaling e detecção de overflow;
- checkpoint/resume exato;
- comparação de throughput e memória.

O modelo `d_model=128` também gera GEMMs pequenos, propensos a launch overhead.
Otimizar apenas batch não substitui fusão; deve-se medir batch/seq por faixa.

## Gaps de auditabilidade

### LayerAudit não é um auditor de pesos

Hoje ele registra:

- estatística de input/output por bloco;
- norma do gradiente de entrada/saída;
- NaN/Inf;
- loss e norma global;
- roteamento MoE.

Ele não registra por parâmetro:

- hash inicial/final;
- norma de peso, gradiente e update;
- `update_norm / weight_norm`;
- cosine entre peso, gradiente e update;
- mínimo/máximo e contagem NaN/Inf;
- distribuição ternária `{-1,0,+1}` e saturação STE;
- trajetória de `A`, `D`, `dt_bias`, normas e LayerScale;
- contribuição residual de Mamba, Attention e FFN;
- bytes de optimizer e ativação.

### LayerAudit conflita com memória GPU nativa

`summarize_tensor` chama `tensor.data()` antes de testar o device. Em execução
GPU estrita, tensores `cudaMalloc` não permitem acesso host por `data()`. O teste
oficial de LayerAudit é CPU e não fecha esse contrato.

### Latência de camada não representa tempo de GPU

A latência usa `steady_clock` ao redor de launches assíncronos. Sem eventos HIP
no mesmo stream, o número mede principalmente enfileiramento no host. Quando a
auditoria copia tensores para CPU, a sincronização ocorre depois do timestamp.

É necessário um modo GPU com eventos, sem D2H por camada, e redução de
estatísticas no device.

### "Strict GPU" não prova residência total

O modo estrito impede fallbacks que passam por `warn_host_fallback_once`, mas não
intercepta cópias `.cpu()` explícitas, como as reduções do bias da convolução.
O manifesto deve contar:

- bytes H2D/D2H/D2D;
- número de sincronizações;
- fallback por operação e motivo;
- launches por componente;
- pico de memória e hit rate do pool.

### O artefato de benchmark é insuficiente para reauditoria

O JSON atual não persiste:

- commit, branch e hash do diff sujo;
- compilador, flags, SDK/driver e DLLs carregadas;
- todas as variáveis `NSOS_*` efetivas;
- topologia efetiva após guardas;
- manifesto de parâmetros antes/depois;
- checkpoint final;
- estados AdamW/scheduler/RNG;
- métricas por peso/camada.

Os cinco arquivos centrais consultam dezenas de knobs `NSOS_*`; o resultado pode
mudar sem aparecer em `config`.

Também há um erro sem efeito imediato no script: ele atribui `64` ao campo
booleano `mamba_state_expansion`, provavelmente pretendendo configurar
`mamba_d_state`. O default de `mamba_d_state` já é 64, por isso o teste não mudou,
mas o contrato está incorreto e o campo nem aparece no JSON.

### Determinismo e estatística

O benchmark chama `set_deterministic_reductions(False)` e usa somente uma seed
nos braços principais. O próprio resultado anterior de Mamba variou bastante
entre campanhas. Uma conclusão arquitetural exige:

- lane determinística curta;
- pelo menos 5 seeds para decisão;
- média, desvio, intervalo de confiança e distribuição;
- comparação pareada usando as mesmas amostras por passo;
- teste de significância e tamanho de efeito.

## Gaps de testes

| Gap | Evidência | Prioridade |
|---|---|---|
| Sem gradcheck do LayerScale híbrido | `test_gradcheck` cobre input de Attention, não o bloco híbrido nem gamma separado | P0 |
| Sem paridade GPU da topologia híbrida de produto | `test_gpu_parity_jamba` usa `attention_period=64` em quatro camadas; apesar do comentário, não instancia Attention | P0 |
| Sem LayerAudit GPU nativo | O smoke oficial usa a lane CPU e `summarize_tensor` acessa `data()` cedo demais | P0 |
| Sem teste de contribuição por ramo | Não há assert para norma/cosine de Mamba, Attention e FFN | P0 |
| Sem round-trip do checkpoint com topologia/guardas efetivas | Variáveis de ambiente podem reconstruir outra arquitetura | P0 |
| Sem teste batelado da loss equivalente ao loop por amostra | A implementação atual serializa o batch | P0 |
| Sem teste de ausência de D2H em treino estrito | `strict_gpu_execution` não conta cópias explícitas | P0 |
| Sem benchmark de kernel/launch persistido | `NSOS_TRAIN_TIMING` é diagnóstico textual e sincronizante | P1 |
| Sem testes de OOM, cancelamento e recuperação de checkpoint no meio do passo | Não existe matriz de fault injection de treino | P1 |
| Sem teste de corrupção/truncamento de todos os artefatos | Packs têm cobertura, mas a cadeia completa não é exercitada | P1 |
| Sem stress concorrente no gate | `test_thread_safety.cpp` existe, mas não está registrado no CMake atual | P1 |
| Sem matriz multi-GPU/driver | Apenas RX 7600 `gfx1102` foi validada em runtime | P1 |
| Sem NVIDIA real | A árvore CUDA está mantida, mas falta execução em hardware | P1 |

Há drift documental:

- `docs/PRODUCT.md` declara `test_thread_safety`, `test_matmul`,
  `test_moe_router` e outros como gates/wired;
- `CMakeLists.txt` não registra vários desses arquivos;
- `docs/TESTING.md` ainda lista `test_moe_training` como candidato, embora ele
  esteja registrado;
- `docs/TESTING.md` registra 66 testes AMD, enquanto a lane HIP mais recente
  observada no workspace possui 67;
- `docs/GPU_OPTIMIZATION_ANALYSIS.md` ainda descreve uma GTX 1050 Ti e vários
  fallbacks que já foram substituídos.

O inventário autoritativo deve ser gerado do CMake configurado e anexado a cada
artefato, em vez de mantido manualmente em três documentos.

## Plano priorizado

### P0 — antes de novo treino longo

1. Corrigir `LayerAudit` para GPU nativa e adicionar `ParameterAudit`.
2. Persistir commit/diff, ambiente efetivo, topologia, manifesto e checkpoint.
3. Separar `gamma_attention` e `gamma_ffn`; adicionar gradcheck.
4. Tornar explícito o contrato das projeções de Attention
   (`exact_linear_mode` ou divergência documentada e ablada).
5. Criar variante híbrida aditiva/overlay sem remover Mamba.
6. Fundir a loss batelada e eliminar as 32 leituras D2H.
7. Manter o bias da convolução no device.
8. Reutilizar o buffer de IDs do embedding.
9. Adicionar paridade CPU/GPU do perfil M-A-M-M e do novo overlay.
10. Corrigir o manifesto de parâmetros do benchmark: registro, treináveis,
    buffers, bytes e FLOPs separados.

### P1 — desempenho e robustez

1. Empacotar `x/z/B/C/dt` em uma projeção e expor views compatíveis com o
   checkpoint.
2. Unificar as três convoluções.
3. Implementar `add_grad` in-place.
4. Mover clip e decisão de estabilidade para o device.
5. Adicionar eventos HIP e contadores de launch/cópia/fallback.
6. Implementar fault injection: OOM, NaN, interrupção, arquivo truncado e resume.
7. Promover os testes órfãos relevantes ao gate.
8. Fixar scorecard multi-seed e tarefas que realmente exijam Attention.

### P2 — depois da paridade

1. BF16/FP16 por operação com master weights FP32.
2. Scan chunked/paralelo correto.
3. Especializações `gfx1102` para shapes de produto.
4. Grafo de treino/captura onde shapes são estáveis.
5. Matriz adicional AMD e validação NVIDIA real.

## Gates propostos de aceitação

### Arquitetura

- O perfil híbrido preserva a mesma quantidade de blocos Mamba do baseline.
- Attention-core, FFN e gates são abláveis separadamente.
- Todos os pesos e buffers têm nome estável, papel e versão de schema.

### Correção

- Gradcheck de todos os novos gates e projeções.
- Paridade CPU/HIP forward, backward e um passo AdamW.
- Round-trip de checkpoint com outputs, gradientes e próximo passo idênticos.
- Lane determinística bitwise e lane rápida tolerante separadas.

### Auditoria

- Manifestos pre/post com hashes.
- Métrica por peso e por ramo.
- Configuração efetiva e ambiente completos.
- Dataset, amostragem e RNG reproduzíveis.
- Contadores H2D/D2H e fallbacks iguais a zero no caminho declarado
  GPU-residente, salvo exceções formalmente permitidas.

### Desempenho

- Nenhuma leitura D2H por amostra.
- Nenhuma alocação `cudaMalloc/cudaFree` no steady state do passo.
- Tempo por forward/loss/backward/optimizer medido com eventos HIP.
- Throughput, pico de VRAM e launches/step versionados como artefatos.
- Regra de regressão definida antes da otimização.

### Produto

- Cinco ou mais seeds nas decisões arquiteturais.
- Suite com recorrência, MQAR, cópia, recuperação por conteúdo e contexto longo.
- Comparação pareada por orçamento e protocolo.
- OxtaMem avaliado com latência, capacidade, persistência, concorrência,
  corrupção e consultas que não sejam apenas chaves estruturadas.

## Decisão recomendada

Congelar novas campanhas longas. A próxima etapa deve ser a camada P0 de
observabilidade e correção estrutural. Só então executar probes curtos e
pareados:

1. M-M-M-M atual.
2. M-A-M-M atual, mas com Attention-core e FFN separados.
3. M-(M+Attention)-M-M com gates separados.
4. A mesma matriz com projeções exatas e não exatas explicitamente ablada.

Sem essa sequência, um novo número de acurácia continuará misturando arquitetura,
normalização, FFN, LayerScale, seed e custo de GPU, sem permitir explicar por que
melhorou ou piorou.
