# Relatório de benchmark — Jamba oficial vs NSOS vs OxtaMem

**Data da execução:** 22 de julho de 2026  
**Status:** executado e validado em GPU real  
**Repositório:** [vitorGgC569/reimagined](https://github.com/vitorGgC569/reimagined)  
**Branch:** `nsos-gpu-phases12`  
**Código carregado no runtime:** `ec97b3e`  
**Notebook reproduzível:** [bench_jamba_attention_vs_nsos_oxtamem_t4.ipynb](../../colab/bench_jamba_attention_vs_nsos_oxtamem_t4.ipynb)  
**Commit que registrou o braço MoE:** [8eda86f](https://github.com/vitorGgC569/reimagined/commit/8eda86f730e68c956b62097a3f93cbe2d235b6ff)  
**Abrir no Colab:** [executar benchmark](https://colab.research.google.com/github/vitorGgC569/reimagined/blob/nsos-gpu-phases12/colab/bench_jamba_attention_vs_nsos_oxtamem_t4.ipynb)

## 1. Resumo executivo

O benchmark comparou quatro braços:

1. Jamba oficial denso, usando a implementação `transformers.JambaForCausalLM`;
2. Jamba oficial com o MoE nativo do Jamba;
3. NSOS híbrido Mamba+Attention, sem MoE e sem OxtaMem;
4. o mesmo NSOS, sem MoE, usando OxtaMem como memória externa.

No orçamento controlado de 2.500 atualizações, o NSOS sem MoE aprendeu a tarefa MQAR muito melhor que os dois braços oficiais. Dar ao rival o MoE completo — 16 especialistas, top-2 e perda auxiliar de roteamento — aumentou sua capacidade de 1,35 milhão para 7,26 milhões de parâmetros, mas não melhorou o recall sintético. Quando os fatos excederam a janela disponível, o OxtaMem alcançou 1,000 de acurácia e 1,000 de retrieval@1 em todos os tamanhos testados.

A conclusão sustentada pelos dados é:

> O NSOS é um protótipo de pesquisa muito forte em aprendizagem rápida de recall associativo, e o sistema NSOS+OxtaMem foi excepcional neste teste de memória externa. O experimento não demonstra superioridade geral sobre modelos de linguagem de produção.

## 2. Ambiente validado

| Item | Valor |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Arquitetura CUDA compilada | `sm_75` |
| CUDA toolkit | 12.8.93 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.13.1 |
| Python | 3.12.13 |
| Build | CMake + Ninja, Release, CUDA habilitado |
| Extensão NSOS | `nsos_ext` nativa |
| OxtaMem | wheel Rust nativa construída com Maturin |
| Referência oficial | `JambaForCausalLM`, Hugging Face/AI21 |
| Kernels Mamba oficiais fundidos | indisponíveis; referência executada no fallback PyTorch |

O Jamba é documentado oficialmente como uma arquitetura híbrida Transformer–Mamba com MoE: [documentação do Transformers](https://huggingface.co/docs/transformers/model_doc/jamba) e [paper da AI21](https://arxiv.org/abs/2403.19887).

## 3. Pré-condições verificadas

Antes do benchmark:

- `nsos_ext` foi compilado e importado na T4;
- o wheel nativo do OxtaMem foi compilado e importado;
- `cargo test --release --locked`: 19 testes aprovados, 0 falhas e 2 ignorados;
- gate de alinhamento, busca e persistência do OxtaMem: 10/10;
- leitura após reabertura do banco foi validada;
- retrieval vetorial top-1 foi validado;
- todos os braços produziram tensores com a forma esperada;
- nenhum resultado usou o fallback NumPy como prova oficial do OxtaMem.

## 4. Configuração comum dos modelos

| Parâmetro | Valor |
|---|---:|
| Vocabulário | 1.266 |
| Dimensão do modelo | 128 |
| Camadas | 4 |
| Agenda de camadas | Mamba, Attention, Mamba, Attention |
| Cabeças de atenção | 4 |
| Cabeças KV | 2 |
| Estado Mamba | 64 |
| Convolução Mamba | 4 |
| Expansão Mamba | 2 |
| Dimensão intermediária | 512 |
| Contexto máximo configurado | 1.024 |
| Dropout | 0 |
| Embedding/head compartilhados | sim |
| Cache de geração | desativado no treino |
| Precisão | FP32 |
| Semente de inicialização | teste histórico usou 0; esse valor é sentinela não determinística no NSOS |

A agenda 1:1 foi escolhida para isolar a implementação híbrida nas mesmas quatro camadas. Ela não reproduz a proporção 1:7 normalmente associada às configurações de produção do Jamba.

## 5. Braços avaliados

### 5.1 Jamba oficial denso

- classe: `transformers.JambaForCausalLM`;
- 4 camadas com agenda Mamba/Attention pareada ao NSOS;
- `num_experts=1`;
- `num_experts_per_tok=1`;
- 1.354.896 parâmetros;
- sem kernels Mamba fundidos no runtime;
- treino em 117,4 segundos;
- pico observado pelo allocator do PyTorch: 417,8 MB.

### 5.2 Jamba oficial com MoE

- mesma base do Jamba denso;
- 16 especialistas;
- top-2 especialistas por token;
- camada MoE a cada duas camadas;
- offsets: `expert_layer_period=2`, `expert_layer_offset=1`;
- duas camadas `JambaSparseMoeBlock`;
- perda auxiliar do roteador habilitada;
- coeficiente da perda auxiliar: 0,001;
- 7.257.232 parâmetros;
- treino em 156,5 segundos;
- pico observado pelo allocator do PyTorch: 534,3 MB.

### 5.3 NSOS sem MoE

- implementação nativa C++/CUDA;
- Mamba2 faithful habilitado;
- expansão de estado 64;
- expansão interna 2;
- head Mamba 64;
- um grupo Mamba;
- atenção com período 2 e slot 1;
- MoE, KAN e TTT desativados;
- quantização progressiva desativada neste teste de paridade float;
- 768.152 parâmetros;
- treino em aproximadamente 34 segundos.

### 5.4 NSOS + OxtaMem

É exatamente o mesmo modelo NSOS treinado no braço anterior. Não há novo treino.

Para cada chave:

1. o trunk do modelo produz um vetor de 128 dimensões para `[QUERY, key]`;
2. o OxtaMem nativo grava chave, valor e vetor;
3. a consulta realiza busca vetorial cosine top-1;
4. o valor recuperado é injetado no modelo como `[key, value, QUERY, key]`;
5. a resposta final continua sendo produzida pelo NSOS.

## 6. Dados sintéticos MQAR

Formato de uma amostra:

`key_1, value_1, ..., key_n, value_n, QUERY, queried_key -> queried_value`

Distribuição:

| Item | Valor |
|---|---|
| Token de consulta | 1 |
| Tokens de valor | 2 a 65 |
| Tokens de chave | 66 a 1.265 |
| Número de pares no treino | uniforme entre 1 e 8 por batch |
| Pares por batch | mesmo comprimento dentro do batch |
| Batch | 32 |
| Passos | 2.500 |
| Gerador de dados de treino | `random.Random(7)` |
| Valores possíveis | 64 |
| Chance aleatória | 1/64 = 0,015625 |

As chaves de cada sequência são amostradas sem reposição. O par consultado é escolhido uniformemente entre os pares presentes.

## 7. Otimização

| Parâmetro | Valor |
|---|---:|
| Otimizador | AdamW |
| Learning rate máximo | 0,002 |
| Beta 1 | 0,9 |
| Beta 2 | 0,999 |
| Epsilon | 1e-8 |
| Weight decay | 0,01 |
| Gradient clipping | 1,0 |
| Warmup | 100 passos |
| Scheduler | cosine |
| Piso do learning rate | 10% do LR máximo |
| Atualizações | 2.500 |

O Jamba+MoE somou à cross-entropy a perda auxiliar oficial de balanceamento do roteador multiplicada por 0,001.

## 8. Convergência durante o treino

Médias móveis por janela de 500 passos:

| Passo | NSOS loss | Jamba denso loss | Jamba MoE CE | Jamba MoE aux |
|---:|---:|---:|---:|---:|
| 500 | 8,589 | 3,630 | 3,638 | 2,202 |
| 1.000 | 2,359 | 2,701 | 2,627 | 2,110 |
| 1.500 | 0,420 | 2,561 | 2,372 | 2,061 |
| 2.000 | 0,196 | 2,292 | 2,152 | 2,062 |
| 2.500 | 0,114 | 2,045 | 1,942 | 2,046 |

Os valores absolutos de loss são diagnósticos e não devem ser comparados como uma métrica perfeitamente normalizada entre implementações, pois os trainers e caminhos numéricos são diferentes. A comparação decisiva é feita nas mesmas amostras de avaliação.

## 9. Avaliação MQAR multissemente

Sementes de avaliação:

- 999;
- 9.008;
- 12.345;
- 31.415;
- 27.182.

Foram avaliados 300 exemplos por semente e por comprimento: 1.500 exemplos por linha da tabela. O desvio é o desvio-padrão populacional das cinco acurácias de semente.

| Pares | Jamba denso | Jamba + MoE | NSOS sem MoE |
|---:|---:|---:|---:|
| 1 | 1,000 ± 0,000 | 0,998 ± 0,002 | 1,000 ± 0,000 |
| 4 | 0,259 ± 0,025 | 0,241 ± 0,032 | **0,994 ± 0,002** |
| 8 | 0,128 ± 0,023 | 0,127 ± 0,030 | **0,955 ± 0,014** |

Execuções individuais do Jamba+MoE:

| Pares | Resultados por semente |
|---:|---|
| 1 | 0,9967; 1,0000; 1,0000; 0,9967; 0,9967 |
| 4 | 0,2500; 0,1900; 0,2700; 0,2767; 0,2200 |
| 8 | 0,1167; 0,1367; 0,1300; 0,0800; 0,1733 |

Diferenças principais:

- em 4 pares, o NSOS superou o Jamba denso por 0,735 ponto absoluto e o MoE por 0,753;
- em 8 pares, o NSOS superou o Jamba denso por 0,827 e o MoE por 0,828;
- em acurácia relativa, o NSOS foi aproximadamente 7,5 vezes melhor que os dois Jambas em 8 pares;
- o MoE não apresentou ganho sobre o braço denso nesse orçamento.

## 10. Avaliação de memória longa

Protocolo:

- número total de fatos `N`: 50, 200 e 1.000;
- janela disponível ao modelo sem memória: os últimos 8 fatos;
- consulta uniforme entre todos os `N` fatos;
- 3 sementes: 0, 1 e 2;
- 200 consultas por semente;
- 600 consultas por linha;
- OxtaMem nativo com busca cosine top-1.

| N | Jamba denso | Jamba + MoE | NSOS sem memória | NSOS + OxtaMem | retrieval@1 |
|---:|---:|---:|---:|---:|---:|
| 50 | 0,020 ± 0,018 | 0,025 ± 0,011 | 0,132 ± 0,014 | **1,000 ± 0,000** | 1,000 |
| 200 | 0,022 ± 0,008 | 0,023 ± 0,012 | 0,043 ± 0,012 | **1,000 ± 0,000** | 1,000 |
| 1.000 | 0,010 ± 0,004 | 0,013 ± 0,009 | 0,023 ± 0,002 | **1,000 ± 0,000** | 1,000 |

Execuções individuais do Jamba+MoE:

| N | Resultados por semente |
|---:|---|
| 50 | 0,035; 0,030; 0,010 |
| 200 | 0,020; 0,040; 0,010 |
| 1.000 | 0,020; 0,000; 0,020 |

Leitura:

- sem memória externa, a acurácia cai com o aumento de `N`;
- o comportamento do NSOS sem memória acompanha a probabilidade de o fato consultado estar na janela recente;
- o OxtaMem recuperou o valor correto em todas as consultas;
- em `N=1.000`, o sistema NSOS+OxtaMem foi aproximadamente 42,9 vezes melhor que o NSOS sem memória;
- o benefício do OxtaMem não veio de retreinar ou aumentar o modelo: veio da recuperação e injeção do fato correto.

## 11. Parâmetros, tempo e capacidade concedida ao rival

| Modelo | Parâmetros | Relação vs NSOS | Tempo de treino | Relação vs NSOS |
|---|---:|---:|---:|---:|
| NSOS sem MoE | 768.152 | 1,00x | 34,0 s | 1,00x |
| Jamba oficial denso | 1.354.896 | 1,76x | 117,4 s | 3,45x |
| Jamba oficial + MoE | 7.257.232 | 9,45x | 156,5 s | 4,60x |

O braço MoE recebeu muito mais capacidade e mais computação por passo. Portanto, o resultado não pode ser explicado por termos restringido artificialmente o rival. Mesmo com essa vantagem, a capacidade extra não resolveu a tarefa no orçamento de treino escolhido.

Os tempos não representam velocidade de produção: o Jamba oficial executou sem seus kernels Mamba fundidos, enquanto o NSOS usou sua implementação CUDA nativa.

## 12. Gates automáticos aprovados

O notebook exige:

- OxtaMem nativo disponível;
- retrieval@1 de pelo menos 0,99;
- ganho do OxtaMem em `N=1.000` superior a 0,30 sobre os braços sem memória;
- vantagem do NSOS sobre o Jamba denso em MQAR de 8 pares superior a 0,30;
- resultados MoE finitos;
- contagem de parâmetros MoE superior à densa;
- execução completa sem exceções.

Todos os gates passaram.

## 13. O que o teste demonstra

O teste fornece evidência forte de que, nesta configuração:

1. o NSOS apresenta excelente eficiência de aprendizagem de recall associativo;
2. o ganho não depende de MoE;
3. aumentar o número de parâmetros do Jamba não compensou sua dinâmica de aprendizagem no pequeno orçamento;
4. o OxtaMem resolve o limite imposto pela janela recente quando o retrieval encontra o fato correto;
5. o conjunto NSOS+OxtaMem é significativamente mais capaz que o modelo isolado nessa classe de tarefa.

## 14. O que o teste não demonstra

Este resultado não prova que:

- o NSOS é um modelo de linguagem geral melhor que checkpoints Jamba de produção;
- a vantagem continuará em bilhões de parâmetros;
- a vantagem continuará em linguagem natural;
- o NSOS vencerá com o Jamba usando kernels fundidos em throughput ou latência;
- as tabelas históricas das seções 8–10 usam uma execução nominalmente chamada seed 0; ela foi reclassificada como não determinística e não representa uma semente reproduzível;
- o OxtaMem sempre terá retrieval perfeito em bases ruidosas ou semânticas;
- MoE não funciona em escala; apenas não ajudou neste teste pequeno;
- a agenda 1:1 usada aqui representa a configuração de produção 1:7 do Jamba.

## 15. Ameaças à validade

- o MQAR corrigido usa três sementes de treino por braço, mas o bAbI ainda tem apenas uma semente para os dois rivais;
- tarefa sintética pequena;
- avaliação multissemente, mas sem múltiplos treinos;
- comparação em FP32 e sem quantização no caminho de treino;
- fallback PyTorch no rival oficial;
- diferenças inevitáveis entre trainers e implementações numéricas;
- parâmetros e FLOPs não normalizados;
- OxtaMem compara um sistema com memória externa contra modelos sem memória externa;
- embeddings de retrieval foram produzidos pelo próprio NSOS;
- nenhum teste de ruído, colisão semântica, atualização conflitante ou esquecimento;
- o bAbI QA1 foi incluído, mas ainda não cobre linguagem aberta, pré-treino geral ou retrieval semântico livre.

## 16. Próximos experimentos necessários

Para elevar a alegação de “protótipo impressionante” para “arquitetura comprovadamente superior”:

1. repetir o treino em pelo menos 5 sementes por braço;
2. medir curvas de 2.500, 5.000 e 10.000 passos;
3. comparar por número de passos, tempo de parede e FLOPs estimados;
4. testar a agenda de produção com 1 Attention para 7 Mamba em ambos;
5. ativar kernels fundidos no rival para comparação de desempenho;
6. executar MQAR com mais pares e extrapolação de comprimento;
7. testar selective copy, induction heads e passkey retrieval;
8. testar linguagem natural e memória de documentos;
9. adicionar ruído, chaves semanticamente próximas e valores conflitantes ao OxtaMem;
10. medir múltiplos `k`, reranking, atualização e persistência;
11. testar quantização do NSOS nos mesmos benchmarks;
12. publicar checkpoints, logs brutos, curvas e hashes de ambiente.

## 17. Veredito e nota

**Nota atual do projeto após a rodada ampliada: 9,1/10.**

Justificativa:

- arquitetura própria funcional;
- treinamento GPU nativo;
- Mamba+Attention validado;
- quantização funcional já demonstrada em testes anteriores;
- OxtaMem nativo com persistência e retrieval validado;
- ganho end-to-end muito grande e repetido em múltiplas sementes de avaliação;
- comparação realizada contra implementação oficial densa e com MoE;
- reprodutibilidade por notebook e gates automáticos.

A nota subiu porque agora há múltiplas sementes de treino, um benchmark público de linguagem natural e extrapolação até 64 pares. O que impede nota maior é a alta variância de otimização do NSOS, a ausência de múltiplas sementes dos rivais no bAbI, a falta de pré-treino/linguagem aberta e a necessidade de retrieval semântico não estruturado.

Formulação segura:

> “O NSOS+OxtaMem apresentou resultados excepcionais em benchmarks sintéticos controlados de recall associativo e memória longa, superando neste orçamento um Jamba oficial denso e um Jamba oficial com MoE, apesar de usar substancialmente menos parâmetros.”

Formulação que ainda não é sustentada:

> “O NSOS é melhor que o Jamba como modelo de linguagem geral.”


---

## 18. Errata de reprodutibilidade: seed 0

A auditoria estática e duas repetições no mesmo runtime mostraram que seed 0 não é uma semente determinística no NSOS. Em OXN/nsos/src/tensor.cpp, o gerador só segue o caminho determinístico quando global_seed != 0; caso contrário, usa std::random_device. O próprio DeterminismManager informa que global_seed igual a zero pode representar modo não determinístico.

Duas repetições nominais de seed 0 no mesmo runtime produziram trajetórias muito diferentes:

| Repetição | Loss final aproximada | MQAR 8 pares |
|---:|---:|---:|
| A | 3,159 | falha de convergência; sanity inicial 0,277 |
| B | 0,238 | sanity 0,920 |

Consequência: os números históricos das seções 8–10 continuam válidos como uma execução observada, mas não como uma execução reproduzível de seed 0. A campanha estatística corrigida abaixo usa apenas sementes não nulas.

## 19. MQAR com múltiplas sementes de treino corrigidas

Protocolo:

- sementes de treino: 1, 2 e 3;
- mesmos dados por semente entre os três braços;
- 2.500 passos, batch 32 e LR 0,002;
- 600 exemplos fixos de avaliação por dificuldade;
- dificuldades: 1, 4 e 8 pares;
- NSOS com reduções determinísticas habilitadas;
- média, desvio-padrão amostral e IC95% com t de Student (2 graus de liberdade).

| Pares | Jamba denso | Jamba MoE 16/top-2 | NSOS |
|---:|---:|---:|---:|
| 1 | 0,999 ± 0,001 | 0,999 ± 0,001 | 1,000 ± 0,000 |
| 4 | 0,301 ± 0,076 | 0,361 ± 0,137 | **0,846 ± 0,229** |
| 8 | 0,146 ± 0,028 | 0,157 ± 0,054 | **0,694 ± 0,354** |

Execuções individuais em 8 pares:

| Braço | Semente 1 | Semente 2 | Semente 3 |
|---|---:|---:|---:|
| Jamba denso | 0,128 | 0,178 | 0,132 |
| Jamba MoE | 0,132 | 0,120 | 0,218 |
| NSOS | 0,288 | 0,937 | 0,858 |

A média do NSOS permanece muito acima dos rivais, mas sua variância é alta. O IC95% com apenas três treinos é largo e não permite uma alegação estatística definitiva. A conclusão correta é vantagem forte e promissora em MQAR, acompanhada de instabilidade de otimização que precisa ser corrigida.

Os tempos do NSOS nesta rodada (246,3 ± 26,0 s) não são comparáveis ao caminho rápido histórico, porque as reduções determinísticas usam um caminho mais lento. Jamba denso: 110,4 ± 0,2 s; Jamba MoE: 145,3 ± 0,3 s no fallback PyTorch.

## 20. Benchmark público de linguagem natural: bAbI QA1

Fonte: [dataset facebook/babi_qa](https://huggingface.co/datasets/facebook/babi_qa), configuração en-10k-qa1, revisão imutável 1d86ad39d1c3ea2ff4b77eeb85f7c6ebd622a95f. O bAbI foi apresentado em [Towards AI-Complete Question Answering](https://arxiv.org/abs/1502.05698). Licença CC BY 3.0.

Integridade dos Parquets oficiais:

| Split | Exemplos de pergunta | SHA-256 |
|---|---:|---|
| treino | 10.000 | f1d67aa230d7aba0ed310df0d696a3ba9a07270e1670fe64c6901c24e5016f3f |
| teste | 1.000 | 9875dfad271cbfa5c49adb5809dd67be3826394a5d4d66dc74cab0c81483e2f8 |

Pré-processamento:

- tokenização por palavras e pontuação;
- vocabulário criado somente com treino: 25 tokens;
- zero tokens desconhecidos no teste;
- comprimento máximo: 83;
- história anterior completa mais pergunta;
- padding à esquerda;
- baseline majoritário: 18,7%.

Configuração comparável: semente 11, 1.500 passos, batch 32, LR 0,002, quatro camadas [Mamba, Attention, Mamba, Attention], dimensão 128 e estado Mamba 64.

| Braço | Parâmetros | Acurácia teste | Tempo T4 |
|---|---:|---:|---:|
| Jamba oficial denso | 1.196.048 | 49,3% | 1.590,6 s |
| Jamba oficial MoE 16/top-2 | 7.098.384 | 27,1% | 1.619,8 s |
| NSOS sem memória | 609.304 | **76,7%** | 93,7 s |
| NSOS + OxtaMem | mesmo modelo | **100,0%** | avaliação |

O OxtaMem atingiu retrieval@1 de 100,0% em 1.000 perguntas. A memória foi chaveada pela entidade extraída do texto, manteve a sentença mais recente por entidade e devolveu a sentença completa ao modelo. Não usou supporting_ids nem respostas de teste para recuperar. Este braço valida memória estruturada em texto público; não valida retrieval semântico livre.

Sementes adicionais do NSOS no mesmo bAbI:

| Semente | Acurácia |
|---:|---:|
| 11 | 76,7% |
| 12 | 65,2% |
| 13 | 63,7% |
| Média ± DP amostral | **68,5% ± 7,1%** |

Os rivais ainda têm apenas uma semente no bAbI, porque cada treino no fallback levou aproximadamente 27 minutos. Não se deve calcular significância entre arquiteturas com essa assimetria.

## 21. Escala: extrapolação MQAR até 64 pares

Rodada independente com semente 21. Todos os braços foram treinados apenas em 1–8 pares por 2.500 passos e avaliados em 400 exemplos fixos por comprimento.

| Pares | Tokens | Jamba denso | Jamba MoE | NSOS |
|---:|---:|---:|---:|---:|
| 8 | 18 | 12,5% | 12,8% | **95,8%** |
| 16 | 34 | 7,5% | 6,3% | **78,3%** |
| 32 | 66 | 4,3% | 4,8% | **49,3%** |
| 64 | 130 | 2,3% | 2,8% | **28,5%** |

Chance aleatória: 1/64 = 1,56%. O NSOS degrada de forma clara fora da distribuição, mas permanece muito acima dos rivais e do acaso até 64 pares. Como esta curva usa uma única semente, ela é evidência preliminar de extrapolação, não uma curva estatística final.

## 22. Veredito atualizado

A formulação tecnicamente segura agora é:

> O NSOS apresentou vantagem grande sobre Jamba oficial denso e MoE em MQAR com três sementes de treino, generalizou acima dos rivais até 64 pares e superou os rivais em uma rodada pública bAbI QA1. O NSOS+OxtaMem atingiu 100% quando recebeu uma chave de entidade estruturada. A arquitetura continua experimental: há alta variância entre sementes e ainda não foi validada como modelo de linguagem geral.

Nota atual: **9,1/10**.
