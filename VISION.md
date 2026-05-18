# OXTA — Visão de Pico

**Documento estratégico. Sem coisinhas, sem retórica.  O mais alto que essa estrutura pode chegar, com cada nível claramente etiquetado de "aterrado", "credível" ou "sonho".**

Última revisão: 2026-05-17 · Estado de validação: 12 testes de teses rodados localmente (GTX 1050 Ti) · NSOS em treino federado v11 no Colab T4.

---

## Índice

- [Parte 0 — A pergunta de fundo](#parte-0--a-pergunta-de-fundo)
- [Parte I — Onde estamos hoje (sem maquiar)](#parte-i--onde-estamos-hoje-sem-maquiar)
- [Parte II — Os cinco eixos reais de elevação](#parte-ii--os-cinco-eixos-reais-de-elevacao)
- [Parte III — Tier 1: o pico aterrado (12 meses)](#parte-iii--tier-1-o-pico-aterrado-12-meses)
- [Parte IV — Tier 2: a ambição credível (36 meses)](#parte-iv--tier-2-a-ambicao-credivel-36-meses)
- [Parte V — Tier 3: o teto da carreira (5+ anos)](#parte-v--tier-3-o-teto-da-carreira-5-anos)
- [Parte VI — O que é sonho (e por que é OK falar disso)](#parte-vi--o-que-e-sonho-e-por-que-e-ok-falar-disso)
- [Parte VII — As próximas 4 semanas](#parte-vii--as-proximas-4-semanas)
- [Apêndice A — Decisões irreversíveis a tomar antes de dezembro](#apendice-a--decisoes-irreversiveis-a-tomar-antes-de-dezembro)
- [Apêndice B — O que cortar do escopo HOJE](#apendice-b--o-que-cortar-do-escopo-hoje)

---

## Parte 0 — A pergunta de fundo

A pergunta que esse documento responde é:

> **"Considerando o que JÁ EXISTE no NSOS, o que VALIDAMOS empiricamente em 12 testes, e os recursos realistas (1 pessoa, GTX 1050 Ti local + Colab T4 + acesso esporádico a hardware maior) — qual é o teto realista pra esse projeto, e qual é a sequência de movimentos pra chegar lá?"**

Resposta curta: o teto aterrado é **um runtime de inferência local-first de classe mundial pra modelos de 40M-1B parâmetros, com tooling de deploy soberano e diferencial técnico real em 1.58-bit + hybrid SSM/attention/MoE**.  O teto credível é **um modelo open-source brasileiro que compete com Llama-3-8B em edge metrics em 36 meses**.  O teto-de-carreira é **infraestrutura de IA soberana pra Brasil/LATAM com base acadêmica em CEIA-UFG ou similar**.

AGI / superinteligência (que as duas teses prometem) **não está nesse roadmap como entrega**.  Está na Parte VI como o horizonte que motiva a direção, não como milestone.

Esse documento existe pra você parar de gastar ciclos em micro-otimizações que não movem a agulha estratégica, e começar a executar o que de fato eleva o projeto.

---

## Parte I — Onde estamos hoje (sem maquiar)

### O que NSOS é, contado por linhas de código e commits

| Componente | Linhas | Estado real |
|---|---|---|
| Runtime C++20 (kernel, tensor, autograd, BitNet 1.58, Mamba2 SSD, attention, MoE-KAN, TTT Hamiltoniano) | ~30K LoC | Compila, treina, converge.  GPU fast path validado pós-fixes (`82c5ee3`).  **Real.** |
| Tokenizer BPE com cache + UTF-8 boundary fix | ~1.5K LoC | Real. |
| HTTP API server com auth + admin gating | ~2K LoC | Real, mas vai pra reverse proxy em prod. |
| MemorySystem causal + FFI OxtaMem (Rust) | ~3K LoC | Real, mas OxtaMem ainda incubação. |
| MCTS Yggdrasil (System-2 reasoning) | ~2K LoC | Compila, raramente exercitado em prod. |
| CHRASS (topological routing) | ~5K LoC fora de `OXN/nsos/` | Incubação. |
| Trainer industrial (Adam, Muon, Sophia, AdamW, QAT progressivo, replay) | ~4K LoC | Treinando v11 agora. |
| Curriculum (16 datasets reais + sintéticos) | ~3K LoC scripts | Funciona, mas datasets ainda subaproveitados. |
| Testes (40 testes CTest + 12 Python) | ~12K LoC | 21 passam no gate, 10 ainda órfãos. |
| UI (React standalone, splash + login opcional + planos + temas) | ~2K LoC | Prototype clicável. |
| Validation harness das teses (12 testes isolados) | ~3.5K LoC | Real, rodou local, números concretos. |

**Soma honesta:** ~80K linhas de código próprio, sendo ~50K em `OXN/nsos/` produto suportado.  Isso é grande.  Pra contexto: GPT-2 original tinha ~5K LoC; llama.cpp tem ~25K.  **Você já tem mais infraestrutura escrita à mão do que a maioria dos labs open-source.**

### O que VALIDAMOS empiricamente nos 12 testes (números reais, não retórica)

Da rodada local em `research/theses_validation/`:

**✅ Funcionam de verdade nessa escala:**
- **CfC (Closed-form Continuous-time)**: ganha **+12.5%** PPL vs RNN baseline (PPL 7.16 vs 8.18). Ganho de qualidade real, único entre as variantes de arquitetura sequencial.
- **HGF (FP residual paralelo ao ternário)**: ganha **+2.7%** PPL com ~0.5% mais parâmetros. Marginal mas honesto.
- **GRPO sem critic**: usa **50% menos parâmetros** que PPO (4.6K vs 9.1K), 25% mais rápido. Qualidade pior no CartPole (60 vs 42 retorno) mas o claim de economia de memória é REAL.

**🟡 Empate ou inconclusivo:**
- **Decoupled-STE vs STE clássico**: 0.4% diff (ruído). Init correto + LayerNorm já estabiliza STE puro.
- **LrcSSM**: Mamba2 baseline pulou no Windows (sem `mamba_ssm`), então não dá pra falar. Precisa rodar em Linux.

**🔴 NÃO replicam (refutam as teses):**
- **Continual QAT (FP→ternary anneal)**: cold ternary GANHOU (PPL 9.48 vs 9.62). Schedule da tese precisa ser mais sofisticado.
- **Denoising Dequant Transform (Ridge gradient)**: REGRIDIU 45% (PPL 14.6 vs 10.1).
- **SKAN single-param**: PERDE 3.3× pra KAN multi-param (MSE 0.45 vs 0.14). Tese diz "mais estável" — na verdade muito menos expressivo.
- **KAN vs MLP**: MLP esmaga KAN 68× em regressão simples. KAN não é universal magic.
- **AB-MCTS vs UCT**: UCT ganhou 14% em árvores sintéticas sem priors.
- **NCA pretraining (escala pequena)**: -3.4% (hurt). Talvez melhore com 164M tokens, mas isso é assumption.

### O que isso significa estrategicamente

**Você tem 2-3 técnicas validadas que de fato ajudam (CfC, HGF, GRPO-memory).** Tudo o resto que as teses prometem é ou marginal, ou regride, ou requer escala que você não tem hoje pra testar.

**Implicação:** pare de planejar a roadmap em torno do que as teses gritam. Planeje em torno do que JÁ FUNCIONA + o que CABE construir.

### A questão estratégica que ninguém quer fazer

Quanto tempo, tempo realista, você tem disposto a investir nisso?

- Se a resposta for **6-12 meses**: Tier 1 da Parte III é seu pico. Foco em ship.
- Se a resposta for **2-3 anos**: Tier 2 da Parte IV vira credível.
- Se a resposta for **5+ anos com fundo/parcerias acadêmicas**: Tier 3 entra em jogo.
- Se a resposta for **AGI ou nada**: você vai se queimar e o projeto morre. Releia a Parte 0.

---

## Parte II — Os cinco eixos reais de elevação

Cada eixo tem (a) o claim grande das teses, (b) reality check do que dá pra fazer, (c) métrica alvo concreta, (d) primeiro entregável.

### Eixo 1 — Edge runtime supremacy

**Claim grande:** "Roda 100B param model em DGX Spark com 128GB."

**Reality check:** Você não tem DGX Spark. Você tem GTX 1050 Ti + T4 Colab. Mas o que VOCÊ TEM que ninguém mais tem é:
- Kernel 1.58-bit ternário em C++20 SIMD funcional (`gemm_158bit_ultra`)
- Mamba2 SSD batched implementado nativo
- Mix híbrido attention + MoE-KAN + SSM no mesmo modelo
- Inferência streaming com `MambaStreamSnapshot`
- Tudo isso compilando como UM binário Docker

Isso é diferenciado. Nenhum outro projeto open-source tem essa combinação específica.

**Métrica alvo (12 meses):**
- Inferência **40M model em 50+ tok/s em CPU Ryzen 5 / i5** (sem GPU)
- Inferência **150M model em 200+ tok/s em GTX 1050 Ti** (Pascal sm_61, o pior GPU NVIDIA ainda relevante)
- Inferência **400M model em 500+ tok/s em T4** (sm_75)
- Footprint de pack: **< 80 MB pro 40M, < 300 MB pro 150M**
- Cold start (load → primeiro token): **< 2 segundos**

**Primeiro entregável (2 semanas):**
- Suíte de benchmarks `OXN/nsos/benchmarks/edge_throughput.py` que mede tok/s em 3 hardwares (CPU, 1050 Ti, T4) com 3 tamanhos de modelo, escreve JSON, gera tabela markdown.
- README do NSOS aponta pra essa tabela como diferencial.

### Eixo 2 — Qualidade de treino (credibilidade)

**Claim grande:** "Active Inference + NCA pretrain + Continual QAT + 6 outras coisas = AGI."

**Reality check:** 5 das 6 técnicas falharam ou empataram no nosso teste. O que de fato move qualidade é o boring básico:
- Mais tokens de qualidade
- LR schedule decente (cosine warmup, ok)
- Mixed precision real (BF16 — já habilitado)
- Replay-aware curriculum (já implementado)
- Boa tokenização (já feito)

**Métrica alvo (12 meses):**
Em modelo 40-80M (Chinchilla-ótimo pro budget de tokens que você consegue):
- **HellaSwag**: > 35% (random = 25%, GPT-2 small = 31%)
- **ARC-Easy**: > 40% (random = 25%, GPT-2 small = 44% — sim, alvo competitivo)
- **MMLU stem (subset 100 questões)**: > 27% (random = 25% — sair do ruído já é vitória)
- **HumanEval (Python)**: > 5% pass@1 (TinyLlama 1.1B faz 10% — meta proporcional)
- **WikiText-2 PPL**: < 25 (GPT-2 small = 29)

**Primeiro entregável (4 semanas):**
- Pipeline de eval `OXN/nsos/eval/standard_benchmarks.py` que roda os 5 acima contra qualquer model pack `.bin`.
- Roda no v11 checkpoint atual, baseline numbers documentados em `OXN/nsos/docs/SCORECARD.md`.
- Cada novo checkpoint regrava a tabela. Public-facing.

### Eixo 3 — Raciocínio híbrido (diferenciador)

**Claim grande:** "MCTS + R-MCTS + AB-MCTS + Active Inference + ReST-MCTS* = pensamento profundo verdadeiro."

**Reality check:** AB-MCTS perdeu pra UCT no teste. Mas MCTS no espaço latente em conjunto com:
- **CfC pra TTT** (validado: +12.5%)
- **GRPO no loop de polish** (validado: economiza VRAM)
- **Verifier-based reward** (compilador Python, sympy pra matemática) já existe no curriculum

...isso é uma chain real. Não AGI, mas é um pequeno-modelo-que-raciocina-mais-que-seu-tamanho.

**Métrica alvo (12 meses):**
- 80M model com MCTS @ 16 simulações ganha **+30% no GSM8K subset** vs mesmo modelo sem MCTS
- TTT com CfC ativo durante decode dá **+15% em ARC-Challenge** vs decode estático
- Latência: MCTS adiciona ≤ 5× wall-time vs decode direto (não pode ser 50×)

**Primeiro entregável (8 semanas):**
- `OXN/nsos/eval/reasoning_chain.py` que pega um pack, roda 3 modos (vanilla, MCTS-only, MCTS+TTT-CfC) em 100 problemas GSM8K, reporta accuracy e latência.
- Decisão go/no-go em 8 semanas: se MCTS+TTT não bater vanilla por >20%, abandona o caminho e foca no Eixo 2.

### Eixo 4 — Deploy soberano (caso de uso)

**Claim grande:** "Substitui OpenAI."

**Reality check:** Você não vai substituir OpenAI. Mas você PODE oferecer um **runtime local-first que roda no PC do usuário, sem cloud, sem subscription, sem telemetria** — exatamente o que DreamServer faz pra Llama/llama.cpp, mas com diferencial de 1.58-bit + hybrid arch.

**Métrica alvo (12 meses):**
- **1 comando** instala em Windows / Linux / Mac
- Roda em **8GB RAM mínimo** (40M model)
- UI web local na porta 8080 (ou Electron wrapper) — já tem o protótipo em `colab/IdeiaSiteOxta` adaptável
- **100 self-host installs validados** em 6 meses (Discord small + GitHub stars)
- Tiered offering: Free (40M), Plus (80M + memória OxtaMem), Pro (200M + MCTS) — sem cobrar ainda, só estrutura

**Primeiro entregável (6 semanas):**
- `install.ps1` Windows + `install.sh` Linux/Mac que baixam binário pré-built + pack 40M, sobem servidor HTTP, abrem browser
- Docker Compose `docker-compose.cpu.yml`, `.cuda.yml` modelando DreamServer
- Documentação `OXN/nsos/docs/INSTALL.md` com tempo-pra-funcionar < 5 min
- Tela web local consumindo `nsos_api_server`

### Eixo 5 — Output de pesquisa (multiplicador de credibilidade)

**Claim grande:** "Vai aparecer em paper."

**Reality check:** Você TEM trabalho original digno de paper. Não na escala que as teses prometem, mas concretamente:
- Validation harness das teses (12 técnicas) é por si só uma **negative-results paper** valiosa pra workshop NeurIPS — comunidade frequentemente publica "X não replica em Y condições"
- Combinação BitNet 1.58 + Mamba2 + CfC TTT é nova
- Edge throughput em hardware modesto (1050 Ti) é diferenciado

**Métrica alvo (12 meses):**
- **1 paper em workshop NeurIPS / ICLR / EMNLP** sobre validation harness
- **1 blog post técnico** detalhando arquitetura híbrida com benchmarks
- **GitHub stars > 200** (proxy de adoção open-source)
- **2-3 issues / PRs externos** = sinal de comunidade

**Primeiro entregável (4 semanas):**
- Rascunho de paper (8 páginas) baseado em `research/theses_validation/` — "On the (Non-)Replicability of Modern Sub-Bit Quantization Claims at Small Scale"
- Tabela completa dos 12 testes + ablations
- Submeter pro workshop "Has it Trained Yet?" ou "Practical ML" — workshops aceitam negative-result papers melhor que conferências main

---

## Parte III — Tier 1: o pico aterrado (12 meses)

**Tese:** em 12 meses, com 1 pessoa em dedicação parcial e os recursos atuais, você entrega um **edge-first LLM runtime de classe profissional, com diferencial técnico real e ~1K usuários self-host**.

### Marcos trimestre a trimestre

**Q1 (jun-ago 2026) — Fechar MVP, primeiros números públicos**
- ✅ Treino v11 termina (40M model, ~1.5h pós-BF16 fixes)
- 📦 Pack edge ≤ 80 MB validado, carrega < 2s
- 🧪 Eval suite com 5 benchmarks rodando, números publicados
- 🚢 Install script Windows/Linux/Mac
- 📝 Paper draft submetido pra workshop

**Q2 (set-nov 2026) — Pôr nas mãos de usuários**
- 🚀 Launch público em ProductHunt + r/LocalLLaMA
- 👥 Discord com 100 usuários
- 📈 Tier free roda no Steam Deck + Raspberry Pi 5
- 🔌 Wrapper LangChain/llamaindex compatível com Oxta API
- 🎯 80M model v12 trained com Continual QAT REAL (schedule sofisticado, não o broken test version)

**Q3 (dez 2026 - fev 2027) — Diferenciador empilha**
- 🧠 MCTS reasoning rodando em produção (Pro tier)
- ⏱️ CfC TTT integrado (validado em paper)
- 🌐 OxtaMem 4D promovido pra produto (não mais incubação)
- 🎓 Apresentação em conferência acadêmica brasileira (SBC?)
- 💰 Primeiro pricing structure: $0 free / $X plus / $Y pro

**Q4 (mar-mai 2027) — Solidificar**
- 🏗️ 200M model v13 destilado de Llama-3-8B (open weights → seu runtime)
- 📊 Scorecard mensal público (GitHub Pages dashboard)
- 🤝 Parceria com 1 universidade (CEIA-UFG ou USP) pra acesso a hardware maior
- 👥 1000 self-host installs medidos
- ⭐ 500+ GitHub stars

### Métricas-chave do Tier 1

| Métrica | Estado hoje | Alvo 12 meses |
|---|---|---|
| Modelo melhor treinado | 40M, PPL ~10 char-level | 200M, HellaSwag 35%+ |
| Tok/s em CPU médio | n/a medido | 50+ (40M) |
| Tok/s em T4 | ~5-10 (com perf fixes) | 200+ (40M) |
| Pack size 40M | 25 MB (validado) | mantido < 30 MB |
| Cold start | desconhecido | < 2s validado |
| Self-host installs | 0 | 1000 |
| GitHub stars | 0 (privado?) | 500+ |
| Papers publicados | 0 | 1 |
| Eval benchmarks rodando | 0 públicos | 5 com scorecard mensal |
| API uptime | n/a | 99% em self-host de referência |

### O que isso requer de você

- **20-30h/semana** sustentado por 12 meses (tempo realista pra 1 pessoa entregar tudo isso)
- **$100-300/mês** em Colab Pro + Hugging Face Pro + AWS spot pra rebuilds
- **Disciplina de cortar escopo**: não adicionar nada que não esteja nesse roadmap até Q3
- **Comunicação pública**: blog post mensal, mesmo curto

### O que isso NÃO requer

- ❌ DGX Spark / cluster Blackwell / NVIDIA enterprise
- ❌ Validar todas as 21 técnicas das teses (a maioria NÃO replicou — pare de gastar ciclos nelas)
- ❌ Reinventar Active Inference / HDRAM / Hamilton-Jacobi
- ❌ AGI rhetoric no marketing — vai ser ridicularizado e justo

---

## Parte IV — Tier 2: a ambição credível (36 meses)

**Tese:** em 3 anos, com Tier 1 entregue + acesso a hardware acadêmico (CEIA-UFG ou similar) + 1-2 colaboradores, você entrega um **modelo open-source brasileiro que compete com Llama-3-8B em métricas de edge e inferência local**.

### O caminho

Tudo aqui é construído sobre Tier 1 entregue. Se Tier 1 falhar, Tier 2 não acontece.

**Ano 2 (mai 2027 - mai 2028)**
- **Modelo 1B parâmetros** treinado em hardware acadêmico (CEIA-UFG 31 nós A100 — você cita isso nas teses, então provavelmente tem ou pode obter acesso)
- **Curriculum 50B tokens** real (versão expandida do v11, com TinyStories + Cosmopedia + Brazilian Portuguese)
- **Hybrid arquitetura final**: BitNet 1.58 backbone + 30% attention layers + 20% MoE-KAN + 10% CfC TTT layers — combinação validada empiricamente
- **Real RLHF/GRPO** com verifier (compilador Python + theorem prover Lean — `lean_integration.h` é stub hoje, vira real aqui)
- **Multilingual fine-tune**: EN + PT-BR (BR é o diferencial vs todos os modelos US)

**Ano 3 (mai 2028 - mai 2029)**
- **Modelo 8B** se hardware permitir, ou **modelo 1B híper-otimizado** se não
- **OxtaMem como produto separado** (memória vetorial 4D pra qualquer agent — não só Oxta)
- **2-3 papers publicados** em conferências main (ICLR / EMNLP / NeurIPS)
- **Comunidade de desenvolvedores ativa** (10+ contributors, plugin ecosystem)
- **Possível spin-off comercial**: empresa pequena cobrando enterprise support + cloud-hosted Pro tier

### Métricas-chave do Tier 2

| Métrica | Tier 1 (12m) | Tier 2 (36m) |
|---|---|---|
| Tamanho do modelo flagship | 200M | 1B (mínimo) ou 8B (com hardware) |
| Token budget de treino | ~50M | 50B+ |
| Tokens por segundo (T4) | 200+ | 500+ |
| Self-host installs | 1000 | 10K |
| GitHub stars | 500+ | 5K+ |
| Papers | 1 | 4+ |
| Eval HellaSwag | 35% | 60%+ (Llama-3-8B é ~80%) |
| Eval MMLU | 27% | 40%+ |
| Receita anual (se monetizar) | $0 | $50K-300K (enterprise + Pro tier) |
| Pessoas no projeto | 1 | 2-3 |

### O que isso requer

- **Acesso a hardware acadêmico** real (CEIA-UFG ou parceria similar) — você precisa cultivar essa relação **agora**, durante Tier 1
- **1-2 colaboradores** (research engineer + frontend/devops)
- **Fundação legal**: estrutura jurídica (associação ou startup) pra contratar e receber funding
- **Possível seed funding** $100-500K (anjos LATAM, FAPESP, BNDES Garagem, Y Combinator se for via SaaS)

### Por que isso é credível, não delírio

- O Llama-3-8B foi treinado em 15T tokens. **Você não vai fazer isso.**
- Mas Phi-3-mini (3.8B) foi treinado em 3.3T tokens e ganha de Llama-3-8B em muitas tarefas via **curriculum cuidadoso + sintético**. Esse é o caminho viável.
- TinyLlama (1.1B em 3T tokens) é o exato shape do que você pode replicar com CEIA-UFG por ~3 meses de A100 time.

### Por que pode falhar

- Acesso a hardware não materializar
- Você queimar entre Q3 Tier 1 e início Tier 2 (transição é o ponto crítico de burnout em projetos solo)
- Llama-4 ou Mistral lançar algo que torna o nicho irrelevante
- Brasil entrar em recessão / fundos secarem

Cada um desses tem probabilidade não-trivial. **Esperar 70% chance de Tier 2 dar certo é otimista mas defensável.**

---

## Parte V — Tier 3: o teto da carreira (5+ anos)

**Tese:** em 5+ anos, com Tier 2 entregue + reconhecimento acadêmico + possível fundo institucional, você lidera **a iniciativa de IA soberana de classe-mundial pro Brasil/LATAM**.

Isso já é especulação responsável, mas dentro do plausível.

### O que isso seria

- **OXTA Foundation** — instituto sem fins lucrativos (ou empresa B-Corp) sediado no Brasil
- **Modelos open-source de 8B-70B parâmetros** treinados em hardware brasileiro
- **Foco em Portuguese-BR + idiomas indígenas + linguagens jurídicas/médicas BR** (nichos onde modelos US não competem)
- **Plataforma de fine-tuning low-code** pra empresas brasileiras (similar ao OpenAI fine-tuning mas local-first, LGPD-compliant)
- **Parcerias governamentais**: BNDES, Embrapa, INPE, MEC — todos têm casos de IA que precisam soberania
- **Reconhecimento acadêmico**: cadeira na pós de uma universidade brasileira

### Como isso financia

- **Government grants** (FAPESP, CNPq, BNDES) — Brasil quer IA soberana, dinheiro existe
- **Enterprise contracts** (bancos, agronegócio, jurídico — todos têm pavor de mandar dados pra Azure/AWS)
- **Open-source dual licensing** (modelo grátis pra hobbyistas, license comercial pra empresas)
- **Eventual Series A** ($5-10M) se o caminho enterprise fizer sentido

### Por que isso NÃO é delírio

Veja o que existe hoje:
- **CEIA-UFG** com 31 nós DGX (citado nas teses — você sabe disso)
- **SENAI CIMATEC** com 27.5 petaflops (oficial)
- **Programa Brasileiro de IA** (PBIA): R$ 23 bi anunciado até 2028
- **Mistral AI** (França) atingiu unicórnio em 18 meses fazendo exatamente isso pra Europa
- **CMU + Together AI** ecosystem está mostrando que coordenar pesquisa + treino + serving é viável em escala menor que Big Tech

O nicho de "soberania de IA pra LATAM" está vazio. **O primeiro projeto credível que ocupar esse espaço captura ele.**

### Por que pode dar errado

- Política brasileira instável (cortes de orçamento, mudança de governo)
- Big Tech US oferecendo soluções local-friendly (Azure data residency BR, etc.) erodir o caso de soberania
- Hardware no Brasil envelhecer mais rápido que recurso pra atualizar
- Open-source LLMs (Llama-N, Mistral-N) escalarem tanto que projetos especializados ficam irrelevantes

### A pergunta honesta

> Você quer ser empreendedor / pesquisador-CEO? Tier 3 exige isso. Não é mais "construir software".

Se a resposta é não, pare em Tier 2 e venda/doe o projeto pra alguém que queira tocar. Tier 2 já é uma vitória notável.

---

## Parte VI — O que é sonho (e por que é OK falar disso)

As duas teses prometem AGI / superinteligência. Vamos ser explícitos sobre o que é fundamentado e o que é não.

### Os claims das teses, ranqueados por aterramento

| Claim | Nível |
|---|---|
| LLMs de 1.58-bit rodam em hardware modesto | ✅ Real (BitNet b1.58 paper, 2024) |
| Mamba2 é competitivo com Transformer | ✅ Real (Gu & Dao, 2024) |
| MoE + sparse activation = mais params por compute | ✅ Real (Mixtral, DeepSeek) |
| CfC closed-form RNN existe e funciona | ✅ Real (Hasani et al., MIT) |
| GRPO economiza memória vs PPO | ✅ Real (DeepSeek-R1) |
| KAN tem benefícios em alguns domínios | ✅ Real mas estreito |
| Sparse Autoencoders extraem conceitos interpretáveis | ✅ Real (Anthropic 2024) |
| NCA pretraining ajuda transfer | 🟡 Reportado em 1 paper, não replicado em escala |
| Active Inference para LLMs | 🔴 Teoria não-operacionalizada |
| HDRAM com Hypertoken ECC + Krylov | 🔴 Termo não existe na literatura |
| Hamilton-Jacobi PDE pra TTT | 🔴 Teoria, sem implementação concreta |
| Differentiable Logic elimina alucinação | 🔴 LTNs não escalam pra LLM |
| 45.58% sparsity é uma scaling law universal | 🔴 Número específico sem citação |
| 100B params em DGX Spark | 🔴 Aspiração de hardware |
| "Inteligência divina silicada" | 🔴 Retórica |

Os 5 últimos não são pesquisa, são marketing. Mantê-los no escopo do produto vai te desacreditar.

### Por que ainda vale ter o sonho no horizonte

Sonhos motivam decisões de longo prazo. **O sonho aqui não é "AGI em 2030", é "IA soberana brasileira de classe mundial em 5 anos".** Isso é factível.

Cada vez que você for tentado a adicionar um claim grandioso na documentação, pergunte:
1. Tem paper / código de referência que faz exatamente isso?
2. Eu posso medir se funciona em < 2 semanas?
3. Se funcionar, move uma métrica do Tier 1?

Se a resposta a qualquer é não, joga no `speculative_claims.md` ou nesse documento (Parte VI). Não no produto.

### O que mantemos das teses

Da Tese 1 e 2 combinadas, vai pro produto:
- **Mamba2 + Attention hybrid** (já tem)
- **BitNet 1.58 ternário** (já tem, com kernels SIMD)
- **MoE-KAN** (já tem como opção)
- **CfC pra TTT** (validar e integrar — Eixo 3)
- **GRPO pra RL phase** (Tier 1 Q3)
- **VSA / holographic operations** numa forma mínima (incubação, não produto)
- **Curriculum determinístico** (já tem)
- **Edge-first deployment** (já tem)

O resto: arquive em `legacy/` ou `speculative/`. Não delete (você pode revisitar daqui a 5 anos quando tiver mais escala), mas **pare de planejar em torno disso**.

---

## Parte VII — As próximas 4 semanas

Aqui parou de ser estratégia e começou tática. Cada item tem prazo e entregável.

### Semana 1 (a partir de amanhã)

- [ ] **Treino v11 atual termina** com BF16 + 4 perf fixes (~2-3h). Salvar phase3, phase4, phase5, phase6, polish em Drive.
- [ ] **Eval baseline do v11**: rodar 5 benchmarks (HellaSwag, ARC-Easy, MMLU-100, HumanEval-light, WikiText2-PPL) → primeira tabela `OXN/nsos/docs/SCORECARD.md`.
- [ ] **Decisão MoE-KAN**: vale a complexidade dado que KAN perdeu pra MLP no teste? Se não vale → arquivar especialistas KAN, simplificar pra MoE-MLP standard. **Resposta antes de sexta.**
- [ ] **Smoke test do install.sh** em Linux (qualquer VM): clone, build, baixa pack, sobe servidor, hit `/generate`, valida resposta. Documenta no INSTALL.md.

### Semana 2

- [ ] **Edge throughput benchmark** (`OXN/nsos/benchmarks/edge_throughput.py`): mede tok/s em CPU + 1050 Ti + T4 pra 40M e 80M packs. Tabela markdown como output.
- [ ] **Tela web local** funcionando: o protótipo `colab/IdeiaSiteOxta` apontando pra `nsos_api_server` real, gerar texto via API. Demo gravada.
- [ ] **Comparativo com llama.cpp**: rodar TinyLlama 1.1B no llama.cpp na mesma máquina. Documentar honestamente onde ganhamos e onde perdemos.
- [ ] **Plano de paper rascunhado** (1 página): tese, contribuições, tabela principal, target venue (workshop NeurIPS 2027 ou EMNLP findings).

### Semana 3

- [ ] **`install.ps1` Windows**: testado em VM clean Windows 11. Tempo total < 5 min de zero a servidor up.
- [ ] **Decisão sobre OxtaMem**: produtizar agora (Tier 1 Q3) ou esperar (Tier 2)? Critério: se for genérico o suficiente pra usar fora do Oxta, vale produtizar agora pra brand.
- [ ] **Lista de 5 datasets adicionais** pra v12 (português-BR é o diferencial): wikipedia-PT, brwac, c100-PT, jur-PT, gov-BR.
- [ ] **Outreach**: 3 pessoas (acadêmico, empreendedor, dev local) recebem demo de 15 min via Discord/Zoom. Coletar feedback honesto.

### Semana 4

- [ ] **Treino v12 começa** (80M model com decisões tomadas: MoE on/off, KAN on/off, datasets BR adicionados). Schedule pra rodar 3-4 dias com checkpoints a cada 100 steps no Drive.
- [ ] **Paper draft 4 páginas** (intro + método + tabela 12 testes + discussão honest negative results).
- [ ] **Docker compose** funcionando em 3 perfis (CPU-only, NVIDIA, ROCm-experimental).
- [ ] **Decisão de timeline**: bater Tier 1 em 12 meses real OU estender pra 18 meses real? Importante ser honesto antes de prometer publicamente.

### O que NÃO fazer nessas 4 semanas

- ❌ Investigar Active Inference / FEP / HDRAM / Hamilton-Jacobi
- ❌ Implementar mais variantes de quantização além de BitNet 1.58 e BF16
- ❌ Adicionar features na UI além de "funciona com servidor real"
- ❌ Refactor de god-files em `OXN/nsos/src/` além do mínimo
- ❌ Otimizar perf além do que está no commit `82c5ee3` — já é bom o suficiente pra Tier 1

---

## Apêndice A — Decisões irreversíveis a tomar antes de dezembro

Estas decisões fecham portas de uma vez. Não dá pra voltar atrás sem custo alto. Tome conscientes.

### A1 — Licença

| Opção | Implicação |
|---|---|
| **Apache 2.0** | Permissivo. Big Tech pode embebe sem dar nada de volta. Maior adoção. **Recomendado** se você quer Tier 2/3. |
| **AGPL-3.0** | Copyleft strong. Bloqueia uso cloud sem open-source. Menos adoção, mais proteção. |
| **BUSL** (Business Source License) | Dual: open pra hobbyistas, commercial pra empresas. Permite monetização. Mais complexo. |
| **Custom RAIL** (Responsible AI License) | Restrições éticas. Mais nicho. |

**Sugestão:** Apache 2.0 pra produto + AGPL pra peças experimentais (CHRASS, HDRAM se voltar). Decida e marque commit.

### A2 — Brand: OXTA, NSOS, ou ambos?

Hoje é confuso: NSOS é o engine, OXTA é o produto/UI, OxtaMem é módulo. Vai dar dor de cabeça.

**Sugestão:**
- **OXTA** vira brand público (produto, site, social)
- **NSOS** vira nome interno do engine (como llama.cpp vs Llama brand)
- **OxtaMem** vira componente de OXTA

### A3 — Open-source vs proprietário do modelo

Pesos: open ou closed?
- Open → mais adoção, mais credibilidade, dificulta cobrança Pro tier
- Closed → permite monetização Pro tier, mas precisa entregar muito valor

**Sugestão:** Free tier (40M) open weights, Plus/Pro (80M+) closed weights inicialmente.

### A4 — Política de aceitação de contribuições

Decida ANTES de aparecer PR externo:
- CLA exigido? (defensivo, friccionante)
- DCO simples? (recomendado)
- Mantém só você no `main`? Ou abre commit access pra 2-3 contribuidores?

### A5 — Posicionamento

Você é:
- **Pesquisador** que faz produto como side-effect? → Foca em papers + impacto acadêmico
- **Engenheiro de produto** que faz pesquisa pra dar credibilidade? → Foca em usuários + receita
- **Empreendedor de technology** que quer construir empresa? → Foca em product-market fit + funding

Não dá pra ser os três. Decida.

---

## Apêndice B — O que cortar do escopo HOJE

Pra liberar tempo pras 4 semanas + 12 meses, **arquive os seguintes em `legacy/` ou `speculative/` esta semana**:

### Em incubação que NÃO vai ser produto em Tier 1

- `KernelOpen/` (UHK heterogêneo) — fascinante mas não move agulha em 12 meses
- `CHRASS/` standalone (Kimera V19) — pesquisa interessante, sem caminho de produto
- `CART/` (PEFT framework) — não está no caminho crítico
- `OXB/` (data ingestion) — paralelo, não bloqueia nada
- `pantheon/` (frameworks abstratos) — pesquisa, não produto
- `hardware/` (Verilog) — fora de escopo de software
- `bindings/` root (legado) — substituído por `OXN/nsos/src/bindings.cpp`

### Em `OXN/nsos/src/` que pode sair

- `chat.cpp` (demo legado) — substituído por API HTTP
- `server.cpp` (demo legado) — idem
- `fabric.cpp` v1 — `fabric_v2.cpp` é o que CMake usa
- `lean_integration.h` — stub que nunca foi implementado
- `mpi_mock.h` — manter mas marcar EXPLICITAMENTE MOCK
- `persistent_kernel.cu` — gated atrás de option default OFF
- `inference.py`, `data_pipeline.py`, `evaluation.py` — stubs que viram `NotImplementedError` clara

### Scripts na raiz

Já feito em PR-0.1 e PR-0.2 do plan original. Confirmar que estão em `legacy/scripts/`.

### Claims na documentação

Vasculhar `OXN/nsos/docs/*.md` e remover:
- Toda menção a "AGI", "superinteligência", "divindade silicada"
- Promessas de 100B params
- Active Inference / FEP no escopo produto
- HDRAM / Hypertokens / Krylov subspaces
- "Elimina alucinações via Differentiable Logic"

Mover essas pra `OXN/nsos/docs/SPECULATIVE.md` com nota "future research, not in current scope".

---

## Fechamento

Você tem 80K linhas de código original, 2-3 técnicas validadas que de fato funcionam, infraestrutura federada de treino que persiste em Drive, runtime C++ que compila e converge, e um nicho real (1.58-bit edge + hybrid arch + sovereign Brazilian AI).

Isso é mais do que 95% dos projetos open-source LLM começam com. **Você não tem o problema de "falta de tecnologia". Você tem o problema de "execução focada vs dispersão em coisinhas".**

Esse documento é a fronteira do que dá pra fazer com o que existe.

Nas próximas 4 semanas, ou você começa a executar os marcos da Parte VII e o projeto eleva, ou continua resolvendo perf fix no jamba.cpp e o projeto definha. **Não há terceira opção realista.**

Bora jogar.

---

*Próxima revisão deste documento: **2026-08-17** (3 meses).  Métricas serão atualizadas com números reais ou marcadas como "missed" com retrospectiva honesta do porquê.*
