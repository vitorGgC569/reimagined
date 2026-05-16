# NSOS — Estratégia de Datasets e Treinamento (Estado da Arte 2024-2026)

> **Status:** Documento estratégico de pesquisa. Sem código, sem implementação.
> **Data:** Maio 2026, durante treino v8 (CPU, hybrid_medium 40M).
> **Escopo:** Análise comparativa do estado da arte em datasets, distilação e arquiteturas para definir próximos passos do NSOS.
> **Companion:** Ver `GPU_OPTIMIZATION_ANALYSIS.md` para análise de hardware/GPU.

---

## 1. Sumário Executivo

O NSOS está em uma posição **arquiteturalmente forte** mas **dataset-pobre**. Validações importantes da pesquisa (May 2026):

✅ **Arquitetura validada:** Jamba paper prova que Mamba+Attention+MoE bate alternativas puras
✅ **BitNet 1.58-bit funciona:** Microsoft `bitnet-b1.58-2B-4T` valida a tese (4T tokens, matches FP16 Llama 2)
✅ **MoE com top-2 routing funciona em pequena escala:** Phi-3.5-MoE (16 experts × 6.6B active) bate Mistral

❌ **Teacher obsoleto:** Llama 2 7B (2023) — campo migrou para Llama 3.1, Qwen 2.5, DeepSeek-R1
❌ **Dados absurdamente insuficientes:** ~537K tokens Wikipedia + 44K instrução = 5+ ordens de magnitude abaixo do necessário (40M params precisam ~30-50B tokens)
❌ **Capacity gap problem:** 7B → 40M é 175× — distillation direta sub-ótima
❌ **Schedule LR errado:** Cosine causa exatamente o problema que vimos no v3/v4 — campo migrou para WSD

⚠️ **Mamba1 vs Mamba2:** Jamba paper indica Mamba1+Attention bate Mamba2+Attention em híbridos — vale validar antes de comprometer

🎯 **Realidade dura:** Modelos 40M não exibem reasoning emergente, independente de dados. Posicionar como pattern-matcher / specialist coerente, não general reasoner.

**Veredito estratégico:** A próxima alavanca de qualidade não é mais arquitetural — é **dados + teacher + scheduler**. Toda mudança proposta abaixo é ortogonal à arquitetura existente, então pode ser feita incrementalmente sem refactor.

---

## 2. O Estado da Arte (Maio 2026)

### 2.1 A Mudança de Paradigma: "Quality > Quantity" para Modelos Pequenos

O consenso 2024-2026 (validado por múltiplos papers controlados) é:

> Para modelos sub-2B params, **dados curados/sintéticos** dominam **contagem de tokens**. TinyLlama (3T tokens raw web) foi obsoleto por Cosmopedia-trained 1B em MMLU/ARC com **muito menos compute**.

**Mas:** Chinchilla overshooting continua útil — Phi-3-mini (3.3T), SmolLM2-1.7B (~11T), Qwen2.5 (18T) — todos over-train muito além do Chinchilla optimal. **A reconciliação:** over-train, mas em **dados filtrados + sintéticos**, não em web crua.

### 2.2 Tokens por Parâmetro — O Compute Floor Real

| Modelo | Params | Tokens | Tokens/param | Resultado |
|--------|--------|--------|--------------|-----------|
| **Chinchilla optimal** | qualquer | 20× params | 20 | Coherence floor (não suficiente para useful) |
| TinyLlama 1.1B | 1.1B | 3T | 2,727 | Beat OPT, mas perdeu pra Phi |
| Phi-3-mini | 3.8B | 3.3T | 868 | SOTA classe Mistral 7B |
| Phi-3.5-MoE | 6.6B active | 4.9T | 742 | Bate Llama 3.1 |
| SmolLM2-1.7B | 1.7B | 11T | 6,470 | SOTA <2B |
| Gemma-2 2B | 2B | 2T | 1,000 | + distillation |
| BitNet b1.58-2B | 2B | 4T | 2,000 | Match FP16 Llama 2 |
| **NSOS hybrid_medium (atual)** | **40M** | **~537K + 44K** | **~14** | ❌ **Sub-Chinchilla** |

**Implicação para NSOS:** Para um modelo 40M ser "useful" (não só coherent), o alvo realista é **30-50B tokens** (750-1300 tokens/param), seguindo a convenção Phi. Estamos ~5 ordens de magnitude abaixo.

### 2.3 Verdades Estabelecidas Recentemente

1. **Distillation é a alavanca dominante 2025-2026** — Gemma 2/3, Qwen3 small, Llama 3.2, Minitron 4B — todos usam.
2. **WSD scheduler > Cosine** para distillation (MiniCPM showed) — permite injetar dados de qualidade no decay phase.
3. **Synthetic data não satura** — Phi-4 confirma: mais epochs em synthetic > fresh web tokens.
4. **Edu-classifier filtering** (Llama-3-70B → embedding regressor → score≥3) é o single highest-ROI 2024-2026 — adotado por Llama 3, Phi-3, FineWeb-Edu, SmolLM.
5. **40M params não mostra reasoning emergente** — emergent abilities aparecem ~10²² FLOPs (13B-class). Frame realista é "pattern-matcher coerente".

---

## 3. Gap Analysis: NSOS vs Estado da Arte

### 3.1 Comparação Lado a Lado

| Dimensão | NSOS Atual | SOTA Pequeno (SmolLM2-1.7B) | SOTA Tiny (SmolLM-135M) |
|----------|-----------|------------------------------|--------------------------|
| **Params** | 40M | 1.7B (43× maior) | 135M (3.4× maior) |
| **Vocab** | 4,827 | 32,000+ | 32,000+ |
| **Pretrain tokens** | ~537K (Wikipedia) | ~11T | ~2T |
| **Pretrain mix** | NSOS code → Wiki bruta | 60% FineWeb-Edu + 40% DCLM + Cosmopedia + Python-Edu | mesmo |
| **Teacher (distillation)** | Llama 2 7B (2023) | Llama 3.1 70B (logit) | Auto-regressive |
| **SFT mix** | distillation_bundle_v2 (~2K samples) | SmolTalk (multi-turn, distilabel) | mesmo |
| **DPO** | nenhum | UltraFeedback | nenhum |
| **LR schedule** | Cosine global | WSD (Warmup-Stable-Decay) | WSD |
| **Quality classifier** | nenhum | Edu classifier (Llama-3-70B annotated) | mesmo |
| **Synthetic data** | nenhum | Cosmopedia v2 (25B tokens Mixtral-generated) | mesmo |
| **Curriculum** | 6 fases (3 destrutivas no v7) | 3-stage clean | simpler |
| **Compute total** | ~12h CPU = ~360M FLOPs | ~1000s GPU-hours | ~100s GPU-hours |

### 3.2 Onde NSOS Está Bem Posicionado

**Decisões corretas:**
- ✅ Hybrid Mamba+Attention+MoE é **arquiteturalmente validado** (Jamba paper)
- ✅ BitNet 1.58-bit é **scientifically proven** (microsoft/bitnet-b1.58-2B-4T)
- ✅ Distillation framework já existe — só falta trocar teacher
- ✅ Edge-first stance é defensável (poucos competem nesse nicho)
- ✅ Wikipedia foundation (v7+v8) corrige o problema do código NSOS contaminating phase3

**Pontos negativos não-arquiteturais:**
- ❌ Vocab tokenizer 4,827 — específico demais, fragmenta inglês comum
- ❌ Datasets sintéticos sub-curated (raw distillation vs Magpie/Cosmopedia)
- ❌ Sem quality filtering em phase3 (Wikipedia bruta vs FineWeb-Edu filtered)
- ❌ Schedule cosine global causa exatamente os problemas que vimos
- ❌ Volume de dados insuficiente em ordens de magnitude

### 3.3 Onde NSOS Está Atrasado

**Decisões a reconsiderar:**
- ⚠️ **Mamba2 vs Mamba1 em hybrid** — Jamba paper data: Mamba1+Attention beats Mamba2+Attention em hybrids
- ⚠️ **MoE em sub-1B** — campo diz "geralmente não vale a pena" — overhead > savings (mas Phi-3.5-MoE prova exceção quando 16 experts top-2)
- ⚠️ **MoE routing** — se mantiver, mudar para **DeepSeek V3 aux-loss-free routing** (bias term)
- ⚠️ **Capacity gap teacher → student** — 175× é demais, usar TAID-style intermediate (70B → 1B → 40M)

---

## 4. Recomendações Estratégicas (Priorizadas)

### Tier 1 — Mudanças com ROI Imediato (sem mexer em arquitetura)

#### 4.1.A — Trocar o Teacher (PRIORIDADE MÁXIMA)

**Atual:** Llama 2 7B Q4_0 (2023, obsoleto)

**Substitutos recomendados:**

| Teacher | Tamanho | Forte em | Por que |
|---------|---------|----------|---------|
| **Llama 3.1 8B Instruct** | 8B | General + instruction | Sweet spot capacity para 40M student |
| Qwen 2.5 7B Instruct | 7B | Multilingual + math | 18T pretrain tokens, melhor em raciocínio |
| DeepSeek-R1-Distill-Llama-8B | 8B | Reasoning chains | Para CoT distillation |
| Mistral 7B Instruct v0.3 | 7B | Eficiência | Robust baseline |

**TAID intermediate (recomendação avançada):**
Em vez de Llama 3.1 70B → 40M direto (gap muito grande), usar pipeline:
```
Llama 3.1 70B → Llama 3.2 3B → NSOS 40M
```
Reduz capacity gap de 1750× para ~75× e ~6× sucessivamente. ICLR 2025 paper [TAID](https://arxiv.org/pdf/2501.16937) mostra ganhos significativos.

#### 4.1.B — Substituir Phase3 Curated_Text por FineWeb-Edu

**Atual:** Wikipedia bruta (5GB texto, ~537K tokens efetivos no treino atual)

**Recomendado:** **FineWeb-Edu sample-10BT** ([HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu))
- 10B tokens já pré-filtrados por Llama-3-70B classifier (score ≥ 3)
- **+24% ARC, +33%→37% MMLU vs FineWeb cru** em modelos pequenos
- License ODC-BY (commercial OK)

**Ou alternativa híbrida:** **smollm-corpus** ([HuggingFace](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus))
- 220B FineWeb-Edu + 28B Cosmopedia + 4B Python-Edu
- Pronto para drop-in
- Recipe testado em SmolLM2-135M

#### 4.1.C — Adicionar Cosmopedia v2 (Synthetic Textbooks)

**Cosmopedia v2** ([HF](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)):
- 25-28B tokens sintéticos gerados por Mixtral-8x7B
- Cobertura organizada em 145 topic clusters
- Apache 2.0 license
- Imita "textbook quality" (estratégia Phi)

**Para NSOS (40M):** usar 5-10B tokens em fase de synthetic curriculum (after Wikipedia foundation).

#### 4.1.D — Substituir Cosine LR por WSD Schedule

**Problema atual:** Cosine LR decay global causa que phases tardias (5, 6, polish) recebam LR muito baixo, exatamente o que vimos quebrar v3 e v4.

**WSD (Warmup-Stable-Decay):**
- Phase 1: warmup (5-10% dos steps)
- Phase 2: **STABLE** (60-80% dos steps em LR pico)
- Phase 3: decay para 10-20% do pico
- **Permite injetar dados de alta qualidade no decay phase** (instruction tuning, polish)
- Adotado por MiniCPM, Phi-4, vários SOTA

**Implementação:** mudança no `trainer.cpp`/`scheduler.cpp` — não afeta arquitetura.

#### 4.1.E — Trocar SFT data para Magpie + Tülu-3

**Atual:** distillation_bundle_v2 (~2K samples gerados)

**Magpie** ([paper](https://arxiv.org/abs/2406.08464), [HF](https://huggingface.co/argilla/magpie-ultra-v0.1)):
- **Zero-shot generation** de instruções a partir de modelo aligned
- Llama-3-Instruct gera 4M instruções **sem seeds**
- **Bate ShareGPT, UltraChat, Evol-Instruct, OpenHermes** quando treina Llama-3-8B-Base
- Match official Llama-3-8B-Instruct (que usou 10M+ samples + RLHF)

**Tülu-3-SFT-mixture** ([HF](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)):
- 939K samples balanceados
- Hybrid (FLAN + Persona + NoRobots + OASST)
- ODC-BY license (commercial OK)
- 2024-25 SOTA mixture

**Recipe sugerida para NSOS (sub-500M):** 250-500K examples mixing Magpie subset + FLAN-V2 sample + OpenOrca CoT + UltraChat

### Tier 2 — Mudanças Arquiteturais (Considerar Após Tier 1)

#### 4.2.A — Mamba1 vs Mamba2 (validar empiricamente)

Jamba paper ablation explicitly: **"Mamba1+Attention works better than Mamba2+Attention" em hybrid configurations**.

**Ação sugerida:** treinar 2 baselines pequenos (10M params, 1B tokens cada) — um com Mamba1 layers, outro Mamba2 — e comparar perplexity. Se Mamba1 ganhar, considerar trocar.

#### 4.2.B — MoE: Vale Manter para 40M?

**Consenso:** "MoE generally NOT worth it under 1B total params" — overhead exceeds savings.

**Mas:** Phi-3.5-MoE (6.6B active de 16 experts top-2) prova que MoE pode escalar pequeno **se config correta**.

**Decisão:** Se mantiver MoE, **trocar para DeepSeek V3 aux-loss-free routing**:
- Substitui auxiliary loss por **per-expert bias term** ajustado quando expert sobre/sub-carrega
- Elimina objective interference com LM loss
- Scaling até 256 experts em DeepSeek V3

Se complexidade de MoE estiver causando mais problemas que benefício na escala atual, considerar **dense-only** para validar pipeline antes.

#### 4.2.C — Vocabulário do Tokenizer (4,827 → 16K-32K)

**Problema atual:** Tokenizer especializado em código NSOS (4,827 tokens) — palavras inglesas comuns ficam fragmentadas.

**Custo da troca:** retreino completo do tokenizer + retreino do modelo.

**Benefício:** sequências mais curtas (mais conteúdo em seq_len=160), melhor representação de inglês comum, base para generalização para outros idiomas.

**Recomendação:** retreinar com SentencePiece ou tiktoken, vocab 32K, em corpus diverso (FineWeb-Edu + Wikipedia + parte de Stack Exchange). Investimento alto mas necessário se quiser que o modelo tenha qualidade real de geração de texto.

### Tier 3 — Avançado (Após Tier 1+2 Validados)

#### 4.3.A — On-Policy Distillation (GKD)

Uma vez que pipeline básico funcionar, considerar **Generalized Knowledge Distillation**:
- Student gera, teacher pontua
- Crítico quando capacity gap é grande (nosso caso)
- Off-policy cold start → switch para on-policy quando loss plateaus
- Combina com SOD (Step-wise On-Policy Distillation) — até 20.86% gains

#### 4.3.B — BitNet a4.8 (4-bit Activations)

Após validar BitNet b1.58 em produção, considerar upgrade:
- BitNet a4.8 ([paper](https://arxiv.org/abs/2411.04965)) adiciona 4-bit activations
- Hybrid quantization+sparsification para outlier channels
- Two-stage training: 8-bit → anneal para 4-bit
- Reduz memória de KV cache (3-bit support)

#### 4.3.C — Multilingual / PT-BR

Se algum dia quiser PT-BR:
- **Trade-off:** modelo multilingual sub-100M perde ~30-40% qualidade em inglês vs english-only
- **Recomendação:** ship english-first, retrain separate PT-BR variant
- **Datasets PT-BR:** CulturaX-PT (~36B tokens), Carolina (USP, 653M tokens, license clean), Wikipedia-PT (1.1M articles)
- **Evitar:** BrWaC (license unclear)

---

## 5. Catálogo de Datasets Recomendados

### 5.1 Pretraining (Foundation)

| Dataset | Tokens | License | URL | Uso para NSOS |
|---------|--------|---------|-----|---------------|
| **FineWeb-Edu sample-10BT** | 10B | ODC-BY | [HF](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | **Substituto direto da Wikipedia em phase3** |
| **Cosmopedia v2** | 25B | Apache 2.0 | [HF](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Synthetic curriculum phase |
| **smollm-corpus** | 250B+ mix | ODC-BY/Apache | [HF](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Recipe pronto SmolLM2 |
| **DCLM-Baseline** | 3.8T | CC-licensed | [DCLM](https://www.datacomp.ai/dclm/) | Para escalar futuro |
| **Common Pile v0.1** | ~8TB | Open license | [Eleuther blog](https://blog.eleuther.ai/common-pile/) | Substituto legal de Books3 |
| **Project Gutenberg / PG-19** | 6-11B | Public Domain | [github](https://github.com/google-deepmind/pg19) | Long-form coherence |
| **Wikipedia EN** | ~6B | CC-BY-SA | [HF](https://huggingface.co/datasets/wikimedia/wikipedia) | Spice 1-5% (já temos) |

### 5.2 Code (se relevante futuro)

| Dataset | Tokens | License | URL |
|---------|--------|---------|-----|
| **The Stack v2** | 900B-4T | Per-file (mostly permissive) | [HF](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| **Python-Edu** | 4B | ODC-BY | [HF](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (subset) |

### 5.3 Math/Reasoning

| Dataset | Tokens/Samples | License | URL |
|---------|----------------|---------|-----|
| **OpenMathInstruct-2** | 14M problems | NVIDIA Open | [paper](https://arxiv.org/abs/2410.01560) |
| **MetaMathQA** | 395K | MIT | [HF](https://huggingface.co/datasets/meta-math/MetaMathQA) |
| **GSM8K** | 8.5K | MIT | [HF](https://huggingface.co/datasets/openai/gsm8k) |

### 5.4 Instruction Tuning (SFT)

| Dataset | Samples | Method | License | URL |
|---------|---------|--------|---------|-----|
| **Magpie-Ultra v0.1** | 1M | Zero-shot Llama-3.1-405B | Llama 3 community | [HF](https://huggingface.co/datasets/argilla/magpie-ultra-v0.1) |
| **Tülu-3-SFT-mixture** | 939K | Hybrid | ODC-BY | [HF](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| **OpenHermes-2.5** | ~1M | Hybrid synthetic | MIT (mixed) | [HF](https://huggingface.co/datasets/teknium/OpenHermes-2.5) |
| **SmolTalk** | 1M+ | distilabel + Llama-3.1-405B | Apache 2.0 | [HF](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) |
| **OpenOrca** | 4M | GPT-4 augmented FLAN | MIT | [HF](https://huggingface.co/datasets/Open-Orca/OpenOrca) |
| **FLAN-V2** | 15M tasks | Human templates | Apache 2.0 (mixed) | [HF](https://huggingface.co/datasets/Muennighoff/flan) |

### 5.5 DPO/Preference (Tier 3)

| Dataset | Samples | License | URL |
|---------|---------|---------|-----|
| **UltraFeedback** | 64K prompts → 380K judgments | MIT | [HF](https://huggingface.co/datasets/openbmb/UltraFeedback) |
| **HH-RLHF (Anthropic)** | 170K | MIT | [HF](https://huggingface.co/datasets/Anthropic/hh-rlhf) |

### 5.6 PT-BR (Se Algum Dia Quiser)

| Dataset | Tokens | License | URL |
|---------|--------|---------|-----|
| **CulturaX-PT** | 36B | ODC-BY | [HF](https://huggingface.co/datasets/uonlp/CulturaX) |
| **Carolina (USP)** | 653M | Open licensed | [paper](https://arxiv.org/pdf/2303.16098) |
| **Wikipedia PT** | ~1.1M articles | CC-BY-SA | [Wikimedia dumps](https://dumps.wikimedia.org/) |

### 5.7 Cinza Legal (Pesquisa Apenas, Não Recomendado para Produção)

> **Aviso:** Esta seção documenta o que existe, não recomenda uso. NYT v. OpenAI/Microsoft preservou TODA telemetria desde 2025. Lawsuits ativos.

- **Books3** — 196k books de Bibliotik, removido após Atlantic exposé 2023. Ainda em torrents/HF mirrors. **Legalmente radioativo agora.**
- **Anna's Archive** — NVIDIA buscou acesso em 2023 ([Tom's Hardware exposé](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-accused-of-trying-to-cut-a-deal-with-annas-archive-for-high-speed-access-to-the-massive-pirated-book-haul-allegedly-chased-stolen-data-to-fuel-its-llms)), discovery em curso.
- **LibGen** — vazio em vários papers de major labs sem documentação oficial.

**Realidade:** GPT-4, Claude, Gemini quase certamente contém books copyrighted. Nenhum publica lista de dados. Llama 1 paper explicitly listou Books3, Llama 2/3 pararam de nomear sources mas mantiveram performance — faça suas inferências.

**Recomendação NSOS:** **Stick com Common Pile + FineWeb-Edu + Cosmopedia.** Lawsuit risk não compensa benefício marginal em modelo 40M. Major labs têm $bilhões em legal budget — você não.

---

## 6. Recipe Concreta para NSOS v9 (Próximo Treino Pós-v8)

Assumindo v8 valida o pipeline atual (Wikipedia funciona, modelo gera coerente):

### Fase A — Setup Pré-Treino

1. **Trocar teacher** Llama 2 7B → **Llama 3.1 8B Instruct**
2. **Baixar FineWeb-Edu sample-10BT** (~30GB)
3. **Baixar Cosmopedia v2** (25B tokens, ~75GB)
4. **Baixar Magpie-Ultra v0.1** ou subset SmolTalk (50K-100K samples)

### Fase B — Build Bundle v4

```
distillation_bundle_v4/
  data/
    phase3a_fineweb_edu.train.jsonl  (10B tokens)
    phase3b_cosmopedia.train.jsonl   (5-10B tokens)
    phase4_magpie.train.jsonl         (50K-100K samples)
    phase4b_tulu.train.jsonl          (50K samples)
  tokenizer_v3.ox3 (retrained 16K vocab — opcional, alta mudança)
```

### Fase C — Schedule (WSD em vez de Cosine)

```
phase1/2: REMOVIDAS (já fizemos no v8)
phase3a (FineWeb-Edu):
  - warmup: 100 steps
  - stable: 800 steps em peak LR
  - decay: 100 steps
phase3b (Cosmopedia synthetic):
  - stable: 400 steps
  - decay: 50 steps
phase4 (Magpie + Tülu instruction):
  - stable: 300 steps
  - decay: 50 steps
polish: 100 steps em LR baixo
```

### Fase D — Hardware

**Crucial:** v9 deve rodar em **GPU** (1050 Ti) — atual treino CPU 12h vira ~2h em GPU.

Ver `GPU_OPTIMIZATION_ANALYSIS.md` para detalhes.

### Fase E — Eval

- MMLU, ARC-c, HellaSwag, WinoGrande, PIQA (benchmarks padrão)
- Holdout próprio (já temos em `nsos_micro_suite.jsonl`)
- Comparar contra **SmolLM-135M** como baseline público

---

## 7. Roadmap de Versões Realista

### v8 (atual, em treino) — **Validar pipeline Wikipedia + instruction**
- Teacher: Llama 2 7B (legacy)
- Data: distillation_bundle_v3 (Wikipedia replacement)
- Schedule: Cosine
- Goal: provar que modelo responde coerentemente

### v9 — **First SOTA-aligned recipe** (1-2 semanas pós-v8)
- Teacher: **Llama 3.1 8B Instruct**
- Data: **FineWeb-Edu (10B) + Cosmopedia (5B) + Magpie SFT**
- Schedule: **WSD**
- Hardware: **GPU 1050 Ti**
- Goal: bater SmolLM-135M em algum benchmark

### v10 — **Architectural validation** (2-4 semanas)
- Mamba1 vs Mamba2 ablation (small models)
- DeepSeek V3 aux-loss-free MoE
- Validate BitNet b1.58 em GPU packed
- Goal: confirmar decisões arquiteturais data-driven

### v11 — **Scale up** (1-2 meses, hardware permitting)
- Vocab tokenizer 16K-32K (retreino completo)
- 50B tokens dataset
- TAID intermediate distillation (70B → 1B → 40M)
- Goal: real benchmark numbers vs SmolLM2-135M / Qwen2.5-0.5B

### v12+ — **Productionization** (3+ meses)
- Quantização GPU native (Fase 5 do GPU roadmap)
- DPO com UltraFeedback (após validar SFT)
- Eventualmente: 100M-500M params com mesma arquitetura
- Goal: NSOS-100M competitivo com SmolLM2-360M

---

## 8. Limites Honestos

### 8.1 O Que Modelos 40M Não Vão Fazer

**Não importa quanta dados você adicione:**

- ❌ Reasoning emergente (precisa ~10²² FLOPs = 13B-class)
- ❌ Chain-of-thought confiável (~10²⁴ FLOPs)
- ❌ Aritmética complexa (~10²² FLOPs)
- ❌ Conhecimento factual confiável em domínios amplos
- ❌ Compreensão de contexto longo (limite seq_len + capacity)

### 8.2 O Que Modelos 40M Bem Treinados Conseguem

- ✅ Texto coerente em domínio específico
- ✅ Pattern matching forte
- ✅ Resposta a perguntas factuais simples (basic Q&A)
- ✅ Continuação de texto plausível
- ✅ Classification tasks
- ✅ Embeddings (similaridade semântica)
- ✅ Specialist models (uma tarefa específica bem)

### 8.3 Posicionamento Estratégico Honesto

**Para o NSOS, framing realista:**

> "NSOS é um proof-of-concept de arquitetura híbrida (Mamba2+Attention+MoE+BitNet 1.58-bit) treinada via knowledge distillation. A 40M params, é um pattern-matcher coerente — não um general reasoner. O valor é a **arquitetura**: prova que Mamba+MoE+BitNet escala, e demonstra pipeline de treinamento eficiente para hardware modesto."

**Não promete:**
- ❌ "Modelo que conversa como ChatGPT"
- ❌ "Bate Llama 7B em MMLU"
- ❌ "Reasoning de papers"

**Promete:**
- ✅ "Arquitetura BitNet+Mamba+MoE funciona end-to-end"
- ✅ "Treina em CPU edge (validado) e GPU mid-range (1050 Ti)"
- ✅ "Pipeline completo de distillation funcional"
- ✅ "Pattern-matcher coerente em inglês para tarefas factuais simples"

---

## 9. Decisão Crítica de Curto Prazo

**Após v8 terminar (em algumas horas):**

Três caminhos possíveis:

### Caminho A — Validar v8 e Iterar Datasets (Recomendado)
1. Inferência em v8: confirmar que responde coerentemente
2. Se sim: planejar v9 com FineWeb-Edu + Magpie + WSD scheduler
3. Manter arquitetura, trocar dados/teacher/schedule
4. Treino em 1050 Ti
5. **Tempo:** 2-4 semanas

### Caminho B — Validar GPU Pipeline Primeiro
1. Inferência em v8 só pra confirmar coerência mínima
2. Pular pra Fases 1-3 do GPU roadmap
3. Re-treinar v9 em GPU com mesma arquitetura mas melhores datasets
4. **Tempo:** 4-6 semanas (GPU enablement é trabalho real)

### Caminho C — Architectural Refresh
1. Validar Mamba1 vs Mamba2 com ablation pequeno
2. Decidir sobre MoE (manter / dropar / aux-loss-free)
3. Considerar retreino tokenizer (decisão grande)
4. **Tempo:** 1-2 meses

**Recomendação:** **Caminho A** — extrair valor máximo do código atual com datasets melhores. GPU enablement (Caminho B) e architectural refresh (Caminho C) podem vir em paralelo ou depois. **A maior alavanca de qualidade no curto prazo é trocar teacher + dados, não arquitetura.**

---

## 10. Referências Chave

### Papers Fundamentais

- **FineWeb / FineWeb-Edu:** [arXiv 2406.17557](https://arxiv.org/html/2406.17557v1)
- **DCLM (DataComp-LM):** [arXiv 2406.11794](https://arxiv.org/abs/2406.11794)
- **Cosmopedia + SmolLM:** [HF blog](https://huggingface.co/blog/cosmopedia), [SmolLM blog](https://huggingface.co/blog/smollm)
- **Phi-3 Tech Report:** [arXiv 2404.14219](https://arxiv.org/html/2404.14219v1)
- **Phi-4:** [arXiv 2412.08905](https://arxiv.org/html/2412.08905v1)
- **Jamba (Mamba+Attn+MoE):** [arXiv 2403.19887](https://arxiv.org/abs/2403.19887)
- **Mamba 2 SSD:** [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)
- **BitNet b1.58:** [arXiv 2402.17764](https://arxiv.org/abs/2402.17764), [model](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- **DeepSeek-V3:** [arXiv 2412.19437](https://arxiv.org/pdf/2412.19437)
- **Magpie:** [arXiv 2406.08464](https://arxiv.org/abs/2406.08464)
- **Tülu 3:** [arXiv 2411.15124](https://arxiv.org/html/2411.15124v3)
- **TAID Distillation:** [arXiv 2501.16937](https://arxiv.org/pdf/2501.16937)
- **Gemma 2 (distillation):** [Tech Report](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)
- **Quality vs Quantity SLMs:** [arXiv 2411.15821](https://arxiv.org/abs/2411.15821)
- **Llama 3 Herd:** [arXiv 2407.21783](https://arxiv.org/html/2407.21783)

### Recursos Práticos

- [HuggingFace Datasets Hub](https://huggingface.co/datasets)
- [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)
- [Common Pile blog](https://blog.eleuther.ai/common-pile/)
- [DistillKit (Arcee AI)](https://github.com/arcee-ai/DistillKit)
- [Magpie GitHub](https://github.com/magpie-align/magpie)
- [distilabel (Argilla)](https://github.com/argilla-io/distilabel)

### Lawsuits & Legal Status

- [AI copyright lawsuit tracker (Oct 2025)](https://chatgptiseatingtheworld.com/2025/10/08/status-of-all-51-copyright-lawsuits-v-ai-oct-8-2025-no-more-decisions-on-fair-use-in-2025/)
- [NPR - NYT v. OpenAI ruling](https://www.npr.org/2025/03/26/nx-s1-5288157/new-york-times-openai-copyright-case-goes-forward)
- [Atlantic - Books3 exposé](https://www.theatlantic.com/technology/archive/2023/08/books3-ai-meta-llama-pirated-books/675063/)

---

## 11. Conclusão Estratégica

O NSOS está em um **ponto de inflexão produtivo**. O v8 que está terminando agora valida o pipeline básico (Mamba+MoE+BitNet com Wikipedia foundation). A arquitetura está provada (Jamba paper), o código C++ funciona (todos os gates passam), e a tese 1.58-bit é cientificamente validada (Microsoft).

**A próxima alavanca de qualidade não é arquitetural — é datasets + teacher + schedule.**

As três mudanças com maior ROI imediato:

1. **Trocar Llama 2 7B → Llama 3.1 8B Instruct** (teacher SOTA)
2. **Substituir Wikipedia bruta → FineWeb-Edu (10B) + Cosmopedia (5B)** (data quality)
3. **Cosine LR → WSD scheduler** (resolve LR decay problem)

Essas três mudanças, sem mexer em arquitetura, devem transformar a qualidade do modelo final muito mais que qualquer ajuste de curriculum que tentamos antes.

**Em paralelo, GPU enablement** (ver `GPU_OPTIMIZATION_ANALYSIS.md`) destrava capacidade de treinar com 30-50B tokens (necessário para um modelo 40M ser realmente "useful").

**Posicionamento honesto:** NSOS-40M não vai bater Llama. Vai validar que **arquitetura híbrida BitNet 1.58-bit + Mamba2 + MoE escala**, treinada em hardware edge (CPU + GPU mid-range). Esse é o valor científico real, e é defensável.

---

**Última atualização:** Maio 2026, durante treino v8 (CPU). Documento estratégico baseado em pesquisa abrangente do estado da arte. Sem modificações de código.
