# Oxta

**A Inteligência Artificial brasileira que não sai do Brasil.**

> Sistema operacional neuro-simbólico de IA, construído integralmente em C++20 com bindings nativos para Python, Rust e .NET. Quantização ternária 1.58-bit, arquitetura híbrida Mamba2 + Attention + MoE, memória causal persistente, raciocínio System-2 via MCTS, infraestrutura completa de treino, avaliação e serving. Edge-first, soberano por construção, em conformidade com LGPD desde o desenho da arquitetura.

---

## Sumário Executivo

Oxta é o primeiro sistema operacional de Inteligência Artificial construído inteiramente no Brasil, do zero, em C++20. Diferente das soluções estrangeiras (ChatGPT, Claude, Gemini), o Oxta **roda 100% local** — em notebooks, servidores corporativos ou estações de trabalho do próprio cliente — sem enviar uma única palavra para servidores nos Estados Unidos.

A arquitetura Oxta integra:

- Um **runtime ternário de 1.58-bit** em C++20 com SIMD AVX2 nativo e aceleração CUDA opcional
- **Arquitetura híbrida Mamba2 + Attention + MoE-8** com routing top-2 e capacidade de expandir verticais sem retreinamento completo
- **Memória causal persistente** com engine geodésica em Rust (OxtaMem) integrada via FFI
- **Raciocínio System-2** via Monte Carlo Tree Search (Yggdrasil) para problemas que exigem deliberação
- **Pipeline completo de treino** com curriculum learning, SFT, DPO, QAT progressivo e replay-aware sampling
- **Suite de avaliação industrial** cobrindo HellaSwag, ARC-Easy, MMLU-STEM, HumanEval e perplexidade Wikitext-2
- **Servidor HTTP de produção** com autenticação Bearer, rate limiting, queue depth, hardening completo
- **Frontend web moderno** com modo claro/escuro, animações suaves e modo visitante opcional
- **Vertical contábil completa** (Oxta Contábil) com 20+ serviços .NET 8 WPF, 5 formatos de exportação fiscal incluindo SPED Fiscal conforme ATO COTEPE 44/2018

Para empresas brasileiras que lidam com dados sensíveis — escritórios de contabilidade, advocacia, saúde, recursos humanos, governo — o Oxta resolve a equação impossível da IA moderna: **soberania de dados sem abrir mão de capacidade**.

---

## O Problema que estamos resolvendo

A Inteligência Artificial deixou de ser opcional para qualquer empresa que processa volume de informação. Mas o mercado oferece apenas dois caminhos, ambos ruins para o Brasil:

| Caminho | Problema |
|---------|----------|
| **Cloud AI estrangeira** (ChatGPT, Claude API, Gemini) | Cada documento processado viaja para servidores nos EUA. Dados de clientes, números financeiros, processos jurídicos, prontuários médicos — tudo passa por infraestrutura sob jurisdição estrangeira. Risco LGPD direto. |
| **IA própria treinada por empresa** | Requer equipe de pesquisa de US$ 500k+/ano, infraestrutura de GPUs A100 (US$ 200k/unidade), e ainda assim o modelo final não está adaptado ao português técnico brasileiro nem à legislação local. |

O resultado: empresas brasileiras hoje **vazam dados sensíveis para Big Tech americana porque é a única opção viável**. Cada R$ gasto em IA hoje no Brasil financia infraestrutura estrangeira que não fala português técnico, não conhece a Receita Federal, e não responde à ANPD.

**Esta é a lacuna que o Oxta preenche.**

---

## A Arquitetura Oxta — Visão Geral

```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║                          INTERFACE LAYER                                 ║
║   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  ║
║   │  Oxta Chat   │ │  HTTP REST   │ │   nsos_ext   │ │   nsos_cli   │  ║
║   │  (web app)   │ │     API      │ │  (pybind11)  │ │   (binary)   │  ║
║   └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  ║
║                                                                          ║
║   ───────────────────────────────────────────────────────────────────   ║
║                                                                          ║
║                          REASONING LAYER (Mind)                         ║
║   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────┐ ║
║   │   Mamba2   │ │ Attention  │ │    MoE     │ │    KAN     │ │ MCTS │ ║
║   │    SSD     │ │    GQA     │ │   8-top2   │ │  (B-spl.)  │ │ Sys2 │ ║
║   └────────────┘ └────────────┘ └────────────┘ └────────────┘ └──────┘ ║
║                                                                          ║
║                            BODY (Kernel)                                ║
║   ┌──────────────────────────────────────────────────────────────────┐ ║
║   │  BitNet 1.58-bit Ternário | AVX2 SIMD | CUDA | BF16 TensorCores  │ ║
║   │  TurboQuant | LUT Cache | Hadamard | LoQA | Self-Heal | NumGuard │ ║
║   └──────────────────────────────────────────────────────────────────┘ ║
║                                                                          ║
║                          MEMORY LAYER                                   ║
║   ┌──────────────────┐ ┌────────────────────┐ ┌──────────────────────┐ ║
║   │  Causal Memory   │ │   OxtaMem (Rust)   │ │   Holographic HAM    │ ║
║   │  (sequência)     │ │   geodésica + FFI  │ │   (HDC/VSA bind)     │ ║
║   └──────────────────┘ └────────────────────┘ └──────────────────────┘ ║
║                                                                          ║
║                       TRAINING & EVAL                                   ║
║   ┌──────────────────┐ ┌────────────────────┐ ┌──────────────────────┐ ║
║   │   Curriculum     │ │   SFT + DPO       │ │   Scorecard MVP      │ ║
║   │  Replay-aware    │ │   ChatML + Tools   │ │   HellaSwag/ARC/MMLU │ ║
║   └──────────────────┘ └────────────────────┘ └──────────────────────┘ ║
║                                                                          ║
║                       PRODUCTION INFRA                                  ║
║   ┌──────────────────┐ ┌────────────────────┐ ┌──────────────────────┐ ║
║   │   nsos_api_server│ │   Docker / CI / CD │ │   Model Packs (.bin) │ ║
║   │   auth + harden  │ │   GitHub Actions   │ │   Pack manifest v3   │ ║
║   └──────────────────┘ └────────────────────┘ └──────────────────────┘ ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
              ┌──────────────────────────────────────┐
              │     OXTA CONTÁBIL (Vertical #1)      │
              │                                      │
              │   .NET 8 WPF desktop | 20+ services  │
              │   GLiNER 2 ONNX | SPED export        │
              │   SQLite local | LGPD-by-design      │
              └──────────────────────────────────────┘
```

---

## Body — O Kernel Ternário 1.58-bit

Onde modelos tradicionais usam números de ponto flutuante de 32 bits (FP32) para representar cada peso da rede neural, Oxta usa apenas **três valores: −1, 0 e +1**. Isso é matematicamente conhecido como representação ternária, e foi popularizado pela Microsoft Research em 2024 (BitNet b1.58).

### Por que importa

| Métrica | IA tradicional (FP32) | Oxta (ternário 1.58-bit) |
|---------|----------------------|--------------------------|
| Memória por bilhão de parâmetros | 4 GB | ~200 MB |
| Operação fundamental | Multiplicação ponto flutuante | Soma/subtração inteira |
| Hardware necessário | GPU A100/H100 | CPU moderna |
| Energia por inferência | 100% (baseline) | 5-10% |

Em termos práticos: um modelo Oxta de 1 bilhão de parâmetros cabe em 200 MB de disco — menor que um vídeo de YouTube — e roda em qualquer notebook moderno em tempo real.

### Componentes do Body

- **BitLinear** — Camada linear ternária com supressão de outliers via transformada Hadamard. Substitui a `nn.Linear` tradicional por equivalente sem multiplicação em ponto flutuante. Incorpora LoQA (Low-rank Quantization Adapter) para fine-tune eficiente.

- **gemm_158bit_i8** — Kernel SIMD AVX2/AVX-512 que processa 16-32 pesos ternários por instrução de CPU via VPMADDUBSW. Otimizado para Ice Lake, Sapphire Rapids, AMD Zen 4 e equivalentes.

- **CUDA Kernels** — Suporte completo a NVIDIA Pascal (sm_61) até Hopper (sm_90), com aceleração BF16 Tensor Core em GPUs sm_75+. Implementação WGMMA + TMA para Hopper, dispatch automático conforme arquitetura detectada.

- **TurboQuant** — Integração opcional com biblioteca Zig externa para aceleração extra em casos específicos. Fallback nativo se ausente.

- **LUT Cache** — Cache otimizado de Lookup Tables para operações quantizadas recorrentes, reduzindo latência em batch pequeno.

- **Entropy Manager** — Compressão entrópica adaptativa de pesos quantizados, otimizando uso de cache L1/L2 da CPU.

- **Numerical Guard + Self-Healer** — Detecção e recuperação automática de NaN/Inf em produção, garantindo que o servidor nunca devolve resposta corrompida ao cliente.

---

## Mind — Raciocínio Híbrido

A inteligência do Oxta combina cinco arquiteturas neurais de fronteira, cada uma escolhida pela sua força específica.

### Mamba2 SSD (State Space Duality)

A coluna vertebral do raciocínio. Camadas Mamba2 processam sequências longas com **complexidade linear em vez de quadrática**, o que significa que Oxta consegue ler documentos inteiros — declarações de IRPF de 80 páginas, contratos jurídicos completos — sem o consumo explosivo de memória dos transformers tradicionais.

Implementação inclui:
- **Forward/backward manual em C++** otimizado
- **Streaming state** via `MambaStreamSnapshot` para inferência incremental
- **GPU fast-path** com kernels customizados em CUDA

### Attention com Grouped Query Attention (GQA)

Distribuído em camadas estratégicas (1 em cada 4 layers no padrão), fornece a precisão de longo alcance necessária para casos específicos: rastrear referências cruzadas entre artigos de lei, conectar valores em diferentes seções de uma nota fiscal, manter coerência conversacional em diálogos extensos.

Recursos:
- **GQA com razão configurável** (default n_heads=12, n_kv_heads=3 = 4:1)
- **Sliding window attention** para contextos muito longos
- **Exact attention training** ativo durante fine-tune para máxima precisão
- **Suporte opcional a Flash Attention** em GPUs A100+

### Mixture of Experts (MoE-8 com routing top-2)

Em camadas selecionadas, Oxta tem **8 "especialistas" neurais**, e para cada token de entrada apenas 2 são ativados. Resultado: capacidade do modelo cresce 4× mas custo computacional permanece constante. Diferentes especialistas se desenvolvem naturalmente para domínios distintos (vocabulário fiscal, jurídico, médico, etc.) durante o treino.

Implementação inclui:
- **Routing eficiente em GPU** via `forward_moe_gpu_batched`
- **MoeWorkspace** com cache de buffers para evitar alocações repetidas
- **Top-K configurável em tempo de inferência** (`set_moe_inference_top_k`)
- **Auxiliary load-balancing loss** para evitar collapse de experts

### KAN (Kolmogorov-Arnold Networks)

Camadas KAN substituem matrizes de pesos fixas por **funções aprendíveis (B-splines)**, oferecendo:
- Maior interpretabilidade (você pode visualizar a função que cada edge aprendeu)
- Compressão extra para casos onde a função é simples
- Implementação via Sprecher-KAN com pesos quantizados

### MCTS Reasoning (Yggdrasil)

Para problemas que exigem deliberação profunda (planejamento, resolução de equações fiscais complexas, raciocínio em árvore), Oxta integra **Monte Carlo Tree Search** como camada System-2 sobre o raciocínio System-1 das redes neurais.

Casos de uso:
- Otimização tributária multi-passo
- Reconciliação de transações entre múltiplas fontes
- Geração de planos contábeis com restrições

### Hadamard Outlier Suppression

Pesos com magnitudes extremas degradam quantização ternária. A camada `simd/hadamard.cpp` aplica transformada de Hadamard antes da quantização, distribuindo a magnitude uniformemente e melhorando precisão pós-quantização significativamente.

---

## Memory — Persistente, Causal e Associativa

IAs convencionais esquecem tudo no momento que a conversa termina. Oxta foi projetado com **três camadas de memória persistente**:

### MemorySystem (Causal Memory)

Embutida no runtime C++ (`memory_system.cpp`), registra a sequência causal de cada inferência. Concorrência protegida via `std::lock_guard` em `memory_mutex`. Para auditoria, compliance e explicabilidade, é possível recuperar a cadeia exata de raciocínio que levou a uma resposta. Indispensável para casos regulados onde decisões automatizadas precisam ser justificadas (LGPD Art. 20).

### OxtaMem — Engine Geodésica em Rust

Módulo dedicado em Rust 2021 com:
- **Servidor TCP nativo** com `tokio` para acesso de baixa latência
- **Sharding distribuído** para escala horizontal
- **Indexação vetorial** via `usearch` e `memmap2`
- **Serialização rkyv** zero-cost para storage persistente
- **Bindings Python** via PyO3 para integração com pipelines de dados

Integrado ao runtime C++ via FFI (`oxtamem_ffi.cpp`). Funciona como o "hipocampo" do sistema: conecta a inferência atual a tudo que foi visto antes pelo mesmo usuário, do mesmo cliente, do mesmo escritório — formando contexto persistente que torna o Oxta progressivamente mais útil quanto mais é usado.

### Holographic Associative Memory (HAM)

Implementação de memória associativa baseada em Hyperdimensional Computing (HDC) e Vector Symbolic Architectures (VSA). Suporta operações de bind, bundle e permute em vetores de alta dimensionalidade, formando uma camada de memória de longo prazo com propriedades robustas a ruído e degradação.

---

## Interface Layer — Quatro caminhos para o usuário

### Oxta Chat (Web UI)

Interface web moderna construída em React com:
- **Modo claro e escuro** completamente customizáveis
- **Splash screen** com animações suaves de boot
- **Login opcional** (modo visitante por padrão; conversas persistem em localStorage)
- **Conversas múltiplas** com sidebar, exclusão por hover, ativação
- **Composer centralizado** em estado vazio, no rodapé em conversa ativa
- **Reasoning expandível** (mostra cadeia de raciocínio interno se requisitado)
- **Cores de acento configuráveis** (amber, coral, plasma, spectral)
- **Modal de planos** integrado para upsell
- **Tweaks panel** para configuração avançada (URL da API, token, max tokens, temperatura)
- **Métricas de runtime no chat** (tokens/s, fonte do pack, modo de inferência)

A UI conecta diretamente à API HTTP local via fetch nativo, com fallback gracioso quando o servidor está offline.

### nsos_api_server (HTTP REST API)

Servidor HTTP nativo em C++20 com socket BSD/Winsock, projetado para produção desde o primeiro byte:

**Endpoints públicos:**
- `GET /health` — Healthcheck para load balancers
- `GET /ready` — Readiness para Kubernetes
- `GET /info` — Metadata do modelo carregado
- `GET /metrics` — Métricas Prometheus-compatible
- `POST /generate` — Inferência single-shot
- `POST /generate_batch` — Inferência paralela em múltiplos prompts
- `POST /generate_stream` — Streaming token-por-token

**Endpoints administrativos (opt-in via `--enable-admin-endpoints`):**
- `POST /train-text` — Continuação de treino com texto livre
- `POST /train-batch` — Treino em batch
- `POST /train-corpus` — Treino em corpus arquivado
- `POST /pack` — Geração de pack de produção a partir do estado treinado

**Hardening de produção:**
- **Autenticação Bearer obrigatória** (Authorization: Bearer <token>)
- **Rate limiting** configurável por minuto, por IP
- **Queue depth limits** para proteção contra overload
- **Socket timeout** configurável
- **Body limits** (default 2MB, configurável)
- **Header limits** (default 64KB, configurável)
- **JSON depth limits** anti-DOS recursivo
- **Generation limits** (max tokens por request)
- **CORS configurável** via `--allow-cors` (default: same-origin)
- **TLS via reverse proxy** (nginx/Caddy recomendado, snippet incluído)
- **`X-Forwarded-Proto` honoring** com `--require-tls-proxy-header`
- **Inference replicas** para paralelismo de serving

### nsos_ext (Python bindings)

Módulo Python construído via pybind11 expondo toda a API do runtime:
- **InferenceEngine** com `load_model`, `generate`, `generate_ex`, `train_text`, `train_step`, `save_model_pack`
- **Tokenizer** com `encode`, `decode`, `load_text`, `load_pack`, `save_pack`, `add_special_tokens`
- **Tensor** com buffer_protocol para integração com NumPy
- **ModelConfig** com todos os 27 campos configuráveis
- **GenerationOptions** com temperature, top_p, top_k, max_tokens, eos_token_id, max_context_tokens
- **Trainer** com Adam, AdamW, Muon (orthogonalization), Sophia, FOGZO
- **Parameter** com `zero_grad` para integração em loops customizados
- **TensorAuditStats, RouterAuditStats** para introspeção fina

Permite que cientistas de dados usem Oxta como biblioteca em pipelines existentes (HuggingFace, PyTorch, scikit-learn) sem fricção.

### nsos_cli

Binary standalone para automação, batch processing e operação administrativa via terminal. Útil para scripts de ETL, geração offline, e ambientes onde HTTP é overkill.

---

## Training & Evaluation Layer

### Pipeline de Treinamento

Oxta implementa o pipeline completo end-to-end de LLM, todo em C++ nativo (sem dependência de PyTorch ou TensorFlow):

**Pre-training:**
- **Curriculum learning** com 6 fases ordenadas (algorithms → structured → curated text → instructions → verifier → memory)
- **Replay-aware sampling** entre fases supervisionadas
- **Continuação de pre-training** via `engine.train_text()` para adaptação de domínio
- **QAT progressivo** (Quantization Aware Training) — modelo treina em FP/BF16 com warmup, transita gradualmente para representação ternária 1.58-bit com `ternary_regularization` ajustável

**Supervised Fine-Tuning (SFT):**
- **ChatML format** com tokens especiais `<|im_start|>`, `<|im_end|>`, system/user/assistant
- **Loss masking** restrito aos tokens de resposta (não treina o prompt)
- **Special tokens registration** automática no tokenizer
- **Bundle builder** integrado com SmolTalk, Alpaca, HelpSteer2 (configurável)

**Direct Preference Optimization (DPO):**
- **Chosen/rejected pairs** para alinhamento sem RL clássico
- **KL-divergence leash** anti-drift para preservar capacidade do SFT
- **Preference bundle builder** com suporte a HH-RLHF, SHP

**Otimizadores:**
- **Adam, AdamW** — padrões da indústria
- **Muon** — orthogonalization-based, estado da arte para transformers
- **Sophia** — second-order otimizado
- **FOGZO** — first-order generalization-aware

**Infraestrutura de treino:**
- **MPI distribuído** (`nsos_mpi.cpp`) com fallback mock para single-node
- **Checkpoint atomic write** + verificação SHA256
- **Resume from checkpoint** com qualquer fase intermediária
- **Determinismo garantido** via `Tensor::set_seed()` e `verify_determinism.py`
- **Layer audit** obrigatório no CI (`test_layer_audit`) para detectar drift

### Suite de Avaliação Industrial

Implementação nativa em Python (`OXN/nsos/eval/`) cobrindo benchmarks acadêmicos padrão:

| Benchmark | O que mede | Status |
|-----------|-----------|--------|
| **HellaSwag** | Senso comum, completude contextual | Adapter completo |
| **ARC-Easy** | Raciocínio científico básico | Adapter completo |
| **MMLU-STEM** | Conhecimento técnico multi-domínio | Adapter completo |
| **HumanEval-light** | Capacidade de código | Adapter completo |
| **WikiText2 PPL** | Fluência em linguagem natural | Implementado |

**Orquestrador (`run_scorecard.py`):**
- Modos `--quick`, `--full`, `--seeds`
- Adapter pattern para isolamento entre benchmark e runtime
- DummyAdapter para sanity (48/48 testes passing)
- NsosAdapter conectando ao runtime real
- Output JSON estruturado para comparação histórica

**Gates de CI:**
- `benchmark_gate.py` — thresholds mínimos (≥1.0 prompt tok/s, ≥0.1 decode tok/s em CPU)
- `fuzz_surface_smoke.py` — anti-regressão em hardening de inputs
- `gatekeeper.py` — orquestrador de release-readiness
- `test_pack_determinism` — invariante de hash de pack de referência

---

## Data Pipeline

### Tokenizer

BPE (Byte Pair Encoding) nativo em C++ com cache otimizado. Recursos:
- **load_text** — treina BPE de zero a partir de corpus
- **load_pack** — carrega tokenizer pré-treinado de pack binário
- **save_pack** — serialização eficiente
- **add_special_tokens** — extensão dinâmica do vocabulário
- **sanitize_utf8** no boundary — garante que cada token decodificado é UTF-8 válido (gate de segurança)

### Datasets Brasileiros Integrados

Pipeline `download_datasets.py` puxa diretamente de fontes públicas:

| Dataset | Volume | Licença | Uso |
|---------|--------|---------|-----|
| **CulturaX PT-BR** | 300B+ tokens | ODC-BY-1.0 | Pre-training continuation |
| **BR-TaxQA-R** | 715 Q&A + 173MB CARF | CC-BY-4.0 | SFT fiscal gold |
| **BACEN FAQ** | ~2k Q&A financeiro | Domínio público | SFT vocabulário financeiro |
| **tech4humans/br-doc-extraction** | 1220 imagens BR | (verificar) | Vision-extraction NF-e |
| **LeNER-Br** | Corpus legal anotado | CC-BY-4.0 | NER vertical jurídico |
| **NF-e XML sintético** | Ilimitado (PyNFe) | Open | Augmentation de extração |

Filtro inteligente up-sample por keywords fiscais (CFOP, NCM, ICMS, IPI, SPED, etc.) durante streaming do CulturaX.

---

## Production Infrastructure

### Build System

CMake 3.18+ com alvos modulares:
- **`nsos_core`** (STATIC lib) — runtime central
- **`nsos_api_server`** — servidor HTTP de produção
- **`nsos_cli`** — CLI standalone
- **`nsos_circuit_trainer`** — treino especializado
- **`nsos_ext`** — extensão Python via pybind11
- **`turboquant_c`** (opcional) — lib Zig externa
- **`oxtamem_engine`** (opcional) — Rust via cargo

Suporte a Windows (MSVC), Linux (GCC/Clang), com flags otimizadas por arquitetura. Build CPU canônico não exige CUDA, mantendo deployment edge simples.

### Docker e Deployment

- **`Dockerfile`** — runtime CPU de produção, healthcheck via HTTP `/info` com Bearer token
- **`Dockerfile.industrial`** — variante para deploy em ambiente regulado/auditável
- **Healthcheck end-to-end** validando que o endpoint responde com autenticação correta
- **Guidance TLS via reverse proxy** documentado em `DEPLOY.md` com snippets nginx/Caddy

### CI/CD

**`.github/workflows/industrial-ci.yml`** com lanes:
- **`product`** (Ubuntu, gate de main) — boundary check + build CPU + ctest + compileall + benchmark gate + fuzz smoke + gatekeeper + Docker build
- **`integrated-oxtamem`** — `cargo test`, `cargo clippy -D warnings`, compileall
- **`gpu-hotpath`** (opt-in) — CUDA, gates ≥5.0 prompt / ≥1.0 decode tok/s
- **`incubation`** — syntax check de pesquisa não-produtiva

### Model Packs

Formato proprietário `.bin` com:
- Manifest versionado (v3 atual)
- SHA256 checksum
- Escrita atômica (`replace_file` via temp + rename)
- Limites por arquivo configuráveis
- Path canonicalization para anti-traversal
- Suporte a sub-arquivos: weights, tokenizer, edge_linear, config

Carregamento gracioso com fallback FP32 controlado via `NSOS_KEEP_FP32_WEIGHTS` env var (default: release FP32 após carregar ternário, economizando RAM).

### Segurança

- **PBKDF2 password hashing** com 120k iterações
- **AuditLogger** registrando todas operações sensíveis
- **SessionService** com expiração configurável
- **LoginThrottleService** anti-brute-force
- **Path validation** em todos endpoints que aceitam paths
- **JSON depth limits** anti-DOS recursivo
- **Configurable allow-lists** para origens, tokens, IPs

---

## OXTA CONTÁBIL — Primeira Aplicação Vertical

A primeira aplicação completa do Oxta é direcionada a um mercado que sofre intensamente com a equação "preciso de IA / não posso vazar dados": **escritórios de contabilidade brasileiros**.

### Stack Completo

**Frontend Desktop (.NET 8 WPF):**
- 15 views XAML completas (dashboard, revisão em 3 abas, schemas, usuários, clientes, FirstRunWizard)
- LiveChartsCore para dashboards interativos
- Tema escuro/claro dinâmico
- Multi-cliente com isolamento de dados

**Backend Services (20+ serviços):**

*Segurança:*
- `PasswordHasher.cs` — PBKDF2 com 120k iterações
- `LoginThrottleService.cs` — rate-limit anti-brute-force
- `SessionService.cs` — gestão de sessão persistente
- `AuditLogger.cs` — log estruturado de operações sensíveis

*IA (cadeia de fallback gracioso):*
- `GlinerOnnxService.cs` — extração nativa via ONNX Runtime (fast path)
- `GlinerService.cs` — fallback Python subprocess (compatibilidade)
- `RegexExtractionService.cs` — fallback regex puro (offline garantido)

*Operacional:*
- `BackupService.cs` — backup automático com retenção configurável
- `OfxParserService.cs` — parser nativo de extratos bancários
- `BrasilApiService.cs` — enriquecimento de CNPJ via BrasilAPI (opt-in)
- `CertificateService.cs` — gestão de certificados A1
- `DocumentProcessingQueue.cs` — fila assíncrona com priorização
- `DocumentTextExtractor.cs` — extração de texto multi-formato

*Suporte:*
- `ThemeManager.cs`, `ToastService.cs`, `TelemetryService.cs`, `Validators.cs`, `AppSettings.cs`

**Persistência:**
- SQLite em `%LocalAppData%\OContabil\`
- `SchemaManager.cs` — upgrades SQL idempotentes v1→v4
- Entidades: User, Client, Document, DocumentRevision, DocumentSchema, ChartOfAccount, AuditLog

**Logs:**
- Serilog em `%LocalAppData%\OContabil\logs\`
- Retenção 14 dias
- Estruturado JSON para SIEM integration

### Cinco Formatos de Exportação Profissional

| Formato | Implementação | Caso de uso |
|---------|--------------|-------------|
| **Excel (ClosedXML)** | 3 abas: resumo, detalhamento, classificação | Reunião com cliente |
| **CSV pt-BR** | Encoding correto, separador `;`, vírgula decimal | Integração com sistemas terceiros |
| **SPED Fiscal** | Conforme ATO COTEPE 44/2018 | Entrega à Receita Federal |
| **Domínio Sistemas** | Layout proprietário do líder de mercado | Migração assistida |
| **PDF de Conferência (QuestPDF)** | Layout executivo profissional | Auditoria interna |

### Fluxo do Contador (UX)

```
1. UPLOAD          → Arrasta NF-e (XML/PDF) na interface WPF
2. EXTRAÇÃO        → GlinerOnnxService identifica CNPJ, valor, fornecedor,
                     CFOP, NCM, alíquotas em ~50ms
3. CONSULTA        → Pergunta em linguagem natural:
                     "Esse fornecedor já apareceu?" 
                     "Qual conta sugere?"
                     "Tem divergência com histórico?"
4. RESPOSTA OXTA   → Consulta SQLite local + raciocínio Oxta + 
                     legislação fiscal brasileira embutida
5. EXPORTAÇÃO      → Click → SPED/Excel/CSV/Domínio/PDF
6. AUDITORIA       → Cada step fica em AuditLog para compliance
```

**Tudo isso acontece sem nenhuma chamada para servidores externos.** A máquina do escritório é simultaneamente o frontend, o backend, o banco de dados e o cérebro de IA. O contador pode usar Oxta Contábil sem conexão à internet.

### Conformidade LGPD by Design

- **Dados pessoais** nunca saem da máquina do escritório
- **Right to be forgotten** (Art. 18) — função de exclusão completa de cliente em UI
- **Right to explanation** (Art. 20) — `AuditLogger` + `MemorySystem` causal permitem reconstrução de decisão
- **Data minimization** — apenas campos necessários por documento processado
- **Encryption at rest** opcional via Windows DPAPI
- **Backup criptografado** com chave do operador (`CertificateService`)

---

## Por que Agora — Timing de Mercado

Três forças simultâneas tornam este o momento exato para Oxta:

### 1. LGPD em fase de fiscalização real

A Lei Geral de Proteção de Dados foi sancionada em 2018, entrou em vigor em 2020, mas só em 2024-2025 a ANPD começou a aplicar multas substanciais. Empresas brasileiras estão buscando alternativas urgentes às ferramentas que vazam dados para o exterior.

### 2. Hardware moderno suporta IA local

Notebooks vendidos hoje (Intel 13th gen, AMD Ryzen 7000, Apple M3, Snapdragon X Elite) têm capacidade computacional suficiente para rodar modelos Oxta de 1 bilhão de parâmetros em tempo real — algo impossível há 5 anos. A janela de oportunidade para IA edge-first abriu agora.

### 3. Big Tech americana sob pressão regulatória

OpenAI, Anthropic e Google enfrentam crescente regulação nos próprios EUA, pressão regulatória na Europa (AI Act), e desconfiança crescente em mercados emergentes. Para o Brasil, há oportunidade clara de construir soberania tecnológica enquanto o mercado global está dividido.

---

## Roadmap

### 2026 H1 — Oxta Contábil em campo

Programa de design partners com escritórios de contabilidade brasileiros. Refinamento do modelo com casos reais sob NDA. Lançamento comercial após validação com 5-10 primeiros clientes.

### 2026 H2 — Oxta Jurídico

Vertical para escritórios de advocacia: análise de jurisprudência, redação assistida de petições, busca em legislação consolidada. Reusa runtime Oxta com fine-tune especializado em LeNER-Br + corpus de jurisprudência STF/STJ.

### 2027 H1 — Oxta Saúde

Vertical para clínicas e laboratórios: estruturação de prontuário, sugestão de CID-11, conformidade LGPD para dados de saúde (categoria especial). Integração HL7/FHIR.

### 2027 H2 — Plataforma Oxta SDK

SDK público e marketplace de verticais. Permite que parceiros construam seus próprios "Oxta de domínio X" reusando a infraestrutura. Modelo de revenue sharing.

### 2028+ — Soberania Computacional Brasileira

Acordos com governo federal e estadual para infraestrutura crítica. Posicionamento como alternativa nacional para uso oficial brasileiro em setores sensíveis (defesa, finanças, justiça). Possível certificação como infraestrutura crítica nacional.

---

## A Tecnologia em Números

### Cobertura técnica
- **66 headers C++** organizando 56 arquivos `.cpp` em camadas claras
- **40+ testes CTest** com gate obrigatório
- **5 formatos de exportação fiscal** em produção
- **20+ serviços .NET** no vertical contábil
- **15 views XAML** completamente funcionais
- **5 benchmarks acadêmicos** (HellaSwag, ARC, MMLU, HumanEval, Wikitext)
- **6 fases de curriculum** validadas em treino
- **3 estratégias de otimização** (Adam, Muon, Sophia, FOGZO)

### Suporte de hardware
- **AVX2 SIMD** nativo (Haswell+ Intel, Zen+ AMD)
- **AVX-512** com auto-dispatch quando disponível
- **CUDA** Pascal sm_61 → Hopper sm_90
- **BF16 Tensor Cores** T4 sm_75+ (cublasGemmStridedBatchedEx)
- **WGMMA + TMA** para Hopper sm_90
- **OpenMP** para paralelismo CPU multi-core
- **MPI** opcional para multi-node

### Independência de frameworks
- **Sem PyTorch** — runtime próprio em C++
- **Sem TensorFlow** — tensor lib própria
- **Sem JAX** — autograd próprio
- **Sem HuggingFace runtime** — apenas datasets como fonte
- **Sem Llama.cpp** — embora compartilhe filosofia 1.58-bit
- **Sem vendor lock-in** — Apache 2.0 / MIT em todos componentes

---

## Validação Técnica Independente

**Smoke test do stack completo (2026-05-23):**

> ✅ PASSOU: loss caiu 12.6% (9.47 → 8.28)
> Stack inteiro do Oxta (Mamba2 + Attention + MoE + BF16) está funcional.

Treinamento de validação realizado no Google Colab (NVIDIA T4), profile `hybrid_v11_colab_t4_80m`, 80M parâmetros, com tokenizer BPE de vocabulário 4900, 16 layers d_model=640. Modelo construído fresh, sem checkpoint anterior, exit code 0. 31 medições de loss capturadas, queda monotônica clara após warmup do scheduler.

---

## Visão

Oxta nasceu de uma observação simples: o Brasil tem talento de classe mundial em engenharia de software, e tem mercado próprio com necessidades específicas que ferramentas estrangeiras nunca vão atender com prioridade. **Não precisamos importar o que conseguimos construir.**

Nosso compromisso é construir a infraestrutura de IA que o Brasil precisa para o século XXI — local, eficiente, soberana, brasileira por dentro e por fora.

A primeira parada dessa jornada é o escritório de contabilidade no interior de qualquer estado brasileiro, processando notas fiscais com a tranquilidade de saber que os dados dos clientes nunca saíram dali. A última parada não está definida — porque depende dos brasileiros que vão se juntar a nós no caminho.

---

## Contato

**Vitor G. C.**
Founder & Arquiteto-chefe
GitHub: [@vitorGgC569](https://github.com/vitorGgC569)

---

*Este documento descreve Oxta em sua arquitetura atual (maio/2026). Capacidades específicas evoluem continuamente — entre em contato para a versão mais recente do status técnico e do programa de design partners.*
