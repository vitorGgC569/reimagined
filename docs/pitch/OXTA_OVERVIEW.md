# Oxta

**A Inteligência Artificial brasileira que não sai do Brasil.**

---

## Sumário Executivo

Oxta é o primeiro sistema operacional de Inteligência Artificial construído inteiramente no Brasil, do zero, em C++. Diferente das soluções estrangeiras (ChatGPT, Claude, Gemini), o Oxta **roda 100% local** — em notebooks, servidores corporativos ou estações de trabalho do próprio cliente — sem enviar uma única palavra para servidores nos Estados Unidos.

Para empresas brasileiras que lidam com dados sensíveis — escritórios de contabilidade, advocacia, saúde, recursos humanos, governo — o Oxta resolve a equação impossível da IA moderna: **soberania de dados sem abrir mão de capacidade**.

A primeira aplicação vertical, **Oxta Contábil**, transforma a forma como escritórios brasileiros processam Notas Fiscais Eletrônicas, declarações SPED, extratos OFX e legislação fiscal — tudo na máquina do próprio escritório, em conformidade total com a LGPD por construção.

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

## A Solução: a Arquitetura Oxta

Oxta foi desenhado a partir de uma única premissa: **a IA do cliente brasileiro deve viver na máquina do cliente brasileiro.**

Para isso ser real, a arquitetura precisou ser repensada do zero. Não bastava pegar um modelo americano e rodar localmente — modelos como GPT-4 exigem clusters de GPUs que custam milhões. Oxta nasceu com **eficiência radical de hardware** como princípio fundador.

### As quatro camadas do Oxta

Inspirados na unidade entre forma e função dos sistemas naturais, organizamos Oxta em quatro camadas integradas:

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  ┌────────────┐  ┌────────────┐  ┌────────────┐             ║
║  │   BODY     │  │   MIND     │  │  MEMORY    │             ║
║  │            │  │            │  │            │             ║
║  │  Kernel    │  │ Raciocínio │  │  Memória   │             ║
║  │  Ternário  │  │  Híbrido   │  │  Causal +  │             ║
║  │  1.58-bit  │  │   Mamba    │  │  OxtaMem   │             ║
║  │   (C++)    │  │  + Atten   │  │   (Rust)   │             ║
║  │            │  │  + MoE     │  │            │             ║
║  └────────────┘  └────────────┘  └────────────┘             ║
║          │              │              │                    ║
║          └──────────────┼──────────────┘                    ║
║                         │                                   ║
║              ┌──────────▼──────────┐                        ║
║              │     INTERFACE       │                        ║
║              │                     │                        ║
║              │  Oxta Chat (web)    │                        ║
║              │  HTTP API (REST)    │                        ║
║              │  Python Bindings    │                        ║
║              │  CLI                │                        ║
║              └─────────────────────┘                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Body — O Kernel Ternário 1.58-bit

Onde modelos tradicionais usam números de ponto flutuante de 32 bits (FP32) para representar cada peso da rede neural, Oxta usa apenas **três valores: −1, 0 e +1**. Isso é matematicamente conhecido como representação ternária, e foi popularizado pela Microsoft Research em 2024 (BitNet b1.58).

**Por que importa:**

| Métrica | IA tradicional (FP32) | Oxta (ternário 1.58-bit) |
|---------|----------------------|--------------------------|
| Memória por bilhão de parâmetros | 4 GB | ~200 MB |
| Operação fundamental | Multiplicação ponto flutuante | Soma/subtração inteira |
| Hardware necessário | GPU A100/H100 | CPU moderna |
| Energia por inferência | 100% (baseline) | 5-10% |

Em termos práticos: um modelo Oxta de 1 bilhão de parâmetros cabe em 200 MB de disco — menor que um vídeo de YouTube — e roda em qualquer notebook moderno em tempo real. A multiplicação por −1, 0 ou +1 vira uma operação tão simples que pode ser feita em circuitos lógicos básicos, sem precisar de unidades especializadas de ponto flutuante.

Isso é construído sobre nosso runtime C++20 nativo com instruções AVX2 SIMD otimizadas, processando 16-32 pesos ternários por instrução de CPU. **Sem PyTorch, sem TensorFlow, sem dependências de framework americano.**

### Mind — Raciocínio Híbrido

A inteligência do Oxta combina três arquiteturas neurais de fronteira, cada uma escolhida pela sua força específica:

**Mamba2 (State Space Duality)** — A coluna vertebral. Camadas Mamba2 processam sequências longas com complexidade linear em vez de quadrática, o que significa que Oxta consegue ler documentos inteiros — declarações de IRPF de 80 páginas, contratos jurídicos completos — sem o consumo explosivo de memória dos transformers tradicionais.

**Attention (Grouped Query Attention)** — Distribuído em camadas estratégicas, fornece a precisão de longo alcance necessária para casos específicos: rastrear referências cruzadas entre artigos de lei, conectar valores em diferentes seções de uma nota fiscal, manter coerência conversacional.

**Mixture of Experts (MoE-8 com routing top-2)** — Em camadas selecionadas, Oxta tem 8 "especialistas" neurais, e para cada token de entrada apenas 2 são ativados. Resultado: capacidade do modelo cresce 4× mas custo computacional permanece constante. Diferentes especialistas se desenvolvem naturalmente para domínios distintos (vocabulário fiscal, jurídico, médico, etc.) durante o treino.

Essa combinação é chamada **arquitetura Jamba híbrida**, e nossa implementação é uma das primeiras nativas para CPU em produção mundial.

### Memory — Memória Persistente e Causal

IAs convencionais esquecem tudo no momento que a conversa termina. Oxta foi projetado com duas camadas de memória persistente:

**Memória Causal** — Embutida no runtime C++, registra a sequência causal de cada inferência. Para auditoria, compliance e explicabilidade, é possível recuperar a cadeia exata de raciocínio que levou a uma resposta.

**OxtaMem** — Engine de memória semântica/geodésica escrita em Rust, otimizada para armazenamento e recuperação vetorial. É como o "hipocampo" do sistema: conecta a inferência atual a tudo que foi visto antes pelo mesmo usuário, do mesmo cliente, do mesmo escritório — formando contexto persistente que torna o Oxta progressivamente mais útil quanto mais é usado.

### Interface — Quatro caminhos para o usuário

- **Oxta Chat** — Interface web moderna, animações suaves, modo claro/escuro, planos visuais, modo visitante opcional. Pode ser embarcada em qualquer aplicação corporativa via WebView2 ou iframe.
- **HTTP API REST** — Para integração com sistemas existentes (ERP, contabilidade, CRM). Autenticação por token, hardening de produção, endpoints públicos para inferência e admin para treino.
- **Python Bindings (nsos_ext)** — Para cientistas de dados e desenvolvedores que querem usar Oxta como biblioteca em pipelines existentes.
- **CLI** — Para automação, batch processing e scripts de operação.

---

## Primeira Aplicação Vertical: Oxta Contábil

A primeira aplicação completa do Oxta é direcionada a um mercado que sofre intensamente com a equação "preciso de IA / não posso vazar dados": **escritórios de contabilidade brasileiros**.

### O contexto

O contador brasileiro processa centenas de documentos por dia — NF-e, NFS-e, OFX bancários, comprovantes de despesa, declarações SPED, guias DARF — cada um com informação fiscal sensível de múltiplos clientes. Hoje, esse trabalho é feito majoritariamente manual ou com ferramentas de OCR limitadas que exigem revisão humana intensa.

A tentação de usar ChatGPT para "ajudar a categorizar essa nota" é enorme — mas significa, na prática, copiar dados fiscais do cliente para um servidor americano que não está sob jurisdição brasileira. **É violação direta da LGPD**, e a ANPD começou a aplicar multas em 2025.

### O que Oxta Contábil faz

Oxta Contábil é a fusão entre o runtime Oxta (cérebro local) e o motor de extração de entidades GLiNER 2 (especialista em documentos fiscais brasileiros), tudo encapsulado numa interface desktop nativa em .NET 8 WPF.

**Fluxo típico:**

1. **Upload de documento** — O contador arrasta uma NF-e (XML ou PDF) para a interface
2. **Extração automática** — GLiNER 2 identifica CNPJ, valores, fornecedor, código fiscal, alíquotas, em ~50ms
3. **Análise conversacional** — O contador pergunta em linguagem natural:
   - *"Esse fornecedor já apareceu antes para esse cliente?"*
   - *"Qual conta contábil sugere para essa despesa?"*
   - *"Há alguma divergência entre o valor declarado e o histórico?"*
4. **Resposta com base em dados locais** — Oxta consulta o banco de dados local do escritório (todos os clientes, todas as notas, todo histórico) e responde com fundamento em legislação fiscal brasileira que está incluída no modelo
5. **Auditoria** — Cada interação fica registrada com data, hora, usuário, prompt e resposta, formando trilha de auditoria completa

**Tudo isso acontece sem nenhuma chamada para servidores externos.** A máquina do escritório é simultaneamente o frontend, o backend, o banco de dados e o cérebro de IA. O contador pode usar Oxta Contábil sem conexão à internet.

### Recursos de exportação

Oxta Contábil exporta para **cinco formatos profissionais** já no MVP:

- **Excel** (ClosedXML) — 3 abas com resumo executivo, detalhamento e classificação
- **CSV pt-BR** — para integração com sistemas terceiros
- **SPED Fiscal** — conforme ATO COTEPE 44/2018, pronto para entrega à Receita
- **Domínio Sistemas** — formato proprietário do líder de mercado brasileiro
- **PDF de Conferência** (QuestPDF) — relatório executivo para reuniões com cliente

---

## Por que Agora

Três forças simultâneas tornam este o momento exato para Oxta:

### 1. LGPD entrou em fase de fiscalização real

A Lei Geral de Proteção de Dados foi sancionada em 2018, entrou em vigor em 2020, mas só em 2024-2025 a ANPD começou a aplicar multas substanciais. Empresas brasileiras estão buscando alternativas urgentes às ferramentas que vazam dados para o exterior.

### 2. Hardware moderno suporta IA local

Notebooks vendidos hoje (Intel 13th gen, AMD Ryzen 7000, Apple M3) têm capacidade computacional suficiente para rodar modelos Oxta de 1 bilhão de parâmetros em tempo real — algo impossível há 5 anos. A janela de oportunidade para IA edge-first abriu agora.

### 3. Big Tech americana sob pressão geopolítica e regulatória

OpenAI, Anthropic e Google enfrentam crescente regulação nos próprios EUA, pressão regulatória na Europa (AI Act), e desconfiança crescente em mercados emergentes. Para o Brasil, há oportunidade clara de construir soberania tecnológica enquanto o mercado global está dividido.

---

## Diferenciação Competitiva

| Critério | ChatGPT / Claude API | Modelos open-source (Llama, etc.) | **Oxta** |
|----------|---------------------|----------------------------------|----------|
| Dados ficam no Brasil | ❌ EUA | ⚠️ Depende do hosting | **✅ Máquina do cliente** |
| LGPD-compliant por construção | ❌ | ⚠️ Requer setup complexo | **✅ Por design** |
| Treinado em português técnico brasileiro | ❌ Português genérico | ❌ Foco em inglês | **✅ Fiscal/jurídico BR** |
| Roda offline | ❌ | ⚠️ Requer GPU dedicada | **✅ CPU moderna** |
| Custo recorrente | US$ por token | Hardware + ops | **Licença previsível** |
| Construído no Brasil | ❌ | ❌ | **✅** |
| Especialista em domínio brasileiro | ❌ | ❌ | **✅ Contábil, em breve jurídico/saúde** |

---

## Roadmap

### 2026 H1 — Oxta Contábil (em andamento)
Programa de design partners com escritórios de contabilidade brasileiros. Refinamento do modelo com casos reais sob NDA. Lançamento comercial após validação com 5-10 primeiros clientes.

### 2026 H2 — Oxta Jurídico
Vertical para escritórios de advocacia: análise de jurisprudência, redação assistida de petições, busca em legislação. Reusa runtime Oxta com fine-tune especializado.

### 2027 H1 — Oxta Saúde
Vertical para clínicas e laboratórios: estruturação de prontuário, sugestão de CID, conformidade LGPD para dados de saúde (categoria especial).

### 2027 H2 — Plataforma Oxta
SDK e marketplace de verticais. Permite que parceiros construam seus próprios "Oxta de domínio X" reusando a infraestrutura.

### 2028+ — Soberania Computacional
Acordos com governo federal e estadual para infraestrutura crítica. Posicionamento como alternativa nacional para uso oficial brasileiro em setores sensíveis (defesa, finanças, justiça).

---

## A Tecnologia em Números

- **~80.000 linhas de código C++20** no runtime central
- **186 arquivos .NET** no Oxta Contábil
- **5 formatos de exportação fiscal** prontos para produção
- **Bindings nativos para Python, REST, CLI**
- **Engine de memória Rust** integrada via FFI
- **AVX2 SIMD nativo** com fallback para CPUs sem suporte
- **Suporte CUDA** para aceleração opcional via GPU
- **Construído do zero** — sem dependências de PyTorch, TensorFlow ou qualquer framework de pesquisa estrangeiro

---

## Visão

Oxta nasceu de uma observação simples: o Brasil tem talento de classe mundial em engenharia de software, e tem mercado próprio com necessidades específicas que ferramentas estrangeiras nunca vão atender com prioridade. **Não precisamos importar o que conseguimos construir.**

Nosso compromisso é construir a infraestrutura de IA que o Brasil precisa para o século XXI — local, eficiente, brasileira por dentro e por fora.

A primeira parada dessa jornada é o escritório de contabilidade no interior de qualquer estado brasileiro, processando notas fiscais com a tranquilidade de saber que os dados dos clientes nunca saíram dali. A última parada não está definida — porque depende dos brasileiros que vão se juntar a nós no caminho.

---

## Contato

**Vitor G. C.**
Founder & Arquiteto-chefe
GitHub: [@vitorGgC569](https://github.com/vitorGgC569)

---

*Este documento descreve Oxta em sua arquitetura atual. Datas e capacidades específicas evoluem rapidamente — entre em contato para a versão mais recente do status técnico e do programa de design partners.*
