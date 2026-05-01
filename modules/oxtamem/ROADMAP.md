# Roteiro de Engenharia de Sistemas: Oxta Mem (4D Causal Memory)

Este guia separa a Ciência (provar que é verdade) da Engenharia (fazer funcionar rápido).

## FASE 1: Validação Teórica Absoluta (The Scientific Siege)
**Objetivo:** Provar matematicamente que a estrutura é superior às alternativas existentes antes de escrever uma linha de código de produção. Se falhar aqui, o projeto morre.

### 1.1. Análise Assintótica Formal (Big-O Battles)
Você deve criar um documento de prova comparando o seu grafo 4D contra as três estruturas dominantes da indústria:
*   **B-Trees (SQL Relacional):** O padrão para busca em disco.
*   **LSM-Trees (Log-Structured Merge-Trees - RocksDB/Cassandra):** O padrão para alta escrita.
*   **HNSW (Hierarchical Navigable Small World - Vector DBs):** O padrão para IA.

O Benchmark Teórico deve provar:
*   **Recuperação Temporal:** Demonstrar $O(1)$ ou $O(k)$ (onde k é profundidade causal) vs $O(log N)$ das B-Trees.
*   **Write Amplification:** Calcular quantos bytes físicos são escritos no disco para cada byte lógico. Estruturas imutáveis tendem a gastar mais espaço; a prova deve mostrar que a deduplication via hash compensa isso.
*   **Branch Prediction Failures:** Estimar teoricamente a taxa de "cache misses" da CPU ao navegar no grafo vs arrays lineares.

### 1.2. Verificação Formal de Modelos (TLA+)
Não confie em testes unitários para a lógica central. Use uma linguagem de especificação formal como TLA+.
*   **Objetivo:** Modelar o algoritmo de concorrência.
*   **O que provar:** Que em um cenário de escritas simultâneas em múltiplos threads, a integridade do grafo causal (DAG) nunca é violada (sem ciclos, sem "dangling pointers").

### 1.3. Simulação de Entropia de Dados
Rodar simulações estatísticas (não código de produção) para prever o comportamento do sistema após 10 anos de dados.
*   **Fragmentação:** O grafo fica "esparso" demais na memória?
*   **Profundidade:** Qual é a latência de navegar 1 milhão de passos para trás?

---

## FASE 2: Arquitetura e Implementação ("Bare Metal")
**Objetivo:** Construir o Engine para rodar próximo à velocidade do hardware, eliminando gargalos de software.

### 2.1. A Escolha da Linguagem: Rust
Não há outra escolha racional para um produto novo de infraestrutura em 2026.
*   **Por que não C++?** Gerenciamento manual de memória em grafos complexos é propenso a Memory Leaks e Segfaults. Rust garante segurança de memória em tempo de compilação.
*   **Por que não Go?** O Garbage Collector (GC) introduz pausas imprevisíveis (latency spikes), inaceitáveis para HFT ou Kernels de tempo real.
*   **Alvo:** `no_std` (Rust sem biblioteca padrão) para permitir que o Engine rode dentro do Kernel Linux (eBPF) ou como um Unikernel.

### 2.2. Gestão de Memória: Arena Allocation (Slab)
O maior inimigo de estruturas de grafos é o Cache Miss. Se cada nó estiver em um lugar aleatório da RAM, a CPU perde tempo buscando.
*   **Estratégia:** Implementar um alocador de arena linear.
*   **Como funciona:** Os nós temporalmente adjacentes (t, t+1, t+2) são gravados fisicamente contíguos na memória.
*   **Benefício:** Pré-carregamento (Prefetching) da CPU torna a navegação no grafo quase tão rápida quanto ler um array.

### 2.3. Persistência: Zero-Copy Serialization (rkyv)
Esqueça JSON, Protobuf ou SQL.
*   **Estratégia:** O layout dos dados no Disco deve ser idêntico ao layout na RAM.
*   **Técnica:** `mmap` (Memory Mapped Files). O sistema operacional carrega o arquivo de disco direto para a memória virtual.
*   **Biblioteca Alvo:** `rkyv` (Rust) garante desserialização com custo zero. O Engine "acorda" instantaneamente, sem parsing.

### 2.4. I/O Assíncrono: io_uring
Para alta performance no Linux.
*   Substituir chamadas de sistema padrão (read/write) por `io_uring`. Isso permite submeter milhares de operações de disco em lote para o Kernel sem trocas de contexto (Context Switching) caras.

---

## FASE 3: A Camada de Interface (O Produto)
**Objetivo:** Tornar a tecnologia utilizável por quem não é engenheiro de kernel.

### 3.1. Protocolo de Rede: Compatibilidade com Redis (RESP)
*   **Estratégia Cavalo de Troia:** Faça o Engine falar o protocolo do Redis.
*   **Por que:** Qualquer empresa já tem drivers de Redis instalados. Eles apenas apontam o IP para o seu Engine.
*   **Diferencial:** Comandos estendidos.
    *   Redis: `GET key`
    *   Seu Engine: `GET key @time=yesterday` (Viaja no grafo)

### 3.2. Bindings Nativos para IA (Python/PyO3)
Criar uma biblioteca Python que encapsula o núcleo Rust.
*   **Integração:** Plug-in direto para PyTorch e LangChain.
*   **Uso:** `memory_tensor = engine.recall(query_vector, causality_depth=5)`

---

## FASE 4: O Benchmark Industrial (A Prova de Venda)
**Objetivo:** Criar os gráficos que vão para o Slide Deck de vendas/investimento. Comparar contra o "Estado da Arte".

### 4.1. Cenário: "The Flash Boys Test" (Latência Pura)
*   **Concorrente:** Redis e kdb+ (banco de HFT).
*   **Métrica:** Latência de leitura no percentil 99.99 (P99).
*   **Vitória Esperada:** Mostrar que sua latência é estável (jitter zero) independentemente do tamanho do histórico, enquanto os outros degradam.

### 4.2. Cenário: "The Amnesia Test" (Para IA)
*   **Concorrente:** Pinecone / ChromaDB (Vector Search).
*   **Teste:** Inserir fatos contraditórios ao longo do tempo (ex: "O presidente é A", depois "O presidente é B").
*   **Query:** "Quem era o presidente quando o evento X ocorreu?"
*   **Vitória Esperada:** O Vector DB falha ou alucina (mistura A e B). O seu Engine retorna a resposta exata com 100% de precisão causal.

### 4.3. Cenário: "The Audit Storm" (Para Compliance)
*   **Teste:** Reconstruir o estado exato do sistema em 1.000 pontos aleatórios do passado.
*   **Concorrente:** Bancos com Change Data Capture (CDC).
*   **Vitória Esperada:** Seu sistema faz isso em milissegundos. O concorrente leva minutos fazendo "replay" de logs.

---

## Resumo dos Alvos Tecnológicos

| Camada | Tecnologia Escolhida | Justificativa |
| :--- | :--- | :--- |
| **Linguagem** | Rust | Segurança de memória, performance, ecossistema moderno. |
| **Kernel** | io_uring (Linux) | I/O assíncrono de altíssima performance. |
| **Memória** | Arena / Slab Allocator | Localidade de cache (evitar pointer chasing lento). |
| **Disco** | Memory Mapped (mmap) | Zero-copy load. O disco é extensão da RAM. |
| **Rede** | TCP (Protocolo RESP) | Compatibilidade imediata com infraestrutura existente. |
| **Algoritmo** | Merkle-DAG Causal | A estrutura 4D central que valida o produto. |
