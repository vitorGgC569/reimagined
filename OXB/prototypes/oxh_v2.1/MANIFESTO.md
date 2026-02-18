# Manifesto OXH v2.1: Arquitetura de Ingestão de Alta Performance

**Status:** Validado e Benchmarked (Protótipo Congelado)
**Versão:** 2.1 (Híbrida com Metadados)

## 1. O Problema: "Gagueira" de I/O
O treinamento de LLMs sofre com o gargalo SSD -> CPU -> GPU. Formatos como JSON desperdiçam 40% de banda e matam a CPU com tokenização em tempo real.

## 2. A Solução: Protocolo OXH
1.  **Hibridização Dinâmica:** uint16 para a maioria, uint32 apenas quando necessário (escape 0xFFFF).
2.  **Zero-Copy:** Uso de `mmap` para tratar disco como RAM.
3.  **Indexação .idx:** Offsets de 64 bits para acesso randômico instantâneo.

## 3. Resultados de Benchmark (Sandbox)

| Métrica | JSON Tradicional | OXH v2.1 | Diferença |
| :--- | :--- | :--- | :--- |
| **Throughput (1M samples)** | ~25.683 samples/s | **~43.322 samples/s** | **1.69x Mais Rápido** |
| **Tamanho em Disco** | 240.33 MB | **112.53 MB** | **53% Menor** |
| **Tempo de Carga** | 38.9s | 23.0s | -15s |

*Nota: Teste realizado com 4 workers e warm-up de JIT.*

## 4. Estrutura do Código
*   `oxtacore/engine.py`: Motor de descompressão Numba.
*   `oxtacore/converter.py`: Conversor JSONL -> OXH.
*   `oxtacore/dataset.py`: PyTorch Dataset com otimização de array numpy.
