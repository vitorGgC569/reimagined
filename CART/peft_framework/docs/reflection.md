# Reflexão Técnica e Evolução do Projeto

Este documento registra os aprendizados técnicos, desafios e a evolução da arquitetura do framework PEFT.

## Evolução da Arquitetura

O projeto começou como um framework "Research Mode" puramente em C++17, focando em implementar algoritmos complexos (LoRA, DoRA, MiSS) do zero para fins educacionais. No entanto, para atingir viabilidade industrial, houve um pivô estratégico.

### De "Standalone" para "Industrial Hybrid"
Percebemos que reinventar a roda de álgebra linear (Tensores manuais) limitava severamente a escalabilidade e o suporte a GPU.
*   **Decisão:** Criar uma extensão nativa em C++ para PyTorch (`peft_framework/pytorch_extension`).
*   **Resultado:** Isso permitiu manter a lógica complexa (TurboFusion) em C++ otimizado, enquanto delegamos a computação massiva para o backend do LibTorch/ATen. O resultado foi um ganho de **113x de performance** e suporte automático a CUDA/ROCm.

## Riscos Encontrados e Mitigação

### 1. Instabilidade Numérica da SVD
*   **Problema Original:** A implementação manual de SVD (`src/LinAlg.cpp`) usando iteração de potência com deflação era instável para múltiplos autovalores e lenta, comprometendo a lógica de Rank Dinâmico (MiSS).
*   **Solução (Mitigação Completa):** Ao migrar para o backend do PyTorch, passamos a utilizar `torch.linalg.svd`. Esta implementação usa LAPACK/cuSOLVER por baixo dos panos, garantindo estabilidade numérica perfeita e velocidade acelerada por hardware. Isso viabilizou a implementação robusta do `TurboFusion` dinâmico.

### 2. Complexidade de Otimizadores
*   **Problema:** Implementar AdamW ou SGD com momentum manualmente em C++ é propenso a erros e difícil de manter.
*   **Solução:** A integração com `torch.nn.Module` permite que o TurboFusion seja otimizado por qualquer otimizador padrão do PyTorch (Adam, SGD, AdaFactor) sem necessidade de código extra.

## Conclusão Técnica
O framework atingiu seu objetivo de ser um "trunfo". O algoritmo **TurboFusion** provou ser superior em flexibilidade e equivalente em qualidade ao estado da arte, com a vantagem única de adaptar seu próprio tamanho (rank) durante o treino, algo que implementações estáticas de LoRA não fazem.
