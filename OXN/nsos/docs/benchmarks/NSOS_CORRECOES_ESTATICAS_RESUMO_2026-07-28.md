# NSOS — resumo das correções estáticas

Data: 2026-07-28  
Escopo: NSOS, arquitetura Mamba/Attention, OxtaMEM, AMD ROCm/HIP, robustez,
auditabilidade e desempenho de treino.

## Regra de trabalho

Toda a implementação abaixo foi feita por inspeção e correção estática. Nenhum
build, teste, script de projeto, benchmark ou treino foi executado nesta fase.
A validação dinâmica somente deve começar depois da auditoria estática final.

## Implementado

- Contratos de configuração, topologia e checkpoint foram versionados e
  endurecidos, incluindo validação antes de mutação, escrita crash-safe,
  SHA-256 e rejeição transacional de artefatos inválidos.
- A composição híbrida de produção passou a preservar o ramo Mamba e adicionar
  Attention/FFN em paralelo, com gates e normalizações independentes, backward
  exato e telemetria de contribuição/cancelamento entre ramos.
- O registro de parâmetros ganhou nomes estáveis, aliases explícitos,
  trainabilidade correta e hashes reproduzíveis para mapear cada peso.
- `BitLinear` exato deixou de usar magnitude legada; importação ternária
  empacotada agora valida dimensões, escala, finitude e códigos antes do commit.
- O `Trainer` ganhou loss batelada, objetivos batelados, AdamW fundido,
  zeragem multi-tensor, loss scaling, validação completa antes do forward e
  checkpoint transacional do otimizador/scheduler/RNG.
- As propriedades Python do `Trainer` agora usam o mesmo mutex do treino;
  estatísticas são somente leitura e o binding mantém o modelo vivo.
- O Mamba-2 fiel passou a usar layout oficial combinado `[z,x,B,C,dt]`.
  Em FP32, as cinco projeções compartilham uma GEMM; em QAT, `z/x` continuam
  ternárias com STE e `B/C/dt` compartilham uma GEMM exata.
- Foram adicionadas views seguras de storage para preservar os nomes canônicos
  dos pesos sem cópias por passo, além de contadores separados dos caminhos
  agrupados completo e sensível.
- O backward das projeções agrupadas foi fundido, com empacotamento GPU de
  gradientes, uma GEMM de `dW`, uma de `dX`, verificação de versão e consumo
  exato do estado de forward.
- O slice de `Tensor` em GPU foi generalizado para qualquer eixo/rank e deixou
  de tentar acessar ponteiro de device pelo host.
- Hot paths GPU receberam buffers persistentes, cópias verificadas, contadores
  H2D/D2H/D2D, sincronizações observáveis e timing por eventos, evitando drains
  globais usados apenas para diagnóstico.
- O MoE eliminou um download D2H redundante: contagens agora são derivadas dos
  offsets, com validação de monotonicidade e limites.
- Foram corrigidos fallbacks, cópias e erros silenciosos em Mamba, Jamba,
  embedding, Tensor, holographic, TTT, SDK, kernels e scripts de benchmark.
- O backend AMD foi estruturado sobre a camada compatível ROCm/HIP para
  compilar as fontes `.cu`, com identificação de backend/vendor/capacidades,
  modo GPU estrito e suporte de precisão mista condicionado ao hardware.
- A auditoria de camadas/parâmetros registra ativações, gradientes, updates,
  hashes e métricas híbridas; checkpoints e manifests preservam proveniência.

## Estado atual

- As correções recentes passaram por `git diff --check`.
- O workspace já continha muitas alterações do usuário; nenhuma foi descartada.
- Por solicitação posterior do usuário, o build Release AMD/HIP `gfx1102`
  foi executado. `nsos_core`, OxtaMEM, executáveis configurados e
  `nsos_ext.cp312-win_amd64.pyd` foram compilados e linkados com sucesso.
- O build revelou uma chamada incorreta à API de determinismo no teste
  `test_gpu_parity_jamba`; ela foi substituída pela função pública correta e
  a recompilação terminou com sucesso.
- Nenhum teste, benchmark ou treino foi executado. A meta permanece ativa e
  ainda não está integralmente validada em runtime.

## Pendente antes de qualquer build

1. Concluir o manifesto autoritativo por peso/camada: bytes físicos e lógicos,
   device, trainabilidade, aliases/storage compartilhado, FLOPs e hashes.
2. Fechar a contabilidade de todas as transferências/sincronizações GPU e
   incorporar os contadores aos artefatos de benchmark.
3. Revisar todos os produtos dimensionais/índices restantes para overflow e
   terminar os hot paths que ainda serializam scan ou experts.
4. Adicionar os testes estáticos previstos: paridade das projeções agrupadas,
   slices/views, corrupção transacional, concorrência, ausência de D2H,
   gradcheck híbrido, HIP e fault injection.
5. Registrar testes órfãos no CMake e alinhar documentação, schemas e gates.
6. Fazer a auditoria estática requisito por requisito. Somente depois executar
   builds CPU/HIP, testes, paridades, benchmarks multi-seed e treino.
