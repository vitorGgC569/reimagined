# KAN: candidato preservado, desenvolvimento adicional suspenso

## Decisão vigente do usuário

Em 2026-09-30 o usuário determinou interromper desenvolvimento adicional do
KAN, preservar código/testes/resultados negativos e manter todos os caminhos
novos opt-in. Retomar somente depois de ablação que demonstre melhor
qualidade/custo contra SwiGLU/MLP. Esta decisão substitui a prioridade KAN do
objetivo anterior; não suspende o objetivo integral nem autoriza apagar o lote.

`ModelConfig.use_kan` continua false por padrão. As políticas
`NSOS_KAN_RECOMPUTE_TRAINING` e `NSOS_KAN_WMMA_TRAINING` também são false por
padrão. Nenhuma foi promovida. Não ligar KAN, TTT, MoE e attention simultaneamente
por conveniência ou chamar uma composição mais complexa de mais inteligente.

## Baseline escalar realmente validado

O candidato remove a base RBF global e sua matriz de gradientes, calcula escalas
QAT no dispositivo e recomputa tiles no backward. Entradas do novo tape são
próprias; formato, versão, endereço, quantização e precisão são verificados.
Consumo do tape e transição de dispositivo encerram sua validade.

Política: `device_qat_tree256_rbf_tile16_recompute_v1`. Preserva operand rounding
FP32/BF16/FP16, FP32 accumulation e o STE declarado, não diferenciação da
quantização discreta. A árvore ordenada QAT e a ordem aritmética mudam em relação
ao legado; identidade distinta é obrigatória para retomada.

Após essa implementação (antes do incremento WMMA):

- CPU60/60 em30.44 s; HIP122/122 em152.42 s; Python17/17.
- Directed GPU2/2 em5.30 s: paridade KAN e continuação de checkpoint KAN.
- Referência independente double e diferenças finitas no primal não quantizado;
  referência STE quantizada; rank2/rank3, tails, precisão e rejeições de tape.
- Binário anterior real recusado antes do primeiro sample por identidade KAN
  ausente (`stale-binary-rejection.json`). Isso não significa zero transferências
  na inicialização: pesos foram inspecionados antes da checagem de política.

Artefatos: `artifacts/gpu_kan_recompute_20260930`; logs `cpu-full.log`,
`hip-full.log`, `hip-directed.log`, `python-policy.log`. Falha inicial de build
HIP por min(int,int64) e reparo permanecem em `hip-build.log` e
`hip-build-repair1.log`.

HIP validado SHA256:
`961f8e3b307963e6aba9957d733da44e401b7110620b475491d0324f367a5f58`.
CPU validado SHA256:
`d45790d223e7fe01b22d508726b2da8c2bd4e758b2195fc15617dafcfb99e8d1`.
HIP pré-KAN:
`d251673123ebc69f0c6ad189236c7c1ea22ed33d55a7856def1e1cb27c8b881f`.
`reference-before` e `reference-scalar` preservam binários; rocBLAS é junction
para o build, portanto não são pacotes autocontidos.

## Campanha pareada: regressão larga preservada

RX7600/gfx1102, dois Mamba-2 faithful com KAN quantizado, state64, grid5,
vocab257, seed7301, determinismo, perfis redesign iguais. Ambos os braços usam
o mesmo binário validado acima. QAT do scheduler OFF não desliga o fake-quant
dos pesos KAN, que Jamba habilita. B2 usa SFT sintético heterogêneo/máscaras;
B1 usa IDs causais sintéticos. Sem tokenizer e sem certificação PT-BR.

16 steps por processo; primeiro excluído do p50. Série1 legacy->tiled;
série2 tiled->legacy. Os16 probes passaram finite/gradient-path checks.

| Geometria | Legado série1/2 (ms) | Tiled série1/2 (ms) |
|---|---:|---:|
| D128/S256/B1 FP32 | 9.965 / 10.837 | 9.171 / 9.196 |
| D128/S256/B1 BF16 | 10.013 / 10.025 | 9.882 / 9.188 |
| **D768/S512/B1 BF16** | **44.779 / 45.266** | **57.146 / 57.088** |
| D128/S128/B2 BF16 | 10.613 / 10.500 | 10.216 / 9.996 |

**D768 ficou27.62%/26.12% mais lento**, apesar de D2H8->4 chamadas/step.
Fences permaneceram4/step. D2D6->8: clone imutável da entrada adiciona uma
cópia por camada. Peak live allocation largo508.875->503.125 MiB; essas são
alocações rastreadas pelo pool, não utilização física total nem largura de banda.
Não inferir melhoria global de memória só pela eliminação da base RBF.

Hashes iniciais iguais por par; finais diferentes. Max diferença de loss:
FP32 9.54e-7, BF16 até2.11e-4. Paridade numérica não prova igual convergência ou
qualidade/custo. Ganhos pequenos nas formas menores não justificam default.

Reprodução histórica (usar novo diretório/nomes para não sobrescrever):
`run_paired.ps1` e `summarize_paired.ps1` no diretório dos artefatos.
JSONs individuais e `paired-summary.log` preservam também os resultados ruins.

## Incremento WMMA existente quando o usuário interrompeu

Código opt-in para RBF forward/dW/dX BF16/FP16 tile32, testes e identidade
`rdna3_implicit_rbf_tile32_minrows16_mindim32_v1` foram escritos ANTES da
interrupção. Os builds já iniciados terminaram; NÃO houve campanha numérica,
checkpoint ou benchmark desse incremento. Python19/19 verifica políticas,
não execução dos kernels.

HIP compilado, **não certificado**:
`4ed5489286ca25c940bdb9a95bd5c977d34cbf978d7f71aebff6cc40f0d1b54a`.
CPU compilado, **não recertificado**:
`082a3dfbc181c6def22c5ad9b13e97e1d14c9e8e4857d4410ca3e0e09ede8bd7`.
Logs `hip-build-wmma.log`, `cpu-build-wmma.log`, `python-policy-wmma.log`.
Não atribuir CPU60/HIP122 ao código posterior; não afirmar que WMMA resolveu
a regressão. Nenhum desenvolvimento/campanha adicional KAN foi iniciado após
a decisão do usuário.

## Gate para retomar

Atualização posterior de certificação: a regressão geral do lote paralelo
CPU61/HIP126 passou, incluindo os testes numéricos/checkpoint já preservados
do candidato. Ver `GPU_PARALLEL_FOUNDATIONS_2026-09-30.md` para a revisão dos
binários e logs. Isso não é desenvolvimento nem campanha de desempenho KAN;
não altera a regressão D768 histórica nem satisfaz o gate abaixo.

Ablação futura KAN versus SwiGLU/MLP: mesmos dataset/tokenizer/holdout, máscaras,
precisão, seeds e protocolo de otimização. Comparar parâmetros efetivamente
treináveis/ativos e reportar tanto orçamento de tokens quanto wall-clock;
uma arquitetura não pode ganhar apenas por receber mais treino/compute.
Medir held-out loss, respostas corretas/generalização, repetição/boilerplate,
tempo por step, tempo até qualidade-alvo e VRAM. Reportar variabilidade e
resultados negativos; definir limiares antes de rodar. Ganho isolado de loss
ou kernel não basta. Sem esse gate, manter desenvolvimento suspenso e opt-in.
