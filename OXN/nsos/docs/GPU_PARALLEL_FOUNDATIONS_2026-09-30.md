# Lote paralelo: fundamentos Mamba-3, Attention RDNA e Adam esparso

## Escopo e estado

Dois subagentes nativos Codex, mesmo modelo/effort herdados, papéis distintos:
Banach (Attention) e Turing (otimizador). Lead: matemática e backward Mamba-3,
revisão de contratos, registro CMake e validação final conjunta. Sem SENTRA.
KAN congelado, incluindo código/testes e resultados negativos preservados.
Nenhum default ou treinamento de produção foi alterado neste lote.

**Este lote NÃO completa o roadmap nem certifica SOTA/ganho de throughput.**
São fundamentos isolados para integração posterior, deliberadamente sem
reaproveitar checkpoints Mamba-2 como se fossem Mamba-3.

## Mamba-3 — referência matemática FP64

`include/mamba3_reference.h`, `src/mamba3_reference.cpp`: recorrência SISO
exponential-trapezoidal, rotação em pares adjacentes, GQA, biases pós-BCNorm,
skip/gate opcionais, prefixos válidos e estado completo fase/SSM/K/V. VJP de
todos os operandos, biases/skip e quatro estados iniciais, incluindo sementes
nos estados finais. Fronteiras temporais têm adjunto explícito, sem detach
silencioso. Forward não armazena o histórico SSM completo; backward da
referência o recompõe integralmente. Não é backend de produção otimizado.

`include/mamba3_preprocessing.h`, `src/mamba3_preprocessing.cpp`: BC RMSNorm,
heavy-tail/floor de A dependente do dado, DT softplus/bias, ADT=A*DT,
broadcast dos ângulos compartilhados sobre heads e seus VJPs. Padding é
ignorado antes das operações, não apenas zerado na saída. Pesos de norm/bias,
ângulos e DT têm gradientes separados. EPS/floor são constantes fixas.

Fonte de contrato: módulo e referências SISO oficiais, revisão fixada
`e9594ce1c732d97440f0332fdc43170a2294dbfa`.
[Código original](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py).
Ainda faltam execução upstream lado a lado, projeções/output/norm do módulo,
backend HIP SISO, histórico por fronteiras na GPU, MIMO, estado/checkpoint e
integração Jamba/Trainer. Não anunciar a referência como bloco completo.

Teste novo: `tests/test_mamba3_reference.cpp`, primal matricial quadrático
independente, diferenças finitas com sementes de saída+estado, composição de
VJP entre chunks, máscaras/NaNs em padding e contratos inválidos. A cadeia de
pré-processamento tem equações escalares independentes e diferenças finitas.

## Attention RDNA — provedor híbrido isolado

`include/cuda/attention_rdna_training.cuh`,
`src/cuda/attention_rdna_training.cu`: ABI explícita com identidade
`rdna3_wave32_qk_wmma_lowp_ste_fp32_pv_vjp_exp_owner_v1`.
QK usa rocWMMA lowp/FP32 acumulador, inclusive recomputação dos scores no
backward. PV/dP/dQ/dK/dV permanecem contrações FP32 escalares. Casts Q/K/V
usam STE explícito. Não é paridade com atenção FP32 nem FA4 integral.
Softmax estável com `expf`, causal/janela/GQA/prefixo/tails, owners determinísticos,
sem S² global. Sem dropout, bias, RoPE ou cache dentro do provedor.

Suporte: gfx1100/1101/1102 wave32, arquitetura compilada e atributos verificados,
D<=256, H<=1024 divisível por KV, escala positiva<=16. Operand/dO válidos finitos
com magnitude<=64; status device por batch rejeita violações e prefixos inválidos.
`true` significa enqueue, não sucesso numérico. Aproximadamente 52 KiB de LDS
no maior tile; ainda sem perfil ISA/ocupação. Estatísticas/delta/status globais:
`12*B*S*H+4*B` bytes, fora payloads e gradientes.

Pendente: tape imutável/ownership, seletor opt-in e fingerprint incluindo precisão,
telemetry, gate de status no Trainer antes de publicação, integração de RoPE/
projeções/grad accumulation, benchmark end-to-end e qualidade. O provedor não
é chamado pelo modelo atual; sua ABI explícita é a única entrada deste lote.

Teste novo `tests/gpu/test_gpu_attention_rdna_training.cpp`: referência FP64
com operandos lowp congelados, VJP e diferenças finitas STE, padding venenoso,
tails/GQA/janelas, status, canários, alias e repetição determinística.

## Otimizador — predicados device-side isolados

Sidecar `NsosActivityAwareOptimizerDesc`, sem alterar launch ABI legado/chunk16B.
Predicados apontam para bytes de contribuição no dispositivo; tabela/entrada
nula significa tensor denso. Inativo não lê gradientes/weights/moments, não recebe
clip/Adam/decay. Zero com contribuição continua ativo. União de microbatches,
current/first-write e abort sticky são explícitos e separados de existência do
buffer. Atualização valida offsets antes de publicar qualquer byte de atividade.

Finite/scaling/norma FP64 tree256/clip device-side/Adam recebem o mesmo contrato.
Momentos têm armazenamento reservado separado de presença lógica lazy; init
ocorre só para ativos e depois do finite gate. Overflow de update NÃO é
transacional: rollback deve incluir pesos, momentos, presença e versões/caches.
Gate de pré-update tem que permanecer estável e separado do status de update.
Ownership é por grupo/device/stream; reset/reuso precisa de ordenação explícita.

Ainda NÃO integrado a `GpuMoeTraining::materialize_counts()`, Parameter/Trainer,
regularizador de criticalidade, momentos/checkpoint e versões/cache. Esses
pontos são obrigatórios antes de alegar MoE/Adam inteiramente device-side.

Teste novo `tests/gpu/test_gpu_device_sparse_optimizer.cpp`: inativo com NaN,
inativo após ativo, zero legítimo, união/reset/abort, offsets/chunks inválidos,
momentos lazy e finite/clip/Adam contra referência escalar independente.

## Validação

Implementação e registros concluídos antes de iniciar build/testes. Builds
Release CPU/HIP concluíram. Suíte CPU **61/61** em 16,55s; HIP **126/126** em
146,94s; novos testes dirigidos **3/3** em 0,98s. Testes GPU executaram na
AMD Radeon RX7600, HIP/gfx1102, runtime reportado12000, sem skip bem-sucedido.

A primeira suíte CPU passou60/61: manifesto estático de kernels não listava a
nova source de Attention. Corrigido `test_gpu_backend_configuration.py` e
repetida a suíte completa com61/61. Falha e reparo permanecem nos logs.
`git diff --check` dos arquivos rastreados relevantes não encontrou problemas.

Artefatos: `artifacts/gpu_parallel_foundations_20260930/`, incluindo logs de
build, suites/respectivos outputs completos e `validation-manifest.json` com
comandos, resultados e19 hashes de fontes/binários. Worktree suja, manifesto
de fontes PARCIAL: não é snapshot completo nem baseline de throughput.

SHA256 módulo Python CPU:
`0b021097d3ab1e14a996e6c707a83d6c6333139bd405b27ab443a235436e8a86`.
SHA256 módulo Python HIP:
`2bf2aeddfe7fc2986c66da53b77c1aee576ad328fc5d332826c89adbc302df33`.
Os executáveis dirigidos também possuem hashes próprios no manifesto, pois
primitivas isoladas podem não ser linkadas a consumidores que não as chamam.

Geometria da referência SISO neste lote: N par e `1<=R<=N/2`. Não cobre MIMO
nem modos sem rotação. Não foi executada comparação numérica com upstream;
as equações foram verificadas contra seu contrato e oráculos independentes.

Paridade e checkpoint do candidato KAN preservado passaram como parte da
regressão geral. Isso NÃO retoma desenvolvimento KAN, não mede sua velocidade
e não substitui a ablação SwiGLU/MLP exigida para qualquer promoção futura.
Nenhuma medição de velocidade, treinamento real, ablação de qualidade ou
campanha adicional de KAN foi realizada neste lote.
