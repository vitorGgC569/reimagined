# Mamba-3: pré-processamento e SISO encadeados na GPU

## Escopo e contrato

Este lote completa o caminho **pós-projeção** SISO: BC RMSNorm com pesos
aprendíveis antes dos biases por head, heavy-tail/floor para A dependente do
dado, DT=softplus(rawDT+dt_bias), ADT=A*DT e broadcast de ângulos compartilhados
[B,S,R] para os heads. Encadeia o VJP dessas operações ao operador SISO HIP
existente, incluindo todos os quatro adjuntos do estado. Não há conversão
numérica à CPU dentro de forward/backward.

Contrato original fixado:
[mamba3.py e9594ce](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py).
Referência FP64 local de pré-processamento já existe e seu teste independente
faz diferenças finitas. Não houve execução do upstream Triton/CuTe lado a lado.
Na igualdade do floor adotamos explicitamente derivada zero, a convenção da
referência local; não alegar equivalência upstream de subgradiente no kink.

`gpu_mamba3_projected.h/.cpp` possui operandos clonados, inversas, partials,
status e tape SISO interno. `cuda/mamba3_preprocess_kernels.cuh/.cu` fornece ABI
raw, sem alocação de device storage/readback/fence. Identidade:
`rdna3_fp32_bcnorm_fp64_radial_vjp_heavy_dt_broadcast_v2`. FP32 contíguo, gfx11 wave32,
P<=64, N par<=64, chunks16/32; falha explícita fora desse contrato. O original
tem default N128: o limite atual NÃO equivale a suporte integral ao original.
Norm eps[1e-12,1], floor(0,64], operandos brutos/weights/biases |x|<=64,
Q/K preparados<=64, DT<=16, ADT<=0. Sem clamp silencioso de DT/Q/K.

## Reduções, máscaras e falhas

Inversas Q/K [B,S,G,2] retidas permitem reuso no VJP dos pesos. O VJP de
inputs RMS recalcula squares e weighted dots em FP64, com subtração FMA e
cast final FP32: a versão só FP32 perde o pequeno resíduo radial de epsilon
com adjuntos grandes. Isso roda no dispositivo, mas seu custo na RDNA ainda
não foi perfilado. Forward e demais reduções continuam FP32. dNorm é reduzido em
tiles128 rows, quatro owners por CTA e árvore determinística, depois redução
de partials. dDT_bias usa tiles128 tokens; dAngles soma heads com owner único.
Sem atomics de gradiente. Prefixos0 e padding NaN não são lidos como dados.
Fronteiras/replay e os adjuntos temporais continuam no tape SISO proprietário.

O SISO passa à identidade `rdna3_fp32_siso_boundary_replay_owner_v2`:
SSM/K/V iniciais e dy/seeds finais aceitam valores finitos, sem os antigos
limites arbitrários de65536/64; a fase inicial permanece canônica em +/-2pi.
Overflow calculado continua status3, não é ocultado. Um caso legal gerou
previousK=181.019 e SSM=92681.7: v1 rejeitava o estado que ele próprio produzia
na chamada seguinte. A fixture mantém dy=100 e seeds>64, comparação FP64 e
equivalência forward/VJP entre sequência inteira e streaming, sem afrouxar
tolerâncias (atol3e-4 + rtol8e-4).

Status0/1/2/3 = válido/prefixo/range/numeric. O SISO aceita status upstream
clonado device-side e não o reinicializa como válido. Depois do backward da
recorrência o status é transferido D2D ao VJP de pré-processamento. Uma falha
adicional zera adjuntos de inputs/estado do batch e impede publicação de TODOS
os parâmetros compartilhados. Trainer futuro deve consumir o status antes de
publicar gradientes; retornar Tensor/enqueue não é sucesso numérico certificado.

Thread/device/stream originais obrigatórios; backward single-use, cancelamento
retém storage, caller mutation não reescreve tape. Erro de enqueue exige descarte
e drain de lane, não rollback automático. `audit_status()` é D2H+fence explícito.

## Custos que continuam abertos

Há clones privados duplicados entre composição e tape SISO, duas pequenas
uploads síncronas de prefixos, alocações e possíveis sincronizações implícitas
na destruição dos DeviceBuffers. Preservar tapes no stream até conclusão.
Preparação bruta:5 kernels forward,8 backward e2 para o gate encadeado;
SISO soma3/6. Counters contam chamadas lógicas, não esses launches físicos.
Não é fusão completa de bloco nem ganho de velocidade demonstrado.

Workspace adicional de inversas/partials em floats, T=B*S:
`2*T*G + ceil(T*G/128)*2*N + ceil(T/128)*H`.
Não inclui operandos/tapes, gradientes, estados ou buffers do SISO.

Projeção de entrada/saída, optional outproj norm/gating, registry de parâmetros,
Jamba/Trainer/bindings/checkpoint, N128 e MIMO ainda precisam de implementação.
KAN permanece congelado e os defaults históricos são preservados. O objetivo
integral e os gates de release/qualidade/custo continuam abertos.

## Validação ao final do lote

Testes escritos: composição versus referência FP64 de todos os14 campos e4
estados, diferenças finitas do objetivo conjunto, streaming/VJP entre calls,
prefixos/NaN padding, GQA/tails/P=N64, floor, S1025, determinismo, lane explícita,
imutabilidade/cancel/thread/stream, status upstream/downstream, canários/alias,
inverse cache inválido, fechamento do estado entre calls com adjuntos grandes
e overflow calculado com adjunto finito.

Builds, resultados, hashes e logs serão registrados após a campanha final.
Sem treino de produção, ablação, perfil ISA/ocupação ou throughput certificado.
