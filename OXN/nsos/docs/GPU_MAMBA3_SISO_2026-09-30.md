# Mamba-3: SISO recorrente AMD com replay por fronteiras

## Escopo

Entrega de um operador **pós-BCNorm SISO FP32** HIP/RDNA3, alinhado ao contrato
da referência FP64 já validada. Não é um alias Mamba-2, nem uma nova flag que
renomeia sua convolution. Identidade: `rdna3_fp32_siso_boundary_replay_owner_v1`.
`gpu_mamba3_siso.h/.cpp` fornece o tape Tensor proprietário; ABI de kernels em
`cuda/mamba3_siso_kernels.cuh/.cu`. GPU-only, sem fallback CPU oculto.

O bloco Mamba-3 completo ainda exige projeções, BCNorm/heavy-tail/DT em device,
output/norm, registro de parâmetros, Jamba/Trainer/bindings e checkpoint.
MIMO também continua obrigatório e pendente. Os kernels deste lote ainda são
um baseline recorrente, não o SSD matricial maximamente otimizado.

## Matemática e memória

Forward inclui exp-trapezoidal, fase acumulada, biases pós-normalização,
rotação adjacente original SISO, GQA, D skip e SiLU gate opcionais. Estado de
sessão completo: fase/SSM/K anterior/V anterior. Prefixos válidos, inclusive
zero, não leem dados de padding. ADT/DT são entradas independentes deste
operador; o bloco deverá encadear ADT=A*DT e DT=softplus(rawDT+bias).

Uma CTA por batch/head mantém o estado em LDS e registra apenas entradas de
chunks16/32. Backward recompõe um chunk em scratch global reutilizado e mantém
os quatro adjuntos durante toda a sequência. Sementes opcionais nos quatro
estados finais e gradientes dos quatro estados iniciais são explícitos. Não
há detach silencioso entre chunks/calls. GQA e biases/skip compartilhados têm
owners determinísticos, sem atomics de gradiente. Atomics são usados apenas
para latch de falha numérica, não para reduzir gradientes.

Buffers FP32, BH=B*H, C=ceil(S/chunk):

- Boundaries: `BH*C*(P*N+N+P+R)` floats.
- Replay: `BH*((chunk+1)*P*N+2*chunk*N+chunk*R)` floats.
- Grad partials: `BH*(2*S*N+2*N+1)` floats.

Exemplo especificado B1/H24/S1024/P64/N64/R16/chunk32: histórico denso SSM FP32
teórico384MiB; os três buffers definidos acima somam37,23MiB. Isso não é
medição do pico total de VRAM: exclui entradas privadas, estados, saídas e
gradientes. Não é comparação de velocidade com Mamba-2, cuja matemática muda.

## Ownership, integração e falhas

`mamba3_siso::Tape::forward` recebe operandos FP32 contíguos e estado inicial
explícito. Faz clones privados imutáveis e possui todos os buffers/metadados.
Snapshot final retorna cópias proprietárias para fork de sessão. Backward é
single-use; cancelamento impede reuso, mantendo storage até a destruição.
Thread/device/stream diferentes são rejeitados. Saídas e adjuntos não são
acumulados/publicados automaticamente no registry de parâmetros.

ABI raw: caller mantém storage e metadata vivos/imutáveis. Não há allocation,
H2D/D2H ou fences internos. Wrapper owning: prefixo host faz uma pequena H2D
síncrona para garantir lifetime; clones/allocations também têm custo. Não
prometer eliminação total de overhead só porque os kernels não fazem D2H.
`audit_status()` é uma fronteira explícita D2H+fence. O Trainer futuro deverá
integrar `device_status()` ao finite gate antes de publicar qualquer resultado.
Destruição dos buffers int usa driver free e pode sincronizar; manter tapes
na lane é necessário. Graph capture/retomada/packs não estão certificados.

Capacidades: gfx1100/1101/1102 wave32 compilado, atributos de função/LDS
verificados, P<=64, N par<=64, 1<=R<=N/2, H divisível por G. Precisão FP32;
casts de projeção BF16/FP16 pertencem ao módulo futuro. Inputs/seed finitos
com limites declarados, DT[0,16], ADT<=0. Guardas de dimensão/overflow/alias
antes de enqueue; prefixos/ranges/numerics validados no device. Não colocar
clamps silenciosos sob a política exata. Status por batch: ok0/prefix1/range2/
numeric3. Falhas zeram outputs/adjoints inválidos; qualquer batch rejeitado
invalida a publicação do op inteiro e zera grads dos parâmetros compartilhados.
Enqueue=true não certifica sucesso numérico. Em erro de launch, descartar e
drenar a lane antes de reuso; não alegar rollback transacional automático.

Na revisão final foi corrigido um risco de leitura não uniforme do status
compartilhado pelos heads: somente lane0 captura atomicamente o limite do loop
e o publica em LDS antes das barreiras. Uma falha sinalizada por outra CTA não
pode fazer lanes do mesmo bloco executar números diferentes de iterações.
O risco foi identificado por inspeção; não houve reprodução de deadlock.

## Limites de engenharia e medição

O forward raw lança três kernels e o backward seis; os counters novos
`mamba3_siso_forward/backward` contam chamadas lógicas, não esses launches
físicos. Ainda há preflight, checagens finitas e reduções separados. Não é
monokernel de modelo nem redução comprovada do overhead de treinamento.
O cálculo de capacidade também consulta o runtime/enumeração de devices no
host: a promessa do ABI é ausência de alocação de **storage device** oculto,
não ausência de todas as alocações de metadados host.

Replay global e partials expandidos GQA continuam consumindo tráfego. Clones
privados garantem imutabilidade mas adicionam D2D e duplicação de entradas;
cache de descritores/workspaces, transferência de ownership e liberação por
eventos exigem integração posterior. P/N<=64 é limite do operador, não do
Mamba-3 original. Poucos heads/batches podem subutilizar os32CUs; não houve
perfil ISA/VGPR/LDS/ocupação ou escolha de geometria por desempenho neste lote.

Referência numérica é FP64 local, anteriormente certificada por formulação
quadrática independente e diferenças finitas. Não executamos o upstream
Triton/CuTe lado a lado; o pin original informa o contrato, não certificação
de paridade executada com seu binário. MIMO e bloco completo continuam abertos.

## Validação final do lote

Escrita/revisão e integração CMake concluídas antes de iniciar build/testes.
`test_gpu_mamba3_siso`: referência FP64 de todos os operandos/estados,
diferenças finitas, GQA, tails, gates opcionais, N/P64, chunks16/32, máscaras
incluindo length0/NaN em padding, sequência versus chamadas streaming com
adjunto encadeado, repetição determinística, propriedade/imutabilidade,
cancelamento/reuso/thread/stream, status/ranges, canários/alias, replay com
NaN, várias voltas de fase positivas/negativas nas fronteiras, stream explícito
e limites DT0/exp-underflow. Sem D2H/fence explícito no core verificado por
counters; isso não mede toda sincronização implícita do runtime.

Resultados finais na RX7600/gfx1102, runtime reportado12000, TheRock
`C:/TheRock/build`, Release/Ninja/Python3.12:

- CPU:61/61,17,41s, `cpu-full-repair1.log`.
- HIP:127/127,139,14s, `hip-full-repair1.log`.
- Dirigido SISO:1/1,0,77s, `hip-directed-repair2.log`.
- Mesmo executável com `NSOS_CUDA_SYNC=0`: PASS, `hip-directed-async.log`.
  Inclui stream explícito e todas as checagens; auditoria/resultados ainda
  fazem fences explícitos nos pontos de validação. Não é um teste sem fences
  em toda a execução nem teste de concorrência simultânea entre dois tapes.

Comandos principais (PATH HIP preparado pelo build):

```powershell
cmake --build OXN/nsos/build-gm-cpu --parallel 4
cmake --build OXN/nsos/build-gm-hip --parallel 4
ctest --test-dir OXN/nsos/build-gm-cpu --output-on-failure -j2
ctest --test-dir OXN/nsos/build-gm-hip -R '^test_gpu_mamba3_siso$' -V --output-on-failure
ctest --test-dir OXN/nsos/build-gm-hip --output-on-failure -j1
$env:NSOS_CUDA_SYNC='0'
$env:NSOS_REQUIRE_GPU_TESTS='1'
$env:NSOS_GPU_MEMORY='device'
$env:NSOS_NO_MEMADVISE='1'
& .\OXN\nsos\build-gm-hip\test_gpu_mamba3_siso.exe
```

Módulo HIP SHA256:
`d6602a6b8b9f5aab662401e9d3ba50bea44fd5746f6dccd36ba51b42cc013cae`.
Módulo CPU:
`c1eb39040e8c58cc7b0c496958c5d6ed08677f91065943c27245bdbdcf9177e7`.
Executável SISO:
`3e47772c4f7fe48f56c4b230361744303ab40656a7a8556f0ce3c015a977746f`.

Manifesto e logs em `artifacts/gpu_mamba3_siso_20260930/`.
Antes da correção por inspeção, CPU61/61 (22,34s), HIP127/127 (142,69s) e
dirigido1/1 (0,53s) também passaram, mas não certificam a revisão final.
Os logs foram preservados. Build repair1 incorporou o limite uniforme;
repair2/3 ampliaram testes de fase/stream. Nenhuma tolerância foi relaxada.
Worktree já contém alterações amplas não commitadas; hashes parciais não são
um snapshot completo reproduzível. Não houve CUDA vendor build, memory checker
GPU ou teste upstream executado lado a lado neste lote.

Sem treino de produção, ablação de qualidade, benchmark end-to-end ou promoção
de default. KAN permanece congelado conforme a diretriz mais recente.
