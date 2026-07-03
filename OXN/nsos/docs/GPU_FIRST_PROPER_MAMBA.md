# GPU-first: proper selective SSM (revisão de implementação)

Status: implementado e revisado (2026-06-14). A via corrigida do Mamba
(`NSOS_MAMBA_PROPER_SSM`) deixou de ser host-only e agora é **GPU-residente**.

## O que foi tornado GPU-first

`Mamba2SSD::forward_proper` / `backward_proper` (src/mamba2.cpp) agora são
**device-aware**: ficam no device do input, sem round-trip para o host.

- **Projeções** δ/B/C/z + out_proj: caminho GPU do `BitLinear` (já existente).
- **conv1d causal depthwise**: novos kernels CUDA
  `launch_conv1d_causal_forward` / `_backward` (src/cuda/mamba_kernels.cu).
- **Scan diagonal (readout linear `y=h*C`)**: novos launchers
  `launch_mamba_proper_scan_forward` / `_backward`, implementados como o flag
  `linear_readout` dos kernels de scan seletivo já validados (mesma recorrência
  afim; o readout linear preserva a validade do prefix-scan).
- **silu(conv)**, **gate `y·silu(z)`**, **skip `u·D`**, **grad_D**: ops `Tensor`
  device-agnósticas (já têm fast-path GPU).
- `Mamba2SSD::to(GPU)` move TODOS os componentes da via proper para o device.

Caminho host preservado como fallback (CPU) — mesma matemática.

## Validação

- **CPU**: `test_gradcheck` (build local MSVC) — `mamba2-proper` d/input 6.2e-3,
  d/A 1.9e-3, d/conv1d 3.5e-3, todos < 1e-1. **Verde, reprodutível.**
- **GPU/T4**: `tests/gpu/test_gpu_parity_mamba_proper.cpp` (registrado no CTest)
  compara forward + grad-de-input CPU vs GPU dentro de 1e-3. Rodar no T4.

## Oportunidades de melhoria (revisão — itens honestos, rastreados)

1. **Parallel-prefix para o scan proper — FEITO.** O kernel Hillis-Steele afim já
   validado ganhou o readout linear (flag `linear_readout`) e
   `launch_mamba_proper_scan_forward` roteia para ele sob `NSOS_MAMBA_PARALLEL_SCAN`
   (Seq≤1024): forward `O(log Seq)` para a via diagonal proper. Default OFF mantém
   o kernel sequencial como referência; validar paridade no T4. (Backward proper
   segue sequencial — paraleliza sobre B·D canais, que satura no treino.)
2. **Kernel fundido de SiLU/dSiLU — micro-opt documentada.** silu e sua derivada
   usam ops `Tensor` elementwise (já GPU). Fundir num kernel único cortaria
   ~poucas launches por silo; ganho marginal, deixado como opt futura (não vale
   CUDA adicional não-testável localmente para ganho negligível).
3. **Determinismo na via proper-GPU.** A conv1d-backward (`grad_in`/`grad_weight`)
   e o `grad_A` do scan usam `atomicAdd` (não-determinístico). `NSOS_DETERMINISTIC`
   AINDA NÃO roteia a via proper-GPU para host (o caminho host já é determinístico).
   Refinamento rastreado: sob o flag, computar conv/scan-backward em host (cópias)
   mesmo em modelo GPU. Não foi adicionado às cegas (sem nvcc local) para não
   introduzir código GPU não-validado.
4. **Decode incremental.** `forward()` na via proper recomputa o scan inteiro a
   cada chamada (sem cache de streaming) → decode autoregressivo é O(n²). Um
   caminho single-token com estado persistente (como o legado) é otimização futura.
5. **N-state (Mamba-2 completo) — #13 — FEITO.** Implementado sob
   `MambaConfig::proper_state_expansion`: estado h ∈ R^{H×P×N}, dt/A por-cabeça,
   B/C por-cabeça N-dim, readout linear y=Σ_n h·C. CPU forward+backward (BPTT com
   reduções cross-p/cross-n) gradcheck-ado (mamba2-nstate d/input 1.19e-2, d/A
   5.9e-4, d/conv1d 1.5e-2). Kernels CUDA GPU-residentes `mamba_nstate_forward/
   backward` (estado em registradores, atomicAdd onde há contenção entre canais p;
   fallback host se N>MAX_N=64), com paridade `test_gpu_parity_mamba_nstate`
   (validar no T4). NOTA: o `mamba_ssd_forward_kernel` chunked legado segue
   presente mas o caminho novo usa kernels próprios casados 1:1 com o host.

## Oportunidades GPU-first pré-existentes (fora desta entrega)

Do `docs/GPU_OPTIMIZATION_ANALYSIS.md`: MoE top-k via `std::partial_sort` no host;
ausência de pinned memory para H2D; CUDA Graphs para shapes estáticos weight-tied;
`persistent_kernel` stub a remover. Rastreados para PRs próprios.

## CUDA-graph decode fim-a-fim — FEITO (opt-in, validação T4)

`JambaModel::forward_ids_decode_graph` (env `NSOS_CUDA_GRAPH_DECODE=1`) captura
UM forward single-token num CUDA graph e o replay-a por token: a cascata de
launches por token vira 1 `cudaGraphLaunch`. Peças: (a) build opt-in
`-DNSOS_CUDA_PTDS=ON` (per-thread default stream — a legacy stream não é
capturável; muda semântica global, então a suíte inteira de gates roda sob ela
no notebook); (b) token+posição entram por staging PINNED relido pelos nós
memcpy capturados a cada replay; (c) kernels de decode da atenção ganharam
variante device-pos (`pos_dev`) + shared scratch dimensionado pela CAPACIDADE
do cache (args host congelam na captura); (d) passo Mamba GPU
(`NSOS_MAMBA_GPU_STEP=1`) já era capture-safe; (e) BitLinear ganhou cache de
inferência do ternário (a absmean por-forward fazia D2H síncrono — abortaria a
captura e desperdiçava decode); (f) adoção GUARDADA: warm-up eager → captura
com snapshot/restore em falha → fallback eager permanente com razão em
`decode_graph_status()`. MoE/TTT declinam por design (roteamento host-synced;
device-resident routing é follow-up). Gate: `test_gpu_parity_decode_graph`
(sequência graph == eager token-a-token). Medição: pinned vs pageable D2H via
`nsos.bench_d2h_copy` + `NSOS_D2H_TIMING=1` no sampler; notebook
`colab/bench_decode_graph_t4.ipynb`.
