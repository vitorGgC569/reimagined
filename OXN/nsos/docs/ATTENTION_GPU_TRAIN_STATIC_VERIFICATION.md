# Verificação estática — caminho GPU de treino da atenção
### Protocolo: toda constatação por análise estática; execução real só no Colab.

**Escopo:** `src/cuda/attention_train_kernels.cu` + wiring em `src/jamba.cpp`
(branch de save no forward batched-exato; `Attention::backward_exact_gpu`).
Referência de verdade: o caminho host existente (mantido intacto como braço de
paridade via `NSOS_ATTN_BWD_HOST=1`).

## 1. Equivalência matemática (host ⇄ GPU), item a item

| # | Host (jamba.cpp) | GPU | Veredito estático |
|---|---|---|---|
| 1 | split KV: `k=kv[...,:kvd]; v=kv[...,kvd:]` por (b,s) | `kv_split_kernel`, r=b·S+s, mesmas fatias | idêntico |
| 2 | RoPE fwd: `r0=q0·c−q1·s; r1=q0·s+q1·c`, `pos=min(s,max−1)`, pares (i,i+hd/2) | `rope_apply_kernel dir=+1`, mesmos índices/clamp/tabelas (upload 1× de cos/sin) | idêntico |
| 3 | `kv_head = min(h/group, KV−1)` | gather: `sh=min(h/group, src_heads−1)` (group>1); group==1 ⇒ sh=h (e KV==H) | idêntico |
| 4 | `score = (Σ_d q·k)·scale`, máscara `j≤i ∧ j<valid ∧ i<valid`, max/exp/Σ, `inv=1/max(Σ,1e-9)`, mascarados=0; linha i≥valid intocada (zeros) | GEMM Q·Kᵀ cru; `masked_softmax`: scale aplicado ANTES do max (mesma ordem), mesmos limites (`limit=min(i,valid−1)`), mesmo guard 1e-9, zeros idênticos, linha i≥valid zerada | idêntico |
| 5 | `d_probs_j = Σ_d dO·v` só p/ j<valid; `row_dot=Σ d_probs·P` | GEMM dP=dO·Vᵀ em todo j; `rowdot=Σ P·dP` — P=0 fora da máscara ⇒ termos extras nulos | equivalente (prova: P_j=0 ⇒ contribuição 0) |
| 6 | `d_score=P·(dP−row_dot)`; scale aplicado depois em dQ/dK; **dV usa P sem scale** | `dS=scale·P·(dP−rowdot)` (scale dobrado em dS); dV = Pᵀ·dO (sem scale) | idêntico (álgebra linear da dobra) |
| 7 | `dQ_i += scale·d_score·K_j` | GEMM dS·K: Σ_j dS[i,j]·K[j,d] | idêntico |
| 8 | `dK_j += scale·d_score·Q_i` (acumulado pelos q-heads do grupo) | GEMM dSᵀ·Q por q-head + `reduce_group` (último kv absorve heads excedentes, espelhando o `min`) | idêntico |
| 9 | `dV_j += P·dO_i` (acumulado pelo grupo) | GEMM Pᵀ·dO + `reduce_group` | idêntico |
| 10 | linhas i≥valid: puladas ⇒ grad zero; colunas j≥valid: máscara ⇒ zero | dS linha 0 / coluna 0 pelos itens 4-6 ⇒ GEMMs produzem zeros | equivalente (zeros propagados) |
| 11 | RoPE bwd: `g0·c+g1·s; −g0·s+g1·c`, start_pos=0; **V sem rotação** | `rope_apply_kernel dir=−1` em dQ,dK; dV intocado | idêntico |
| 12 | montagem: concat heads → `[B,S,d_model]` (tokens<valid; resto zero) | `reshape` (identidade de layout — [B,S,H,hd] ≡ [B,S,d_model] contíguo); zeros já presentes (item 10) | idêntico |
| 13 | `grad_kv_input[...,:kvd]=dK; [...,kvd:]=dV` | `kv_concat_kernel` | idêntico |
| 14 | projeções backward + add + reshape por rank | mesmo código (BitLinear GPU) | idêntico |

## 2. Segurança de memória, streams e vida útil

- Todos os intermediários são `Tensor` (pool) vivos em escopo durante o uso;
  `raw_data()` não sincroniza; kernels e cuBLAS compartilham o stream default ⇒
  ordenação garantida; uploads pequenos (valid, tabelas RoPE) via `cudaMemcpy`
  síncrono.
- Softmax in-place sobre `scores` (saída fresca do matmul, dono exclusivo —
  sem alias). `reshape` é VIEW (verificado em tensor.cpp:1526) ⇒ `clone()`
  antes do RoPE in-place no save do forward.
- `attn_valid_device_buffer`: buffer persistente (cresce sob demanda), thread
  único de treino, falha de alloc ⇒ **throw** (sem caminho silencioso de grad
  errado).
- Erros: cada grupo de kernels seguido de `attn_train_check_cuda`
  (cudaGetLastError); GEMMs cobertos por `cublas_check` no Tensor::matmul.
- Build CPU: branches sob `#ifdef USE_CUDA`; chaves do escopo balanceadas nos
  dois modos (gate `gpu_saved_done` e fechamento ambos dentro do ifdef);
  `backward_exact_gpu` declarado e nunca referenciado no CPU build (ODR ok).

## 3. Condições de despacho (conservadoras)

GPU backward exige: cache exato salvo em GPU ∧ `gpu_custom_kernels_supported()`
∧ `head_dim % 2 == 0` ∧ `NSOS_ATTN_BWD_HOST≠1`. Qualquer outra condição ⇒
caminho host inalterado (byte-idêntico ao comportamento anterior).
Save GPU no forward exige `used_gpu_exact_forward` ∧ projeções em GPU; senão,
bloco host original intacto.

## 4. O que a análise estática NÃO pode fechar (pré-registrado p/ Colab)

- **Numérica FP (reassociação)**: GEMMs reassociam somas ⇒ paridade esperada
  ≤1e-3 nas losses (FP32; com BF16 a tolerância é maior — paridade roda em
  FP32). Gate: `scripts/attn_bwd_parity.py` (braços NSOS_ATTN_BWD_HOST=1 vs 0,
  mesma seed, 3 steps; predição: max|Δloss| ≤ 1e-3).
- Ganho de tempo real (predição: step T4 do v11 cai de ~16 s para ~1-2 s, pois
  o backward host O(B·H·S²·hd) single-thread + migrações UM saem do caminho).

## 4b. VEREDITO DO GATE (Colab T4, 2026-06-10) — PASS (D1) + descoberta

Histórico do instrumento (cada falha pré-registrada e corrigida): v1 media
trajetória através do Adam (caos de sign-flip); v2 capturava grads com atributo
inexistente; v3 gateava rel por-parâmetro (explode em normas ~0); v4 tinha o
controle poluído por `mamba.out_proj` (todo layer tem um). **v5** = 4 braços
(host×2, gpu×2) com piso de ruído + classificador `.attn.` estrito.

Resultado: **efeito host-vs-gpu 3.53e-1 ≈ pisos host-vs-host 3.24e-1 /
gpu-vs-gpu 3.43e-1 (D1-PASS)**; no topo (layer 11), ruído do próprio host
(alvo 1.15e-1) > diferença host-vs-gpu (8.0e-2). O kernel é indistinguível do
não-determinismo pré-existente. **Promovido (default GPU).**

**Descoberta colateral (pré-registrada como desfecho possível): o treino NUNCA
foi determinístico** — mesma seed/dados/código divergem ~32% nos gradientes em
1 step. Fontes: atomicAdd no scatter-add do embedding, grad_A do Mamba, scatter
do MoE e na redução do escalar de loss da CE (o próprio loss oscila 1 ulp:
9.803181 vs 9.803182), amplificadas ×~3/camada pelo backprop. O gate
verify_determinism cobre apenas seeding de Tensor.random — não gradientes.
Implicação: claims de "replay determinístico" valem para inferência/seed, não
para treino. Backlog (não bloqueante): modo NSOS_DETERMINISTIC com reduções
segmentadas ordenadas nos 4 pontos de atomic.

## 5. Limitações conhecidas (honestas)

- O forward batched-exato ainda recomputa RoPE para os SAVES (clone+rotação) em
  vez de o kernel GQA exportá-los — custo pequeno (2 kernels elementwise),
  eliminação total exigiria mudar o kernel fundido (fora do escopo de polish).
- Caminho sparse (SSA) continua host (OFF no perfil v11).
- `softmax`/transpose rank>2 genéricos continuam sem GPU — auditados como FORA
  do hot path de treino (atenção não os usa; BitLinear usa rank-2 coberto).
