# NSOS — Análise de Otimização GPU & Limitações Arquiteturais

> **Status:** Documento de análise (read-only). Nenhuma modificação de código foi feita para produzi-lo.
> **Data da análise:** Pós-treino v7 (CPU, hybrid_medium 40M params).
> **Escopo:** `OXN/nsos/` (produto). Foco: bottlenecks CPU-GPU, oportunidades de aceleração, limitações estruturais.

---

## 1. Sumário Executivo

A arquitetura NSOS (Mamba2 SSD + Attention + MoE + BitLinear 1.58-bit) tem **fundamentos teóricos sólidos para GPU** mas a implementação atual está em **modo edge-first / CPU-only** com integração GPU **parcial e síncrona**. O estado é:

- ✅ **Kernels CUDA de qualidade** (BitNet com `__dp4a`, GEMM tiled, RMSNorm/LayerNorm warp-reduce)
- ✅ **Build CUDA configurado** e suportando arquiteturas 6.1 → 8.6 (Pascal → Ada)
- ❌ **Pipeline I/O é síncrono em todos os lugares** — `cudaMemcpy` bloqueante, `sleep_for(1ms)` no loader, sem streams
- ❌ **Mamba2 GPU fast-path só funciona em decode single-token** — treino batch é forçado para CPU
- ❌ **MoE routing per-token em CPU** — `std::partial_sort` por linha, experts sequenciais
- ❌ **Persistent kernel é stub vazio** (15 linhas, exit imediato)
- ⚠️ **`test_gpu_parity` segfault em CUDA 12.9** — bloqueio direto para CI GPU

**Diagnóstico em uma frase:** o projeto tem GPU como "checkbox" mas não como pipeline de primeira classe. Resultado direto: o treino atual usa `batch_size=3` na CPU porque essa é a única configuração que funciona end-to-end de forma estável.

**Ganho esperado com GPU enablement completo (target: GTX 1050 Ti):**

| Métrica | CPU hoje | 1050 Ti realista | Multiplicador |
|---------|----------|------------------|---------------|
| Throughput de treino | ~40s/step | ~6-12s/step | **3-6×** |
| Batch size estável | 3 | 16-32 | **5-10×** |
| Variância de loss | Alta (spikes 7→10→7) | Baixa | qualitativo |
| Tokens/s inferência | ~0.06 (40M) | 2-5 | **30-80×** |

> **Nota:** as estimativas acima são conservadoras e específicas para a 1050 Ti (768 cores, 4GB, sem tensor cores). Em GPUs modernas (RTX 3060+) os multiplicadores subiriam 3-5×. Ver **Seção 2** para o profile detalhado do hardware-alvo.

---

## 2. Profile do Hardware Alvo: GTX 1050 Ti

Esta análise é calibrada para o hardware atual de desenvolvimento — **NVIDIA GTX 1050 Ti (Pascal GP107)**. Toda referência a "GPU realista" no documento usa estas restrições.

### 2.1 Especificações Verificadas

| Item | Valor | Implicação para NSOS |
|------|-------|---------------------|
| Compute Capability | **sm_61** | ✅ Já incluído em `CMAKE_CUDA_ARCHITECTURES` (linha 22) |
| CUDA cores | 768 | ~2.1 TFLOPS FP32 — modesto mas usável |
| VRAM | **4 GB GDDR5** | ⚠️ **MAIOR LIMITAÇÃO** |
| Bandwidth | 112 GB/s | Memory-bound em ops grandes |
| TDP | 75W | Sem conector PCIe externo, OK para desktop |
| Tensor cores | **NENHUM** | ⛔ FP16 matmul **não acelera** |
| `__dp4a` (DP4A INT8) | Disponível em sm_61 | ✅ **BitNet 1.58-bit kernel funciona** |
| FP16 throughput | **1/64 do FP32** | ⛔ FP16 só serve pra economizar memória, não compute |
| Shared mem por bloco | 48 KB | ✅ Nosso seq_len=160 cabe folgado (640 bytes) |
| `cudaMemcpyAsync` + streams | Sim | ✅ Async básico funciona |
| `cuda::pipeline` (Ampere) | **NÃO** | ⛔ Sem cp.async, sem tile groups avançados |
| Cooperative groups | Básico (Volta+ tem mais) | ⚠️ Patterns simples apenas |
| Suporte CUDA toolkit | Até 12.x (deprecado em 13+) | ⚠️ Janela de 1-2 anos antes de quebrar |

### 2.2 Orçamento de Memória — Modelo Atual (40M params)

Com 4 GB VRAM, fazendo a conta para o `hybrid_medium` (40M params, d_model=512, 12 layers):

| Componente | Tamanho | Notas |
|-----------|---------|-------|
| Pesos FP32 | ~160 MB | 40M × 4 bytes |
| Optimizer Adam (m, v) | ~320 MB | 2× pesos |
| Workspace cuBLAS | ~50 MB | matmul tiling |
| Activations (batch=16, seq=160) | ~130 MB | cached para backward |
| Activations (batch=32, seq=160) | ~260 MB | |
| Activations (batch=32, seq=512) | ~420 MB | |
| Misc (pinned buffers, kernels) | ~80 MB | |
| **TOTAL batch=16, seq=160** | **~740 MB** | ✅ Folgado |
| **TOTAL batch=32, seq=160** | **~870 MB** | ✅ Confortável |
| **TOTAL batch=32, seq=512** | **~1.5 GB** | ⚠️ Possível mas apertado |
| **TOTAL batch=64, seq=512** | **~3-4 GB** | ⛔ No limite, risco de OOM |

**Recomendação:** começar com **batch=16-32, seq=160** (mesma seq_len atual). Tentar seq_len=320 só após Fase 3 (Mamba2 paralelo).

### 2.3 Realidade Específica do Pascal vs Roadmap Original

Três coisas mudam **para pior** com Pascal vs hardware moderno:

1. **FP16 é inútil para compute** (1/64 throughput) — **não dá pra usar mixed precision** como aceleração. Só serve pra economizar memória se necessário.
2. **Sem tensor cores** — operações tipo matmul rodam em SIMT puro, ~10× mais lento que Ampere/Ada equivalente.
3. **Sem `cuda::pipeline`** — não dá pra usar padrões modernos de overlap async memory + compute. Tem que usar streams + events tradicionais.

Três coisas funcionam **igual** ao hardware moderno:

1. **`__dp4a` para BitNet 1.58-bit** — Pascal sm_61 tem essa instrução INT8. O kernel BitNet em `kernels.cu:521-577` funciona nativamente.
2. **Streams + `cudaMemcpyAsync`** — async básico está disponível, suficiente para Fase 2 do roadmap.
3. **OpenMP CPU + GPU dispatch** — não muda em função do hardware.

### 2.4 Comparação Realista de Throughput

Para colocar em perspectiva (matmul de tamanho relevante):

| Hardware | FP32 TFLOPS | Speedup vs CPU típico |
|----------|-------------|------------------------|
| CPU (i7/Ryzen 8c) | ~0.1-0.2 | 1× (baseline) |
| **GTX 1050 Ti** | **2.1** | **10-20×** |
| RTX 3060 | 12.7 | 60-120× |
| RTX 4070 | 29.1 | 145-290× |

**A 1050 Ti dá um upgrade real (10-20× compute), mas é uma fração do que GPUs modernas oferecem.** Importante calibrar expectativa: não vamos chegar a "treino em minutos", vamos chegar a "treino em horas em vez de dia".

---

## 3. Estado Atual da Integração GPU

### 2.1 Build & Toolchain

**Arquivo:** `OXN/nsos/CMakeLists.txt` (linhas 141-183)

| Item | Status |
|------|--------|
| Default `NSOS_ENABLE_CUDA=ON` | ✅ |
| Arquiteturas alvo (`61;75;80;86`) | ✅ Pascal → Ada |
| C++17 para CUDA | ✅ |
| cuBLAS + cuRAT linkados | ✅ |
| `build-cuda/` configurado | ⚠️ Configurado mas nunca compilado completo |
| `NSOS_CUDA_SYNC=1` para testes | ✅ Força sync para parity check |

**Conclusão:** infraestrutura de build está pronta. Falta compilar e validar.

### 2.2 Kernels CUDA

**Arquivo:** `OXN/nsos/src/cuda/kernels.cu` (1354 linhas) — **alta qualidade**

Pontos fortes:
- Warp reductions com `__shfl_down_sync` (linhas 21-76)
- Tiled SGEMM 16×16 (linhas 428-486) — manualmente otimizado
- BitNet GEMM com `__dp4a` para 4× INT8 throughput (linhas 521-577) — **excelente para 1.58-bit**
- GQA causal attention com RoPE (linhas 937-1309)
- AdamW kernel CUDA (linhas 813-850)

Limitações:
- Shared memory oversubscription em GQA: `seq_len * sizeof(float)` falha em SM 6.1 (96KB max) — linha 1035
- MoE Top-K hardcoded para k=2 (linhas 1314-1353)

**Arquivo:** `OXN/nsos/src/cuda/mamba_kernels.cu` (349 linhas) — **funcional mas mal escalado**

Problema crítico:
```cpp
// Linhas 268-305 — comentário no próprio código:
// "launch each chunk sequentially to ensure correct state propagation"
for (int chunk = 0; chunk < num_chunks; ++chunk) {
    launch_chunk_kernel<<<...>>>(chunk);  // SEQUENCIAL no CPU side!
}
```
- `MAX_N=64` hardcoded (linha 20) — limite de dimensão de estado
- Chunk size fixo em 32 tokens (linha 21)
- `atomicAdd` em backward para grad_A — race condition em escala (linha 229)

**Arquivo:** `OXN/nsos/src/cuda/persistent_kernel.cu` (15 linhas) — **STUB**

```cpp
// Implementação completa atual:
__global__ void universal_persistent_kernel() {
    // Returns immediately to avoid hanging
    return;
}
```
**Não funcional.** Bloqueia uso real de persistent kernel pattern.

---

## 3. Bottlenecks CPU-Side (Caminho Crítico)

### 3.1 SmartLoader — `sleep_for(1ms)` em vez de async real

**Arquivo:** `OXN/nsos/src/smart_loader.cpp` linhas 50-56

```cpp
void SmartLoader::wait_for_completion(Tensor *t) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));  // ← FALSO ASYNC
}
```

**Impacto:**
- `IoRequest` tem `std::atomic<bool> completed` (linha 16) **mas nunca é checado**
- Cada espera queima 1ms fixo independente do tempo real do I/O
- Sem `cudaHostRegister()` → sem pinned memory → transferências GPU 2-3× mais lentas
- Sem `cudaMemcpyAsync()` → sem overlap de I/O com compute

**Ganho potencial:** eliminar 1ms × N waits por batch + permitir overlap = redução de 20-40% no tempo de step.

### 3.2 DataLoader — Sem Pipelining

**Arquivos:** `OXN/nsos/src/dataloader.cpp` linhas 17-26, `dataloader_v2.cpp` linhas 22-32

- Mutex lock em cada `next()` → serialização total worker↔consumer
- Tensors alocados sempre `Device::CPU` (linha 36) → transferência blocking depois
- Queue limitada a 5 (v1) ou 16 (v2) → sem true streaming prefetch
- Sem prefetch de N+1 batch enquanto N treina

**Ganho potencial:** 2-queue async prefetch + alocação direta em pinned memory = -30% wallclock por step.

### 3.3 Tensor Transfer — `cudaMemcpy` Bloqueante

**Arquivo:** `OXN/nsos/src/tensor.cpp` linhas 251-260, 887

```cpp
if (dst_device == Device::GPU || src_device == Device::GPU) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);  // ← BLOQUEIA TUDO
}
```

- `Tensor::to(Device::GPU)` bloqueia trainer
- `cudaMemcpy(...Default)` na default stream (stream 0) — sem overlap
- Linha 539-565: cuBLAS calls em loop sequencial sem stream assignment
- Linha 564: `sync_cuda()` após batch matmul completo

**Ganho potencial:** com streams próprios, cublasSetStream, e cudaMemcpyAsync = 1.5-2× throughput em workloads small-batch.

### 3.4 Bindings Python — GIL Bloqueado em Operações Longas

**Arquivo:** `OXN/nsos/src/bindings.cpp` linhas 26-50

```cpp
py::array_t<float> tensor_to_numpy(Tensor& tensor) {
    auto* holder = new Tensor(
        tensor.get_device() == Device::GPU ? tensor.cpu() : tensor.clone()
    );  // ← Bloqueia em GPU sync
}
```

- Sem `py::gil_scoped_release()` em volta de cuBLAS/kernels
- Toda leitura de métrica força sync GPU→CPU completo
- Logging de loss em cada step → sync forçado

**Ganho potencial:** liberar GIL em hot path = thread Python livre, métricas bufferizadas.

### 3.5 Memory Allocator — Fallback malloc no Hot Path

**Arquivo:** `OXN/nsos/include/nsos_arena.h` linhas 68-80

```cpp
if (block->offset + padded > block->size) {
    auto& fallbacks = get_thread_fallbacks();
    ArenaBlock* fallback = new ArenaBlock(padded);  // ← malloc no hot path!
    fallbacks.push_back(...);
}
```

- 512 MB pool por thread (linha 103) mas fallbacks acumulam sem reuso
- Alinhamento 64B (`posix_memalign`) — cuBLAS prefere 128B+

---

## 4. Limitações Arquiteturais (Não Resolvidas Por GPU Sozinho)

### 4.1 Mamba2 Scan é Intrinsecamente Sequencial em Tempo

**Arquivo:** `OXN/nsos/src/mamba2.cpp` linhas 184-204

```cpp
for (int b = 0; b < batch; ++b) {
    std::fill(state.begin(), state.end(), 0.0f);
    for (int t = 0; t < seq; ++t) {           // ← state[t] depende de state[t-1]
        for (int d = 0; d < dim; ++d) {
            state[d] = state[d] * decay + b_t[d] * x_t[d];
            y_t[d] = std::tanh(state[d]) * c_t[d];
        }
    }
}
```

**Limitação fundamental:** o estado SSM em t depende de t-1. Mesmo na GPU, isso só pode ser paralelizado via:
- **Parallel scan** (Blelloch): dobra trabalho mas paraleliza em log(seq)
- **Chunked scan** (Mamba1 paper): paralelo dentro de chunks, sequencial entre chunks
- **Selective scan kernel** (mamba-ssm reference): kernel CUDA dedicado com state sharing

A implementação atual nem usa chunked scan paralelo — força loop sequencial CPU mesmo quando tensores estão em GPU. **Esse é o maior bloqueador arquitetural.**

### 4.2 MoE Routing — `std::partial_sort` Per Token em CPU

**Arquivo:** `OXN/nsos/src/jamba.cpp` linhas 1171-1183

```cpp
for (int row = 0; row < rows; ++row) {  // ← per-token CPU
    std::partial_sort(ranked_experts.begin(), ...);
    // ...
}
```

E os experts são computados sequencialmente:
```cpp
// Linhas 1202-1244
for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    if (skipped) continue;
    // gather → forward → scatter (com CPU↔GPU copy por expert!)
}
```

**Ganho potencial em GPU:** kernel fused top-k + experts paralelos em streams diferentes = 5-10× speedup do MoE block.

### 4.3 Optimizer Step — OpenMP em CPU mesmo com Tensores em GPU

**Arquivo:** `OXN/nsos/src/optimizers.cpp` linhas 29-37

```cpp
#pragma omp parallel for
for(int i=0; i<p->data.size; ++i) {
    // Adam step em CPU
}
```

✅ Bem paralelizado para CPU
❌ Não usa o kernel CUDA AdamW que existe em `kernels.cu` linhas 813-850

**Ganho potencial:** trocar dispatch para usar kernel CUDA quando tensores estão em GPU = elimina round-trip GPU→CPU→GPU por step.

### 4.4 Vocabulário 4827 Tokens — Limitação de Treinamento, Não de GPU

Não é problema de GPU mas vale documentar: o tokenizer foi treinado com vocab pequeno e palavras inglesas comuns ficam fragmentadas. Isso aumenta:
- Tamanho de sequência efetivo (mais tokens por palavra)
- Dificuldade de gerar respostas longas (limite seq_len=160)
- Dificuldade de generalização para outros idiomas

**Solução:** retreinar tokenizer com 16k-32k tokens em corpus maior. Não relacionado a GPU.

### 4.5 Por Que `batch_size=3`?

**Arquivo:** `OXN/nsos/scripts/train_curriculum.py` linhas 87, 126, 389-431

Não há limite explícito no código. As causas combinadas:
1. **Mamba2 forçado para CPU** — batch maior = mais tempo CPU sequencial
2. **MoE gather/scatter** — overhead linear no batch (linhas 1212-1216)
3. **Memory budget conservador** — sem GPU para spillover
4. **Variância de gradiente alta** com batch pequeno → spikes na loss que vimos no v7/v8

**Em GPU:** com Mamba2 paralelo + MoE paralelo, batch_size=32-64 é trivial em GPU de 8GB+.

---

## 6. Roadmap de GPU Enablement — Adaptado para GTX 1050 Ti

> **Princípios para 1050 Ti:**
> - Não dependa de FP16 mixed precision (tensor cores ausentes, FP16 SIMT é 1/64 do FP32)
> - Cap batch=32 inicial; subir só após Mamba2 paralelo
> - Use `__dp4a` agressivamente (BitNet packed é uma das poucas vantagens reais do hardware)
> - Sem `cuda::pipeline` — use streams + events tradicionais
> - Ficar atento a CUDA 13+ que vai dropar Pascal (janela de ~1-2 anos)

### Fase 1 — Validação Básica (~1-2 semanas)

**Objetivo:** GPU funcional para treino na 1050 Ti, batch ≥ 16, sem mudanças arquiteturais.

1. **Resolver segfault em `test_gpu_parity`** em CUDA 12.9
   - Investigar com `nvprof` (mais confiável em Pascal que `nsys`)
   - **Provavelmente NÃO é o GQA kernel** — nosso seq_len=160 só usa 640 bytes de shared mem (limite é 48KB)
   - Suspeito mais provável: **`persistent_kernel.cu` stub** sendo invocado e crashando em ponteiro nulo
   - Plano B: kernel `bitlinear` ou `mamba` em algum edge case
2. **Compilar `build-cuda`** completo (atualmente só configurado, nunca compilou completo)
   - `cmake -DNSOS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 ...`
   - Forçar sm_61 explicitamente para evitar JIT em runtime
3. **Validar treino end-to-end** com `mamba_small` (perfil baseline, mais leve que `hybrid_medium`)
4. **Subir batch_size progressivo na 1050 Ti**:
   - 3 (baseline) → 8 (validar) → 16 (alvo realista) → 32 (limite)
   - Monitorar VRAM com `nvidia-smi` — se passar 3.5 GB, recuar
5. **Habilitar `NSOS_GPU_CI=true`** **só depois** que tudo passar local

**Bloqueador conhecido:** Mamba2 vai continuar lento porque `ssd_forward()` força CPU. Mas Attention, MoE, BitLinear e optimizer já vão dar 5-8× speedup.

**Ganho realista pós-Fase 1:** treino atual (~12h) → ~3-4h, com batch=16-32.

### Fase 2 — Async I/O Pipeline (~1-2 semanas)

**Objetivo:** Eliminar `cudaMemcpy` bloqueante, implementar prefetch — **especialmente importante na 1050 Ti** porque a bandwidth (112 GB/s) é metade de uma 3060 e overlap se torna mais valioso.

1. **Reescrever `SmartLoader::wait_for_completion`** para polling real do `completed` flag (já existe como `std::atomic<bool>`)
2. **Substituir `cudaMemcpy` → `cudaMemcpyAsync` + streams** em `tensor.cpp::to()`
3. **Implementar pinned memory** para batches e weights — **na 1050 Ti, o ganho de async é maior** porque a CPU termina rápido vs GPU
4. **Pipeline 2-queue prefetch** em `dataloader_v2`
5. **`py::gil_scoped_release`** em chamadas longas em `bindings.cpp`

**Ganho esperado na 1050 Ti:** 25-35% redução no wallclock por step. (Em GPUs com mais bandwidth o ganho seria menor; aqui é maior porque o overlap com I/O é crítico).

**NÃO usar:** `cuda::pipeline` (Ampere+), `cp.async`, async copy via tile groups. Pascal não suporta.

### Fase 3 — Mamba2 GPU Real (~2-3 semanas) — **Maior impacto**

**Objetivo:** Substituir o loop sequencial CPU por scan chunked-paralelo GPU.

1. **Implementar chunked parallel scan** seguindo Mamba1 (Mamba2 reference é mais complexa) — chunk_size=32 já está no código
2. **Substituir `ssd_forward` na rota `Device::GPU`** para usar kernel novo
3. **Manter CPU fallback** para edge deployment
4. **Validar paridade numérica** vs CPU em `test_gpu_parity` (tolerância 1e-4 conservadora)

**Considerações específicas Pascal:**
- Sem `cuda::pipeline` ⇒ usar streams + `cudaStreamWaitEvent` para sincronizar chunks
- Cooperative groups básicos ⇒ scan dentro de bloco usa shared memory tradicional, não tile group reductions
- `MAX_N=64` (linha 20 `mamba_kernels.cu`) é OK para nosso d_state=64

**Ganho esperado na 1050 Ti:** Mamba2 forward 4-8× mais rápido (em hardware moderno seria 8-15×). Isso desbloqueia:
- batch_size 32 estável (limite de VRAM, não de compute)
- seq_len 320-512 (vs 160 atual) — limite é VRAM
- Treinos 12h → ~1.5-2h (composição com Fases 1+2)

### Fase 4 — MoE Paralelo + Persistent Kernel (~2 semanas)

**Objetivo:** Eliminar bottlenecks restantes do MoE.

1. **Top-K kernel CUDA** substituindo `std::partial_sort` per-row em `jamba.cpp:1171-1183`
2. **Experts em streams paralelos** (CUDA streams + `cudaStreamWaitEvent`) — na 1050 Ti, com 768 cores e 2 experts ativos por token, paralelismo entre experts dá overlap real de compute
3. **Gather/scatter fused kernel** evitando CPU↔GPU round-trip por expert (`jamba.cpp:1212-1216, 1237-1241`)
4. **Decisão sobre `persistent_kernel.cu`:** **remover do build** ao invés de implementar
   - Persistent kernel pattern é complexo e mais útil em GPUs com many-SM (RTX 30+)
   - Na 1050 Ti com 6 SMs, ganho marginal — não vale o trabalho
   - Marcar explicitamente como `legacy/` ou deletar

### Fase 5 — Quantização GPU Native + Edge Pack (~3-4 semanas)

**Objetivo:** Habilitar GPU packed (1.58-bit nativo) e otimizar inferência. **Esta fase é onde a 1050 Ti pode brilhar** — `__dp4a` é uma das poucas instruções modernas que ela tem.

1. **Pipeline de quantização float → packed** no `nsos_serializer.h`
   - Reduz pesos de 160 MB FP32 → ~20 MB packed (8×) — libera VRAM para batch maior
2. **Validar `bitnet_gemm` packed** em produção
   - Kernel já existe em `kernels.cu:521-577` usando `__dp4a` (4× INT8 throughput)
   - **Confirmação importante:** Pascal sm_61 tem `__dp4a` — o kernel funciona nativamente
3. **Edge pack carregável diretamente em GPU** sem dequantização CPU
4. **NÃO implementar FP16 variants** — na 1050 Ti FP16 SIMT é mais lento que FP32. Pular essa otimização aqui.

**Ganho esperado na 1050 Ti:**
- Inferência: 2-5 tokens/s → **15-30 tokens/s** com packed (4× a mais que FP32)
- VRAM: 160 MB pesos → 20 MB packed (libera 140 MB para batch maior)
- Treinamento: pouco impacto (treino usa FP32 para gradientes)

---

## 7. Quantificação de Ganhos Esperados — GTX 1050 Ti

### 7.1 Treinamento por Fase (cumulativo, batch_size adequado)

| Cenário | Time/step | Batch | Treino completo | Ganho vs CPU |
|---------|-----------|-------|-----------------|--------------|
| **CPU hoje** (baseline) | ~40s | 3 | ~12h | 1× |
| **Pós-Fase 1** (GPU básico) | ~10-12s | 16 | ~3-4h | 3-4× |
| **Pós-Fase 2** (async I/O) | ~7-9s | 16-32 | ~2-3h | 4-6× |
| **Pós-Fase 3** (Mamba2 paralelo) | ~5-7s | 32 | ~1.5-2h | 6-8× |
| **Pós-Fase 4** (MoE paralelo) | ~4-6s | 32 | ~1-1.5h | 8-12× |
| **Pós-Fase 5** (packed + edge) | ~3-5s | 32-48 | ~45min-1h | 12-15× |

> Números calibrados para GTX 1050 Ti. Em RTX 3060 multiplicar por ~3×; em RTX 4070+ por ~6-8×.

### 7.2 Inferência (após Fase 5) — Modelo 40M params

| Métrica | CPU baseline | 1050 Ti FP32 | 1050 Ti Packed (`__dp4a`) |
|---------|-------------|--------------|----------------------------|
| Tokens/s decode | ~0.06 | 2-5 | **15-30** |
| Latência 1º token | ~5s | ~0.5s | ~0.2s |
| 50 tokens completos | ~14 min | ~10-25s | **~2-3s** |
| Custo VRAM (apenas pesos) | 160 MB (RAM) | 160 MB | 20 MB |

> **Por que packed dá 4× a mais:** `__dp4a` faz 4 multiply-add INT8 em uma instrução. Ainda assim, 1050 Ti tem só 768 cores — em RTX 3060 (3584 cores) o mesmo packed daria 70-150 tokens/s.

### 7.3 Capacidade de Modelo

| Configuração | CPU hoje | 1050 Ti pós-roadmap |
|--------------|----------|---------------------|
| Params máximo viável | ~40M (limite paciência) | ~80-120M (limite VRAM 4GB) |
| Batch size estável | 3 | 16-32 |
| Seq_len treino | 160 | 320 (com Mamba2 paralelo) |
| Variância de gradiente | Alta (spikes 7→10) | Baixa (curva suave) |

### 7.4 Limites Físicos da 1050 Ti que NÃO Mudam

Importante calibrar expectativa — algumas coisas não dependem do nosso roadmap:

- **VRAM 4 GB é teto absoluto** — modelos > 120M params não cabem com Adam state
- **2.1 TFLOPS FP32** vs RTX 4090 (82 TFLOPS) = ~40× mais lento que ponta
- **Sem tensor cores** — não vamos chegar perto dos números de papers de eficiência
- **Bandwidth 112 GB/s** — em workloads memory-bound, esse é o limite duro
- **Sem NVLink** — multi-GPU não é opção no setup atual
- **CUDA 13+ vai parar de suportar** — janela ~1-2 anos antes de upgrade obrigatório

**Uso ideal da 1050 Ti:** **desenvolver e validar o pipeline GPU**. Treinos de produção em escala precisariam GPU melhor.

---

## 7. Riscos & Pontos de Atenção

### 8.1 Riscos Técnicos — Específicos da 1050 Ti

| Risco | Probabilidade | Mitigação |
|-------|---------------|-----------|
| `test_gpu_parity` segfault não trivial | Alta | `nvprof` (nsys tem suporte limitado a Pascal); isolar `persistent_kernel.cu` primeiro |
| Mamba2 parallel scan numericamente diferente | Média | Validar vs CPU com tolerância 1e-4 |
| **OOM com batch=32 + seq=512** | **Alta** | Cap batch=16 ou seq=320 inicialmente; monitorar com `nvidia-smi` |
| **Termal throttling em treino longo** | **Média** | TDP 75W sustentado pode subir a temp; verificar `nvidia-smi -q -d TEMPERATURE` |
| Pinned memory pressão no host | Baixa | Limitar quantidade alocada (1050 Ti não tem muito RAM mapeável) |
| Race conditions em MoE async | Alta | Usar `cudaStreamWaitEvent` rigorosamente |
| Drift de determinismo com streams | Alta | `OXN/scripts/verify_determinism.py` precisa rodar a cada PR |
| **CUDA 13 dropa Pascal** | **Certa em 1-2 anos** | Manter pinning em CUDA 12.x; planejar upgrade de hardware |
| **`__dp4a` em algum edge case** | Baixa | Validar BitNet kernel especificamente em sm_61 antes de declarar Fase 5 pronta |

### 7.2 Riscos de Escopo

- **Não cair na armadilha do KernelOpen / persistent_kernel agora** — é trabalho de pesquisa, não de produto
- **Não trocar para PyTorch/JAX** — perderia toda a tese arquitetural (1.58-bit + edge-first)
- **Manter CPU baseline funcionando** — edge deployment ainda é caso de uso real

### 7.3 Dívidas Técnicas que Não São GPU

Importante registrar separadamente — não são resolvidas por GPU:

- `jamba.cpp` ainda é god-class (`ARCHITECTURE_RISK.md` Stage 2/3 não iniciado)
- Vocab tokenizer 4827 — limitação de qualidade que GPU não resolve
- Phases 5/6 do curriculum (verifier/memory) com qualidade de dados ruim — vimos isso destruir treinos
- Holdout suite pequeno e não diversificado

---

## 8. Comparação com Estado da Arte

A arquitetura NSOS faz escolhas únicas:

| Decisão | NSOS | Mainstream (PyTorch + Llama) |
|---------|------|------------------------------|
| Quantização | 1.58-bit nativo (BitNet) | FP16/INT8 pós-treino |
| SSM core | Mamba2 SSD | Atenção pura |
| Sparsity | MoE 8 experts ativos 2 | Dense |
| Memória | OxtaMem geodésico | RAG vetorial |
| Target | Edge CPU + GPU opt-in | GPU-first |
| Tokenizer | 4827 tokens BPE custom | 32k+ SentencePiece |

**Vantagens potenciais com GPU funcional:**
- Mamba2 paralelo escala melhor que atenção em seq longas (O(n) vs O(n²))
- BitNet `__dp4a` é nativamente eficiente em GPUs Turing+
- MoE sparse compute reduz FLOPs efetivos

**Desvantagens persistentes:**
- Ecossistema (modelos pré-treinados, eval suites) é todo PyTorch
- Tokenizer pequeno limita qualidade de geração
- Comunidade pequena vs frameworks estabelecidos

---

## 9. Próximos Passos Recomendados

### Imediato (após validar v8 amanhã)

1. **Testar inferência v8** — confirmar que modelo responde coerentemente
2. **Decidir prioridade GPU vs outras melhorias** (curriculum, dados, etc.)
3. **Reservar tempo para Fase 1** (~1 semana de validação básica)

### Curto prazo (próximas 2-4 semanas)

1. Fase 1 (validação básica GPU)
2. Fase 2 (async I/O)
3. Re-treinar v9 em GPU com batch=32

### Médio prazo (1-3 meses)

1. Fase 3 (Mamba2 paralelo) — **maior ROI técnico**
2. Tokenizer maior (16k-32k) com retreino completo
3. Fase 4 (MoE paralelo)

### Longo prazo (3-6 meses)

1. Fase 5 (quantização GPU native)
2. Scaling para 100M-500M params
3. Edge deployment validation com modelos quantizados

---

## 10. Referências do Código

### Arquivos críticos para GPU enablement

| Arquivo | Linhas relevantes | Issue |
|---------|-------------------|-------|
| `OXN/nsos/src/cuda/persistent_kernel.cu` | 1-15 | STUB completo |
| `OXN/nsos/src/cuda/mamba_kernels.cu` | 268-305 | Sequential dispatch |
| `OXN/nsos/src/cuda/kernels.cu` | 1035 | GQA shared mem overrun |
| `OXN/nsos/src/smart_loader.cpp` | 50-56 | `sleep_for(1ms)` falso async |
| `OXN/nsos/src/dataloader.cpp` | 17-26, 62-72 | Sem prefetch |
| `OXN/nsos/src/dataloader_v2.cpp` | 22-32 | Mesma issue |
| `OXN/nsos/src/tensor.cpp` | 251-260, 539-565, 887 | `cudaMemcpy` blocking |
| `OXN/nsos/src/mamba2.cpp` | 184-204, 397-453 | CPU sequential, GPU só single-token |
| `OXN/nsos/src/jamba.cpp` | 1150-1247 | MoE per-row CPU |
| `OXN/nsos/src/optimizers.cpp` | 29-37 | OpenMP CPU mesmo com GPU tensors |
| `OXN/nsos/src/bindings.cpp` | 26-50 | Sem GIL release |
| `OXN/nsos/include/nsos_arena.h` | 68-80, 103 | Fallback malloc |
| `OXN/nsos/tests/test_gpu_parity.cpp` | 32-36, 134-154 | Segfault em CUDA 12.9 |

### Documentação relacionada

- `OXN/nsos/docs/ARCHITECTURE_RISK.md` — Stage 2/3 (god-class refactor)
- `OXN/nsos/docs/NSOS_VALIDATION_STATUS.md` — Status validação por modo
- `OXN/nsos/docs/RELEASE.md` — Gates atuais (CPU first)
- `OXN/nsos/docs/MODEL_PACKS.md` — Formato pack atual
- `legacy/docs/INDUSTRIAL_ROADMAP.md` — Roadmap original

---

## 12. Conclusão — Considerando a GTX 1050 Ti

O projeto NSOS tem uma **arquitetura arrojada e teoricamente sólida** (Mamba2 + MoE + BitNet 1.58-bit) mas está em **modo demonstração CPU-only**. Para a 1050 Ti, o caminho realista é:

1. **Curto prazo (Fases 1-2):** corrigir o pipeline I/O destrava 3-6× sem mudar nada arquitetural — chega em **~2-3h de treino vs 12h atual**
2. **Médio prazo (Fase 3):** Mamba2 GPU paralelo é o maior ROI técnico — chega em **~1.5-2h de treino**, desbloqueia batch=32 e seq_len=320
3. **Longo prazo (Fases 4-5):** packed quantization native (`__dp4a`) é onde a 1050 Ti pode brilhar — **inferência 30× mais rápida**

**O que a 1050 Ti consegue fazer:**

✅ Treinar o `hybrid_medium` 40M em 1-2h em vez de 12h
✅ Suportar batch=16-32 com seq_len=160-320 (vs batch=3 hoje)
✅ Inferência packed em 15-30 tokens/s (vs 0.06 hoje)
✅ Validar todo o pipeline GPU end-to-end
✅ Provar a tese 1.58-bit BitNet em hardware modesto

**O que a 1050 Ti NÃO consegue:**

❌ Modelos > 100-120M params (limite VRAM)
❌ FP16 acceleration (sem tensor cores, FP16 SIMT é 1/64 do FP32)
❌ `cuda::pipeline` ou padrões Ampere modernos
❌ Multi-GPU (sem NVLink)
❌ Suporte futuro além de CUDA 12.x

**O melhor sinal:** o código sabe que precisa ser async (tem `IoRequest` com `std::atomic<bool> completed`, tem `cudaMemcpyAsync` disponível, tem `cublasSetStream`), mas a integração nunca foi terminada. Isso significa que a transição CPU → GPU pode ser feita **incremental e segura**, sem grandes refactors arquiteturais — e **inteiramente na 1050 Ti como hardware de desenvolvimento**.

**Recomendação estratégica:**

A 1050 Ti é hardware **suficiente para validação e desenvolvimento**, mas não é hardware de **produção em escala**. Use ela para:
- Provar que cada fase do roadmap funciona
- Treinar modelos `hybrid_medium` (40M) para validação qualitativa
- Validar a tese arquitetural do NSOS (BitNet + Mamba2 + MoE)

Com o pipeline GPU validado, o upgrade para uma RTX 3060+ multiplica os ganhos em 3-5× automaticamente — sem mudar uma linha de código.

---

## 13. Anexo: Estimativas de Compute para Decisão de Compra Futura

Para colocar a 1050 Ti em contexto se em algum momento considerar upgrade:

| GPU | VRAM | TFLOPS FP32 | Tensor Cores | Preço usado (BR ~2026) | Speedup vs 1050 Ti |
|-----|------|-------------|--------------|------------------------|---------------------|
| GTX 1050 Ti | 4 GB | 2.1 | Não | (atual) | 1× |
| GTX 1660 Super | 6 GB | 5.0 | Não | ~R$ 800 | ~2.5× |
| RTX 3060 12GB | 12 GB | 12.7 | Sim | ~R$ 1500 | ~6× + FP16 4× |
| RTX 3060 Ti | 8 GB | 16.2 | Sim | ~R$ 1700 | ~7-8× + FP16 4× |
| RTX 4060 16GB | 16 GB | 15.1 | Sim (4ª gen) | ~R$ 2200 | ~7-9× + FP16 4× |
| RTX 4070 12GB | 12 GB | 29.1 | Sim (4ª gen) | ~R$ 3500 | ~14× + FP16 4× |

**Pontos importantes:**
- **RTX 3060 12GB é o "sweet spot"** se quiser modelos > 100M params
- **Tensor cores triplicam o ganho** quando habilitar FP16 mixed precision (Fase 5 alternativa)
- VRAM > 8GB importa mais que TFLOPS para modelos grandes

Mas o ponto principal: **toda a engenharia de pipeline GPU desenvolvida na 1050 Ti se aproveita 100% em qualquer GPU futura**. O código não precisa mudar.

---

**Última atualização:** Análise realizada com treino v8 em execução (CPU). Sem modificações de código. Calibrada para hardware-alvo: NVIDIA GTX 1050 Ti (Pascal GP107, 4GB VRAM, sm_61).
