# Colab GPU Validation — NSOS Fases 1 & 2

Guia copy-paste para validar **em GPU de verdade (T4/A100/L4)** o trabalho GPU-first:
o caching allocator (pool, ~3.9× medido na 1050 Ti), a precisão mista por arquitetura, os
gates da Fase 1, e o **loop de paridade 1e-4** que destrava qualquer kernel novo
de Fase 2 (parallel-prefix scan etc.) no padrão ouro.

> Por que Colab: o 1050 Ti/4GB valida correção, **não escala** (e trava o desktop
> sob pressão de UM). Em T4/A100 os batches grandes cabem, a ocupação sobe e o
> FP16 entra em V100/T4; BF16 entra em A100/L4/H100.

Runtime recomendado: **A100** (40 GB) ou **T4** (16 GB). `Runtime → Change runtime
type → GPU`.

---

## Célula 1 — Clonar + build do `nsos_ext` (CUDA, arch da GPU do runtime)

```python
import os, subprocess, pathlib
REPO = "/content/reimagined-main"
if not pathlib.Path(REPO).exists():
    # troque pela sua origem (git URL ou cópia do Drive)
    subprocess.run(["git", "clone", "--depth", "1", "<SEU_REPO_GIT>", REPO], check=True)
os.environ["REPO_ROOT"] = REPO

# Bootstrap oficial: detecta a GPU, escolhe arch CUDA, build com ninja, cacheia no Drive.
# (monte o Drive antes se quiser cache: from google.colab import drive; drive.mount('/content/drive'))
%cd $REPO
!python colab/colab_bootstrap.py || true   # se preferir, faça o cmake manual abaixo
```

Build manual alternativo (se não usar o bootstrap):

```python
%cd $REPO
!cmake -S OXN/nsos -B OXN/nsos/build-colab \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNSOS_ENABLE_CUDA=ON -DNSOS_BUILD_PYTHON=ON -DNSOS_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
!cmake --build OXN/nsos/build-colab -j 4
```

---

## Célula 2 — Gates da Fase 1 (parity + determinismo)

```python
import os, sys
BUILD = f"{REPO}/OXN/nsos/build-colab"
# parity (executável C++)
!{BUILD}/test_gpu_parity || {BUILD}/Release/test_gpu_parity
# determinismo
sys.path.insert(0, f"{REPO}/OXN/scripts")
!python {REPO}/OXN/scripts/verify_determinism.py --build-dir {BUILD}
```

Esperado: `GPU parity test passed!` (6/6) e `Todos os testes de determinismo passaram.`

---

## Célula 3 — Pool: validar o ~3.9× EM ESCALA (ON vs OFF)

```python
# batch-sizes auto pela memória (T4: 32/64/128 ; A100: 64/128/256)
!python {REPO}/OXN/nsos/scripts/validate_pool_scale.py \
    --build-dir {BUILD} --profile mamba_small --seq-len 256 --steps 10
```

Lê a tabela `POOL ON vs OFF`: o `speedup` deve **crescer** vs a 1050 Ti conforme
a GPU enche (sem o thrash de UM de 4 GB). `mem ON MB` deve ficar dentro do device.

---

## Célula 4 — precisão mista (Tensor Cores): FP32 vs formato nativo

```python
import os
os.environ["NSOS_GPU_POOL"] = "1"            # pool sempre ON aqui
precision = "bf16" if int(info["arch"]) >= 80 else "fp16"
for prec in ("fp32", precision):
    os.environ["NSOS_MIXED_PRECISION"] = prec  # lido na 1a alocação; processo separado por isso
    print(f"\n===== precision = {prec} =====")
    !NSOS_MIXED_PRECISION={prec} python {REPO}/OXN/nsos/scripts/profile_bottlenecks.py \
        --device gpu --profile mamba_small --batch-sizes 32,64 --seq-len 256 \
        --steps 10 --skip-forward --configs baseline,full
```

> Selecione em runtime com `nsos_ext.set_matmul_precision("fp16")` na T4/V100
> ou `"bf16"` em sm_80+. Pesos-mestre e otimizador seguem FP32. FP16 usa o
> scaler dinâmico do `Trainer`; BF16 dispensa scaling por ter expoente de 8 bits.

---

## Célula 5 — Loop de paridade 1e-4 (padrão ouro p/ kernels novos de Fase 2)

Qualquer kernel novo (ex.: parallel-prefix scan do Mamba) entra **atrás de flag,
default OFF**, com o caminho atual como fallback. Validar assim antes de confiar:

```python
import numpy as np, sys
sys.path.insert(0, BUILD); 
import nsos_ext as nsos   # garanta as DLLs/.so no path (Colab Linux: já resolve)

def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

# Exemplo de harness de paridade p/ um kernel sob flag (pseudo — adapte ao kernel):
#   1. rode o caminho de referência (CPU ou kernel atual) -> ref
#   2. ligue a flag do novo kernel (env ou setter) e rode -> novo
#   3. assert max_abs_diff(ref, novo) < 1e-4
# Gate: só promove o kernel a default quando passar 1e-4 + determinismo (Célula 2).
print("scaffold pronto — conecte ao kernel sob flag quando ele existir")
```

---

## Resumo do estado (para referência)

- **Fase 1:** completa — parity/determinismo/profiler verdes, persistent_kernel
  removido, optimizer na GPU, GIL release, async-I/O (smart_loader/dataloader/
  copy-stream), **BF16 runtime** (`set_matmul_precision`), BitLinear repack-once
  auditado, **GPU pool ~3.9×**.
- **Fase 2:** o scan ativo (`launch_mamba_selective_scan_forward`) já é paralelo
  sobre Batch×D; `forward_moe_gpu_batched` já é GPU-nativo. O **parallel-prefix
  scan** (ganho p/ seq longa/inferência) é o item restante e é **parity-critical**
  — implementar atrás de flag e validar com a Célula 5 antes de ligar por default.
```
