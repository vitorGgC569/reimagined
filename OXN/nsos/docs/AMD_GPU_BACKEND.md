# NSOS on AMD and NVIDIA GPUs

Historical runtime status captured on 2026-07-28:

- AMD HIP/ROCm: built and tested on Radeon RX 7600 8 GiB (`gfx1102`).
- NVIDIA CUDA: retained as a first-class CMake backend and uses the same
  kernel sources, but could not be executed on this workstation because it has
  no NVIDIA GPU or CUDA toolkit.
- CPU: remains available with `NSOS_GPU_BACKEND=NONE`.

The `.cu` files are intentionally shared. NVCC compiles them for CUDA and
AMD clang compiles them as HIP translation units. Do not duplicate the kernel
tree into a second AMD-only implementation.

## AMD SDK on Windows

The tested SDK is AMD ROCm/TheRock 7.14.0 for `gfx110X`, extracted to
`C:\TheRock\build`. This is the package and layout documented by
[AMD's ROCm 7.14 installer documentation](https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html).

```powershell
New-Item -ItemType Directory -Force C:\TheRock\build | Out-Null
Set-Location C:\TheRock
curl.exe -L `
  -o therock-dist-windows-gfx110X-all-7.14.0.tar.gz `
  https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-windows-gfx110X-all-7.14.0.tar.gz
tar.exe -xzf therock-dist-windows-gfx110X-all-7.14.0.tar.gz `
  -C C:\TheRock\build --strip-components=1
```

Use an **x64 Native Tools Command Prompt for VS 2022** or call
`VsDevCmd.bat` before CMake. The AMD compiler uses the MSVC/Windows SDK headers
and libraries for host code.

```powershell
cmd.exe /k """C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"" -arch=x64 -host_arch=x64"
```

Inside that developer shell:

```bat
set ROCM_PATH=C:\TheRock\build
set HIP_PATH=C:\TheRock\build
set PATH=C:\TheRock\build\bin;C:\TheRock\build\lib\llvm\bin;%PATH%

cmake -S . -B build-codex-hip -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER=C:\TheRock\build\lib\llvm\bin\clang++.exe ^
  -DNSOS_GPU_BACKEND=HIP ^
  -DNSOS_HIP_ROOT=C:\TheRock\build ^
  -DNSOS_HIP_ARCHITECTURES=gfx1102 ^
  -DNSOS_BUILD_PYTHON=ON ^
  -DNSOS_BUILD_TESTS=ON

cmake --build build-codex-hip -j 4
ctest --test-dir build-codex-hip -j 1 --output-on-failure
```

`NSOS_HIP_DEVICE_LIB_PATH` can be set explicitly if device-library discovery
fails:

```bat
-DNSOS_HIP_DEVICE_LIB_PATH=C:\TheRock\build\lib\llvm\amdgcn\bitcode
```

Verify the installed runtime before building:

```bat
C:\TheRock\build\bin\hipconfig.exe
C:\TheRock\build\bin\hipInfo.exe
```

## NVIDIA CUDA build

The CUDA lane remains independent of HIP:

```powershell
cmake -S . -B build-cuda -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DNSOS_GPU_BACKEND=CUDA `
  -DNSOS_CUDA_TOOLKIT_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" `
  -DNSOS_CUDA_ARCHITECTURES="75;86" `
  -DNSOS_BUILD_PYTHON=ON `
  -DNSOS_BUILD_TESTS=ON

cmake --build build-cuda -j 4
ctest --test-dir build-cuda -j 1 --output-on-failure
```

Use the architecture that matches the target NVIDIA board. `AUTO` prefers a
hinted HIP SDK; use `NSOS_GPU_BACKEND=CUDA` or `HIP` in production builds so
the selected backend is explicit.

## Runtime and device selection

The Python loader searches the build directory and both SDK families. For a
HIP deployment:

```powershell
$env:NSOS_HIP_ROOT = "C:\TheRock\build"
$env:PYTHONPATH = ".\build-codex-hip"
python -c "from scripts.train_curriculum import load_nsos; from pathlib import Path; n=load_nsos(Path('build-codex-hip')); print(n.gpu_backend_name(), n.gpu_devices())"
```

The runtime selects the discrete GPU by default. It rejects an integrated GPU
unless explicitly allowed:

- `NSOS_GPU_DEVICE=<index>` selects a device.
- `NSOS_ALLOW_INTEGRATED_GPU=1` permits an iGPU.
- `nsos_ext.gpu_devices()` reports name, architecture, VRAM, capabilities and
  whether that architecture was compiled into the loaded binary (`compiled`).
- `nsos_ext.selected_gpu_device()` reports the process-policy device index and
  rebinds the calling host thread to it.
- `nsos_ext.gpu_backend_name()` returns `hip`, `cuda` or `none`.

Device-selection environment variables are read once on the first GPU runtime
use and become immutable for that process. Set them before importing
`nsos_ext`; changing them after an allocation is intentionally unsupported.
Each host thread is rebound and verified against the selected device before an
NSOS GPU boundary, so an external `hipSetDevice`/`cudaSetDevice` call cannot
silently redirect later NSOS kernels.

`nsos_ext.pool_stats()` also exposes release-integrity counters. Production
training and benchmark evidence require zero retained releases, release
failures, unknown deallocations and capture-contract violations; a failed
driver release is retained and never recycled as if it were valid storage.

The tested machine exposes RX 7600 as device 0 (`gfx1102`) and Ryzen integrated
graphics as device 1 (`gfx1103`); the default is therefore device 0.

## RX 7600 training profile

Use the backend-portable 200M architecture with the memory controls tuned for
8 GiB:

```powershell
$env:NSOS_HIP_ROOT = "C:\TheRock\build"
python scripts\train_curriculum.py `
  --build-dir build-codex-hip `
  --profile hybrid_rx7600_200m_chinchilla25b `
  --device gpu `
  --checkpoint-every-steps 100 `
  --rebuild-curriculum
```

The profile starts at batch 1 with gradient checkpointing. Its resident model
uses MoE-4 top-2 with a 1024-wide expert hidden layer; the earlier MoE-8/4096
configuration was actually 562M resident parameters and could not fit
weights, gradients and Adam states in 8 GiB. Do not increase the batch until a
representative phase has been measured for peak VRAM. The corrected model
architecture matches the NVIDIA 200M profile, so model and trainer checkpoint
formats remain backend-neutral.

An exact resume needs all three sidecars:

```powershell
python scripts\train_curriculum.py `
  --build-dir build-codex-hip `
  --profile hybrid_rx7600_200m_chinchilla25b `
  --device gpu `
  --resume-model artifacts\curriculum_runs\latest\phase3_best.bin
```

The script infers `phase3_best.state` and `phase3_best.progress.json`. It fails
closed when either is absent. `--weights-only-resume` is an explicit restart of
optimizer, scheduler and curriculum position, not an exact resume.

Native exact recovery follows the same ordering: load the v4 model checkpoint,
then load the SHA-256-bound Trainer sidecar. If a GPU optimizer launch or its
completion boundary becomes ambiguous, the Trainer is fail-stop poisoned and
cannot train or export again until that exact pair is restored. The GPU
pre-commit gate rejects NaN/Inf in weights, gradients and moments and also
rejects negative Adam second moments. The post-kernel boundary transfers only
the fixed four-byte status word; it is the required completion/error boundary,
not a full weight round-trip.

## Runtime Library / DLL errors

If `import nsos_ext` reports `DLL load failed`:

1. Confirm that `NSOS_HIP_ROOT`, `ROCM_PATH` or `HIP_PATH` points at the
   extracted SDK.
2. Confirm `C:\TheRock\build\bin` and
   `C:\TheRock\build\lib\llvm\bin` exist.
3. Use `scripts/train_curriculum.py`, whose loader keeps Windows DLL-directory
   handles alive for the whole process.
4. Rebuild in the VS x64 developer environment. A cached `cl.exe` path without
   the developer environment cannot find the MSVC standard library.
5. Do not combine binaries from different build directories or Python ABIs.

The smoke gate is:

```powershell
ctest --test-dir build-codex-hip -R "^test_python_binding_smoke$" --output-on-failure
```

## Validation evidence

On the RX 7600:

- HIP Release build, including `nsos_ext`, completed.
- The then-current HIP lane passed 66/66 CTest entries and the CPU Debug lane
  passed 39/39. These are immutable historical results, not the size of the
  current suite.
- `test_gpu_backend_configuration` verifies without vendor hardware that the
  complete kernel manifest, runtime headers, libraries, compile definitions and
  fail-closed test gate remain wired for both HIP and CUDA.
- GPU parity covers tensor operations, MSE loss, BitLinear/DP4A, Mamba
  variants, Jamba, MoE, KAN, sparse attention and TTT.
- checkpoint continuation and mixed-precision contracts passed.
- `train_e2e` completed GPU training, evaluation, generation, MCTS reasoning
  and SelfHealer.
- the corrected production profile instantiated 209,035,480 resident
  parameters (0.779 GiB of FP32 weights) and completed a synchronized
  batch-1, sequence-160 optimizer step on the RX 7600.

This proves the tested NSOS paths on this AMD machine. It is not evidence for
every Radeon architecture, driver version or NVIDIA board; those combinations
still require the same test gate on their target hardware.

For every new build, use `<build-dir>/nsos_test_inventory.txt` as the
authoritative configured test list and archive it beside the CTest/JUnit and
benchmark evidence. Never update this document with a new manual test count.
