# Stage 1: Builder (Heavy Toolchain)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Build Dependencies
# Clean apt cache in the same layer to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    libopenmpi-dev \
    ninja-build \
    cppcheck \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Build-Time Python Deps
# pybind11 is required for compiling the C++ extensions
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir pybind11

# 3. Copy Source Code
# Copy only what's needed for compilation to optimize cache invalidation
WORKDIR /app
COPY OXN /app/OXN
COPY KernelOpen /app/KernelOpen
COPY scripts /app/scripts
COPY src /app/src
COPY include /app/include

# 4. Build C++ Extensions
# Generates /app/build/nsos_ext*.so
ARG CUDA_ARCH=sm_86
ENV CUDA_ARCH=$CUDA_ARCH
RUN chmod +x scripts/build_extensions_gpu.sh
RUN ./scripts/build_extensions_gpu.sh

# Verify build artifact
RUN ls -lh /app/build/nsos_ext*.so

# Stage 2: Runtime (Lightweight)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/usr/local/lib/oxn"

# 1. Install Runtime System Dependencies
# Unhold libraries to allow updates if needed, fixing dependency hell
RUN apt-get update && \
    apt-mark unhold libcublas-12-1 libnccl2 || true && \
    apt-get install -y --allow-change-held-packages \
    python3 \
    python3-pip \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Runtime Python Libraries
# Use --no-cache-dir to keep image small
# Group 1: Core ML (PyTorch - NVIDIA Index)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Group 1.5: General Utilities (Standard PyPI Index)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir \
    numpy \
    psutil \
    matplotlib \
    tqdm \
    pybind11 \
    datasets \
    pandas \
    ruff

# Group 2: Llama-CPP (Pre-built wheels preferred)
RUN --mount=type=cache,target=/root/.cache/pip \
    CMAKE_ARGS="-DGGML_CUDA=on" \
    FORCE_CMAKE=1 \
    python3 -m pip install --no-cache-dir llama-cpp-python --prefer-binary \
    --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu121

WORKDIR /app

# 3. Copy Compiled Artifacts from Builder
# We copy to a system path (/usr/local/lib/oxn) to avoid shadowing if /app is mounted as a volume
RUN mkdir -p /usr/local/lib/oxn
COPY --from=builder /app/build/nsos_ext*.so /usr/local/lib/oxn/

# 4. Copy Application Scripts
# Copy remainder of the repo (benchmarks, tools, docs)
COPY benchmarks /app/benchmarks
COPY bindings /app/bindings
COPY tests /app/tests
COPY tools /app/tools
COPY *.py /app/
COPY *.md /app/

# 5. Entrypoint
COPY OXN/scripts/entrypoint_industrial.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# CONFIGURAÇÃO DE ELITE:
# O Entrypoint roda o script de verificação.
# O CMD é o comando padrão se o usuário não digitar nada.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3", "OXN/scripts/train_industrial.py"]
