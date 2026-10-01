FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    python3-dev \
    python3-pip \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir pybind11==2.13.6

WORKDIR /src
COPY OXN/nsos /src/OXN/nsos

RUN cmake -S /src/OXN/nsos -B /src/OXN/nsos/build-mvp \
    -DCMAKE_BUILD_TYPE=Release \
    -DNSOS_ENABLE_CUDA=OFF \
    -DNSOS_BUILD_PYTHON=ON \
    -DNSOS_BUILD_TESTS=OFF \
    -DNSOS_BUILD_CIRCUIT_TOOLS=OFF \
    -DNSOS_BUILD_CLI=ON \
    -DNSOS_BUILD_API=ON \
    -DNSOS_BUILD_OXTAMEM=OFF \
    -DNSOS_ENABLE_NATIVE_OPTIMIZATIONS=OFF \
    -G Ninja \
    && cmake --build /src/OXN/nsos/build-mvp -j"$(nproc)"

FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/opt/nsos/python

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /home/nsos --shell /usr/sbin/nologin nsos

WORKDIR /opt/nsos
RUN mkdir -p /opt/nsos/bin /opt/nsos/python /opt/nsos/artifacts/model_packs \
    && chown -R nsos:nsos /opt/nsos

COPY --from=builder /src/OXN/nsos/build-mvp/nsos_api_server /opt/nsos/bin/
COPY --from=builder /src/OXN/nsos/build-mvp/nsos_cli /opt/nsos/bin/
COPY --from=builder /src/OXN/nsos/build-mvp/nsos_ext*.so /opt/nsos/python/
COPY OXN/nsos/README.md /opt/nsos/README.md
COPY OXN/nsos/scripts/container_runtime.py /opt/nsos/bin/container_runtime.py

USER nsos
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 /opt/nsos/bin/container_runtime.py health

CMD ["python3", "/opt/nsos/bin/container_runtime.py", "serve"]
