FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04
# build context = this folder
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON=3.12 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    HF_HOME=/models/hf HF_HUB_OFFLINE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORKDIR /app
COPY vendor ./vendor
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-dev --extra gpu --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev --extra gpu
CMD ["uv","run","uvicorn","--factory","videogen_server.service:build_app","--host","0.0.0.0","--port","8800","--workers","1"]
