FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04
# build context = server_selfhost/..  (needs ../video_pipeline/packages/vidgen-core)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON=3.12 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    HF_HOME=/models/hf HF_HUB_OFFLINE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COPY video_pipeline/packages/vidgen-core /video_pipeline/packages/vidgen-core
WORKDIR /server_selfhost/videogen-server
COPY server_selfhost/videogen-server/uv.lock server_selfhost/videogen-server/pyproject.toml ./
RUN uv sync --frozen --no-dev --extra gpu --no-install-project
COPY server_selfhost/videogen-server/src ./src
COPY server_selfhost/videogen-server/configs ./configs
RUN uv sync --frozen --no-dev --extra gpu
CMD ["uv","run","uvicorn","--factory","videogen_server.service:build_app","--host","0.0.0.0","--port","8800","--workers","1"]
