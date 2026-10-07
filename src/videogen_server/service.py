"""FastAPI inference service: one GPU, one worker, sequential queue."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from vidgen_core.models import VideoJobStatus, VideoRequest
from vidgen_core.selfhost import SelfHostConfig, load_selfhost_config

from videogen_server import segments as seg
from videogen_server.segments import GenerationOOM, InvalidInput, SegmentJob

class JobStore:
    """SQLite job state (lives on the data volume) so a restart doesn't lose it."""

    def __init__(self, path: str | Path) -> None:
        self.db = sqlite3.connect(path, check_same_thread=False)  # only the event loop uses it
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, status TEXT NOT NULL,"
            " created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )  # fmt: skip

    async def put(self, st: VideoJobStatus) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO jobs(id, status) VALUES(?, ?)"
                " ON CONFLICT(id) DO UPDATE SET status=excluded.status",
                (st.provider_job_id, st.model_dump_json()),
            )

    async def get(self, job_id: str) -> VideoJobStatus | None:
        row = self.db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        return VideoJobStatus.model_validate_json(row[0]) if row else None

    async def delete(self, job_id: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    async def fail_running(self) -> None:
        for (raw,) in self.db.execute("SELECT status FROM jobs").fetchall():
            st = VideoJobStatus.model_validate_json(raw)
            if st.state == "running":
                await self.put(
                    st.model_copy(update={"state": "failed", "error": "server restarted"})
                )


def validate_against_caps(req: VideoRequest, cfg: SelfHostConfig) -> None:
    c = cfg.constraints
    if req.duration_s > c.max_duration_s:
        raise ValueError(f"duration_s {req.duration_s} > max {c.max_duration_s}")
    if len(req.reference_uris) > cfg.capabilities.max_reference_images:
        raise ValueError("reference_uris not supported")
    if req.last_frame_uri and not req.first_frame_uri:
        raise ValueError("last_frame_uri requires first_frame_uri")


def sign(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def send_webhook(cfg: SelfHostConfig, st: VideoJobStatus) -> None:
    if not cfg.service.callback_url:
        return
    body = st.model_dump_json().encode()
    secret = os.environ.get(cfg.service.callback_hmac_secret_env or "", "")
    headers = {"Content-Type": "application/json", "X-Signature": sign(secret, body)}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(cfg.service.callback_url, content=body, headers=headers)
    except httpx.HTTPError:
        pass  # workflow still polls every 30s


def fetch_image(uri: str) -> Image.Image:
    if not uri.startswith(("http://", "https://")):
        raise InvalidInput(f"unsupported image uri {uri!r} (need presigned http(s) URL)")
    try:
        r = httpx.get(uri, timeout=30, follow_redirects=True)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except (httpx.HTTPError, OSError) as e:
        raise InvalidInput(f"cannot load {uri}: {e}") from e


def upload_s3(cfg: SelfHostConfig, path: str, job_id: str) -> str:
    import boto3

    key = f"{cfg.storage.s3_prefix}{job_id}.mp4"
    boto3.client("s3").upload_file(path, cfg.storage.s3_bucket, key)
    return f"s3://{cfg.storage.s3_bucket}/{key}"


def save_local(cfg: SelfHostConfig, path: str, job_id: str) -> str:
    out = cfg.storage.output_dir
    out.mkdir(parents=True, exist_ok=True)
    shutil.move(path, out / f"{job_id}.mp4")
    return f"{cfg.service.base_url}/v1/jobs/{job_id}/video"


def run_request(engine: Any, cfg: SelfHostConfig, job_id: str, req: VideoRequest) -> str:
    """Split -> generate each segment (chaining last frame) -> concat -> S3. Blocking."""
    c, g = cfg.constraints, cfg.generation
    w, h = seg.round_size(g.render_width, g.render_height, c.size_multiple)
    first = fetch_image(req.first_frame_uri) if req.first_frame_uri else None
    last = fetch_image(req.last_frame_uri) if req.last_frame_uri else None
    seed = seg.pick_seed(req.seed)
    durations = seg.split_duration(req.duration_s, c.max_native_duration_s)
    cfg.storage.scratch_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=job_id, dir=cfg.storage.scratch_dir))
    try:
        parts: list[str] = []
        for k, d in enumerate(durations):
            out = str(work / f"seg{k}.mp4")
            engine.generate(SegmentJob(
                prompt=req.prompt,
                num_frames=seg.frames_for(d, g.fps, c.frame_multiple, c.frame_offset),
                width=w, height=h,
                first=first,
                last=last if k == len(durations) - 1 else None,
                seed=seed + k, out_path=out,
            ))  # fmt: skip
            parts.append(out)
            if k < len(durations) - 1:  # CONTINUOUS: next segment starts at this one's end
                first = Image.open(seg.last_frame(out, str(work / f"last{k}.png"))).convert("RGB")
        final = str(work / "final.mp4")
        seg.concat(parts, final, c.segment_overlap_frames, g.keep_audio)
        save = save_local if cfg.storage.output_dir else upload_s3
        return save(cfg, final, job_id)
    finally:
        shutil.rmtree(work, ignore_errors=True)  # scratch quota is 20GB


def make_app(cfg: SelfHostConfig, store: JobStore, engine: Any) -> FastAPI:
    queue: asyncio.Queue[tuple[str, VideoRequest]] = asyncio.Queue(cfg.runtime.max_queue_depth)

    async def worker() -> None:
        while True:
            job_id, req = await queue.get()
            try:
                uri = await asyncio.to_thread(run_request, engine, cfg, job_id, req)
                st = VideoJobStatus(provider_job_id=job_id, state="succeeded", result_url=uri)
            except GenerationOOM as e:
                st = VideoJobStatus(provider_job_id=job_id, state="failed", error=f"OOM: {e}")
            except InvalidInput as e:
                st = VideoJobStatus(
                    provider_job_id=job_id, state="failed", error=f"invalid_input: {e}"
                )
            except Exception as e:  # noqa: BLE001
                st = VideoJobStatus(
                    provider_job_id=job_id, state="failed", error=f"internal: {e!r}"
                )
            await store.put(st)
            await send_webhook(cfg, st)
            queue.task_done()

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        await store.fail_running()
        task = asyncio.create_task(worker())
        yield
        task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/jobs", status_code=202)
    async def submit(req: VideoRequest) -> dict[str, str]:
        try:
            validate_against_caps(req, cfg)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        if queue.full():
            raise HTTPException(503, "queue full", headers={"Retry-After": "60"})
        job_id = uuid.uuid4().hex
        await store.put(VideoJobStatus(provider_job_id=job_id, state="running"))
        queue.put_nowait((job_id, req))
        return {"job_id": job_id}

    @app.get("/v1/jobs/{job_id}")
    async def status(job_id: str) -> VideoJobStatus:
        st = await store.get(job_id)
        if st is None:
            raise HTTPException(404, "unknown job")
        return st

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    async def delete(job_id: str) -> None:
        st = await store.get(job_id)
        if st is None:
            raise HTTPException(404, "unknown job")
        if st.state == "running":
            raise HTTPException(409, "job still running")
        if cfg.storage.output_dir and job_id.isalnum():
            (cfg.storage.output_dir / f"{job_id}.mp4").unlink(missing_ok=True)
        await store.delete(job_id)

    @app.get("/v1/jobs/{job_id}/video")
    async def video(job_id: str) -> FileResponse:
        out = cfg.storage.output_dir
        f = out / f"{job_id}.mp4" if out and job_id.isalnum() else None
        if f is None or not f.is_file():
            raise HTTPException(404, "no video")
        return FileResponse(f, media_type="video/mp4", filename=f.name)

    @app.get("/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {"name": cfg.provider_name, **cfg.capabilities.model_dump(),
                "max_duration_s": cfg.constraints.max_duration_s}  # fmt: skip

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        free_gb = None
        try:
            import torch

            free_gb = torch.cuda.mem_get_info()[0] / 1e9
        except Exception:  # noqa: BLE001  # torch absent (tests) or no CUDA
            pass
        return {"queue_depth": queue.qsize(), "max_queue_depth": cfg.runtime.max_queue_depth,
                "free_vram_gb": free_gb, "model_id": cfg.model.model_id,
                "revision": cfg.model.revision}  # fmt: skip

    return app


def build_app() -> FastAPI:
    """uvicorn entry: `uvicorn --factory videogen_server.service:build_app --workers 1`."""
    cfg = load_selfhost_config()
    # Defaults so a bare `uvicorn --factory ...` works; the environment still wins. Must precede torch/hf imports.
    os.environ.setdefault("HF_HOME", str(cfg.storage.hf_home.resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights come from `videogen-download`
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    from videogen_server.download import ensure_model
    from videogen_server.engine_ltx2 import LTX2Engine

    if os.environ["HF_HUB_OFFLINE"] != "1":
        ensure_model(cfg)
    free = torch.cuda.mem_get_info()[0] / 1024**3
    if free < cfg.runtime.min_free_vram_gb:
        raise RuntimeError(
            f"free VRAM {free:.1f}GB < min_free_vram_gb {cfg.runtime.min_free_vram_gb}"
        )
    engine = LTX2Engine(cfg)
    engine.load()
    # ponytail: no warm-up clip; first job pays the compile/offload warm cost. Add if p99 matters.
    data = cfg.storage.output_dir or cfg.storage.scratch_dir
    data.mkdir(parents=True, exist_ok=True)
    store = JobStore(data / "jobs.db")
    return make_app(cfg, store, engine)
