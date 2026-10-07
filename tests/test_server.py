import json
import subprocess
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from vidgen_core.models import VideoRequest, VideoSpec
from vidgen_core.selfhost import SelfHostConfig, load_selfhost_config

from videogen_server import segments as seg
from videogen_server.service import JobStore, make_app, sign, validate_against_caps

CFG_PATH = Path(__file__).parents[1] / "configs/ltx25.json"


def test_frames_and_sizes() -> None:
    assert seg.frames_for(5, 24) == 121
    assert seg.frames_for(10, 24) == 241
    assert seg.round_size(1920, 1080) == (1920, 1056)


@pytest.mark.parametrize(("d", "n"), [(3, 1), (10, 1), (10.5, 2), (30, 3)])
def test_split(d: float, n: int) -> None:
    parts = seg.split_duration(d, 10)
    assert len(parts) == n and max(parts) <= 10 and sum(parts) == pytest.approx(d)


def test_concat_cmd_drops_overlap_and_crossfades() -> None:
    cmd = " ".join(seg.concat_cmd(["a.mp4", "b.mp4"], "o.mp4", 1, True))
    assert "[1:v]trim=start_frame=1" in cmd and "[0:v]trim" not in cmd
    assert "acrossfade=d=0.05" in cmd
    assert "acrossfade" not in " ".join(seg.concat_cmd(["a.mp4", "b.mp4"], "o.mp4", 1, False))


def test_config_file_valid_and_bad_cases() -> None:
    cfg = load_selfhost_config(CFG_PATH)
    assert cfg.model.model_id == "Lightricks/LTX-2.5-Diffusers"
    raw = json.loads(CFG_PATH.read_text())
    for patch in ({"pricing": {}}, {"model": {**raw["model"], "model_id": "nope"}}):
        with pytest.raises(ValueError):
            SelfHostConfig.model_validate({**raw, **patch})


def req(**kw: object) -> VideoRequest:
    return VideoRequest(prompt="p", duration_s=5, spec=VideoSpec(), **kw)  # type: ignore[arg-type]


def test_validate_against_caps() -> None:
    cfg = load_selfhost_config(CFG_PATH)
    validate_against_caps(req(), cfg)
    for bad in (
        VideoRequest(prompt="p", duration_s=31, spec=VideoSpec()),
        req(reference_uris=["x"]),
        req(last_frame_uri="http://x"),
    ):
        with pytest.raises(ValueError):
            validate_against_caps(bad, cfg)


async def test_api_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from videogen_server import service

    cfg = load_selfhost_config(CFG_PATH).model_copy(deep=True)
    cfg.service.callback_url = None
    monkeypatch.setattr(service, "run_request", lambda *a: "s3://b/x.mp4")
    app = make_app(cfg, JobStore(":memory:"), engine=None)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/v1/jobs", json=req().model_dump())
            assert r.status_code == 202
            jid = r.json()["job_id"]
            for _ in range(50):
                st = (await c.get(f"/v1/jobs/{jid}")).json()
                if st["state"] != "running":
                    break
                await __import__("asyncio").sleep(0.02)
            assert st["state"] == "succeeded" and st["result_url"] == "s3://b/x.mp4"
            assert (
                await c.post("/v1/jobs", json=req(reference_uris=["x"]).model_dump())
            ).status_code == 422
            assert (await c.get("/v1/jobs/nope")).status_code == 404
            assert (await c.delete(f"/v1/jobs/{jid}")).status_code == 204
            assert (await c.get(f"/v1/jobs/{jid}")).status_code == 404
            assert (await c.delete(f"/v1/jobs/{jid}")).status_code == 404
            assert (await c.get("/healthz")).json()["model_id"] == cfg.model.model_id


async def test_delete_removes_file_and_running_is_409(tmp_path: Path) -> None:
    from vidgen_core.models import VideoJobStatus

    cfg = load_selfhost_config(CFG_PATH).model_copy(deep=True)
    cfg.storage.output_dir = tmp_path
    store = JobStore(":memory:")
    await store.put(VideoJobStatus(provider_job_id="done", state="succeeded"))
    await store.put(VideoJobStatus(provider_job_id="busy", state="running"))
    (tmp_path / "done.mp4").write_bytes(b"x")
    app = make_app(cfg, store, engine=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/v1/jobs/done/video")).status_code == 200
        assert (await c.delete("/v1/jobs/busy")).status_code == 409
        assert (await c.delete("/v1/jobs/done")).status_code == 204
        assert not (tmp_path / "done.mp4").exists()
        assert (await c.get("/v1/jobs/done/video")).status_code == 404


async def test_restart_fails_running_jobs() -> None:
    from vidgen_core.models import VideoJobStatus

    store = JobStore(":memory:")
    await store.put(VideoJobStatus(provider_job_id="j", state="running"))
    await store.fail_running()
    st = await store.get("j")
    assert st and st.state == "failed" and st.error == "server restarted"


def test_sign_is_hmac_sha256_hex() -> None:
    assert len(sign("s", b"body")) == 64


@pytest.mark.skipif(
    not subprocess.run(["which", "ffmpeg"], capture_output=True).stdout, reason="no ffmpeg"
)
def test_concat_real_ffmpeg(tmp_path: Path) -> None:
    parts = []
    for i in range(2):
        p = str(tmp_path / f"{i}.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=d=1:s=64x64:r=24",
                "-f",
                "lavfi",
                "-i",
                "sine=d=1",
                "-shortest",
                p,
            ],
            check=True,
        )
        parts.append(p)
    out = seg.concat(parts, str(tmp_path / "o.mp4"), 1, True)
    n = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            out,
        ],
        capture_output=True,
        text=True,
    ).stdout
    assert int(n) == 24 + 23
