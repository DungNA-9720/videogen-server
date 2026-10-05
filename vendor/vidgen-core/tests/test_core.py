import json

import pytest
from pydantic import ValidationError
from vidgen_core.enums import ContinuityMode
from vidgen_core.errors import BudgetExceeded
from vidgen_core.logging import bind_context, clear_context, configure_logging, get_logger
from vidgen_core.models import Shot, VideoJobStatus, VideoRequest, VideoSpec
from vidgen_core.settings import Settings


def _shot(**kw):
    base = dict(
        id="s1",
        scene_id="sc1",
        index=0,
        duration_s=5,
        shot_size="MS",
        angle="eye",
        camera_move="static",
        action="walks",
    )
    return Shot(**(base | kw))


def test_shot_defaults():
    s = _shot()
    assert s.continuity_mode == ContinuityMode.HARD_CUT
    assert s.prev_shot_id is None


@pytest.mark.parametrize("kw", [{"duration_s": 2}, {"duration_s": 31}, {"action": "x" * 401}])
def test_shot_validation(kw):
    with pytest.raises(ValidationError):
        _shot(**kw)


def test_video_request_roundtrip():
    req = VideoRequest(prompt="p", duration_s=5, spec=VideoSpec())
    assert VideoRequest.model_validate_json(req.model_dump_json()) == req
    assert req.spec.fps == 24 and req.reference_uris == []


def test_job_status():
    assert VideoJobStatus(provider_job_id="j", state="running").result_url is None


def test_settings_env(monkeypatch):
    monkeypatch.setenv("VIDGEN_S3_BUCKET", "b")
    monkeypatch.setenv("VIDGEN_FAL_KEY", "abc123")
    s = Settings(_env_file=None)
    assert s.s3_bucket == "b"
    assert "abc123" not in repr(s)


def test_budget_exceeded_message():
    assert "10.00" in str(BudgetExceeded(10, 2))


def test_logging_binds_context(capsys):
    configure_logging("INFO", json_logs=True)
    bind_context(project_id="p1", job_id="j1")
    get_logger().info("hello")
    clear_context()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["project_id"] == "p1" and out["job_id"] == "j1" and out["event"] == "hello"
