"""Domain models shared by API, LLM structured output, DB DTOs and workflows."""

from datetime import datetime

from pydantic import BaseModel, Field

from vidgen_core.enums import ContinuityMode, ProjectStatus, ShotStatus

__all__ = [
    "CharacterSpec",
    "ContinuityMode",
    "CostEstimate",
    "Project",
    "QCResult",
    "Scene",
    "Script",
    "Shot",
    "Take",
    "VideoJobStatus",
    "VideoRequest",
    "VideoSpec",
]


class VideoSpec(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: int = 24


class Shot(BaseModel):
    id: str
    scene_id: str = Field(max_length=64)
    index: int
    duration_s: float = Field(ge=3, le=30)
    shot_size: str = Field(max_length=64)
    angle: str = Field(max_length=64)
    camera_move: str = Field(max_length=64)
    action: str = Field(max_length=400)
    continuity_mode: ContinuityMode = ContinuityMode.HARD_CUT
    prev_shot_id: str | None = None


class VideoRequest(BaseModel):
    prompt: str
    duration_s: float
    spec: VideoSpec
    first_frame_uri: str | None = None
    last_frame_uri: str | None = None
    reference_uris: list[str] = []
    seed: int | None = None


class VideoJobStatus(BaseModel):
    provider_job_id: str
    state: str  # "running" | "succeeded" | "failed" | "blocked"
    result_url: str | None = None
    error: str | None = None


class CharacterSpec(BaseModel):
    id: str
    name: str
    description: str
    appearance: str = ""
    voice: str | None = None
    reference_asset_ids: list[str] = []


class Scene(BaseModel):
    id: str
    index: int
    location: str
    summary: str
    character_ids: list[str] = []


class Script(BaseModel):
    title: str
    logline: str = ""
    scenes: list[Scene]
    characters: list[CharacterSpec] = []


class Project(BaseModel):
    id: str
    title: str
    status: ProjectStatus = ProjectStatus.DRAFT
    spec: VideoSpec = VideoSpec()
    budget_usd: float | None = Field(default=None, ge=0)
    created_at: datetime | None = None


class QCResult(BaseModel):
    passed: bool
    scores: dict[str, float] = {}
    reasons: list[str] = []


class Take(BaseModel):
    id: str
    shot_id: str
    number: int = Field(ge=1)
    asset_id: str | None = None
    provider: str | None = None
    provider_job_id: str | None = None
    status: ShotStatus = ShotStatus.PENDING
    qc: QCResult | None = None
    cost_usd: float = 0.0
    selected: bool = False


class CostEstimate(BaseModel):
    provider: str
    units: float = Field(ge=0)
    unit: str = "second"
    usd: float = Field(ge=0)
