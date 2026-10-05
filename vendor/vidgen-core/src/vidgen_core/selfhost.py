"""Self-hosted video server config (shared by the GPU server and the provider adapter). No torch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelSection(BaseModel):
    model_id: str = Field(pattern=r"^[\w.\-]+/[\w.\-]+$")  # HF repo id
    revision: str = "main"
    pipeline_class: str = "LTX2ConditionPipeline"
    torch_dtype: Literal["bfloat16", "float16"] = "bfloat16"  # Ampere: không có fp8 compute
    variant: str | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    gated: bool = False


class RuntimeSection(BaseModel):
    device: str = "cuda:0"
    offload_mode: Literal["none", "model", "sequential", "group"] = "model"
    layerwise_fp8_storage: bool = True
    vae_tiling: bool = True
    attention_backend: Literal["native", "sage", "flash", "xformers"] = "native"
    torch_compile: bool = False
    min_free_vram_gb: float = 40.0
    max_queue_depth: int = Field(8, ge=1)


class GenerationSection(BaseModel):
    render_width: int = 1280
    render_height: int = 704
    fps: int = 24
    sigmas_preset: Literal["distilled", "none"] = "distilled"
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    extra_call_kwargs: dict[str, Any] = {}
    keep_audio: bool = True
    condition_strength: float = Field(1.0, ge=0, le=1)


class ConstraintsSection(BaseModel):
    frame_multiple: int = 8
    frame_offset: int = 1
    size_multiple: int = 32
    max_native_duration_s: float = 10.0
    max_duration_s: float = 30.0
    segment_overlap_frames: int = 1

    @model_validator(mode="after")
    def _check(self) -> ConstraintsSection:
        if self.max_native_duration_s > self.max_duration_s:
            raise ValueError("max_native_duration_s > max_duration_s")
        return self


class CapabilitiesSection(BaseModel):
    supports_first_last_frame: bool
    max_reference_images: int = 0
    resolutions: tuple[str, ...]


class StorageSection(BaseModel):
    hf_home: Path
    min_free_gb: float = 40
    scratch_dir: Path
    s3_bucket: str
    s3_prefix: str = ""


class ServiceSection(BaseModel):
    base_url: str
    callback_url: str | None = None
    callback_hmac_secret_env: str | None = None


class SelfHostConfig(BaseModel):
    schema_version: Literal[1]
    provider_name: str
    backend: Literal["diffusers", "comfyui"] = "diffusers"
    model: ModelSection
    runtime: RuntimeSection
    generation: GenerationSection
    constraints: ConstraintsSection
    capabilities: CapabilitiesSection
    pricing: dict[str, float]
    storage: StorageSection
    service: ServiceSection

    @field_validator("pricing")
    @classmethod
    def _price(cls, v: dict[str, float]) -> dict[str, float]:
        if "usd_per_second" not in v or v["usd_per_second"] < 0:
            raise ValueError("pricing.usd_per_second bắt buộc và >= 0")
        return v


def load_selfhost_config(path: str | Path | None = None) -> SelfHostConfig:
    p = Path(path or os.environ["VIDGEN_SELFHOST_CONFIG"])
    return SelfHostConfig.model_validate(json.loads(p.read_text(encoding="utf-8")))
