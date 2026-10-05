"""Shared enums."""

from enum import StrEnum


class ContinuityMode(StrEnum):
    HARD_CUT = "HARD_CUT"
    KEYFRAME_BRIDGE = "KEYFRAME_BRIDGE"
    CONTINUOUS = "CONTINUOUS"
    NATIVE_LONG = "NATIVE_LONG"


class ProjectStatus(StrEnum):
    DRAFT = "DRAFT"
    SCRIPT_REVIEW = "SCRIPT_REVIEW"
    CHARACTERS_REVIEW = "CHARACTERS_REVIEW"
    STORYBOARD_REVIEW = "STORYBOARD_REVIEW"
    RENDERING = "RENDERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ShotStatus(StrEnum):
    PENDING = "PENDING"
    GENERATING = "GENERATING"
    QC = "QC"
    APPROVED = "APPROVED"
    FAILED = "FAILED"


class JobState(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


class AssetKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"
    OTHER = "other"
