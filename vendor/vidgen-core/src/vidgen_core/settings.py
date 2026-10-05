"""Environment-driven configuration (prefix VIDGEN_). No secrets in code."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIDGEN_", env_file=".env", extra="ignore", frozen=True
    )

    env: str = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    database_url: str = "postgresql+asyncpg://vidgen:vidgen@localhost:5432/vidgen"
    redis_url: str = "redis://localhost:6379/0"
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"

    s3_endpoint_url: str | None = None
    # Browser/provider-reachable endpoint used only for presigned URLs (e.g. http://localhost:9000)
    s3_public_endpoint_url: str | None = None
    s3_bucket: str = "vidgen"
    s3_region: str = "us-east-1"
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None
    presign_expires_s: int = 3600

    budget_ceiling_usd: float = 100.0

    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None
    fal_key: SecretStr | None = None
    kling_access_key: SecretStr | None = None
    kling_secret_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
