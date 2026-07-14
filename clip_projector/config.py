from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CLIP_", extra="ignore")

    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    log_level: str = "INFO"
    api_keys: str = ""
    require_api_key: bool = True
    model_device: str = "cpu"
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1024, le=50 * 1024 * 1024)
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10_000)
    cors_origins: str = ""
    checkpoint_path: str = ""

    @field_validator("environment")
    @classmethod
    def valid_environment(cls, value: str) -> str:
        if value not in {"development", "staging", "production", "test"}:
            raise ValueError("environment must be development, staging, production, or test")
        return value

    @property
    def key_set(self) -> frozenset[str]:
        return frozenset(key.strip() for key in self.api_keys.split(",") if key.strip())

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
