from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TextEmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 10_000 for value in values):
            raise ValueError("texts must be nonempty and at most 10000 characters")
        return values


class EmbeddingsResponse(BaseModel):
    model: str
    dimension: int
    embeddings: list[list[float]]


class SimilarityResponse(BaseModel):
    model: str
    similarity: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
