from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from clip_projector.config import Settings, get_settings
from clip_projector.preprocessing import InvalidImageError
from clip_projector.schemas import EmbeddingsResponse, HealthResponse, SimilarityResponse, TextEmbeddingRequest
from clip_projector.security import SlidingWindowRateLimiter, client_identifier, verify_api_key
from clip_projector.service import ProjectorService

REQUESTS = Counter("clip_projector_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("clip_projector_request_duration_seconds", "HTTP request duration", ["path"])


def configure_logging(level: str) -> None:
    logging.basicConfig(format="%(message)s", level=level.upper())
    structlog.configure(processors=[structlog.processors.add_log_level, structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()])


def create_app(settings: Settings | None = None, service: ProjectorService | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    projector = service or ProjectorService(settings.model_device, settings.checkpoint_path)
    limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="CLIP Projector API", version="1.0.0", lifespan=lifespan)
    app.state.projector, app.state.settings, app.state.limiter = projector, settings, limiter
    app.add_middleware(CORSMiddleware, allow_origins=settings.origins, allow_methods=["POST", "GET"], allow_headers=["X-API-Key", "Content-Type"], allow_credentials=False)

    @app.middleware("http")
    async def audit_request(request: Request, call_next: object) -> Response:
        started = time.perf_counter()
        try:
            response = await call_next(request)  # type: ignore[operator]
        except Exception:
            REQUESTS.labels(request.method, request.url.path, "500").inc()
            raise
        LATENCY.labels(request.url.path).observe(time.perf_counter() - started)
        REQUESTS.labels(request.method, request.url.path, str(response.status_code)).inc()
        return response

    @app.exception_handler(InvalidImageError)
    async def invalid_image(_: Request, exc: InvalidImageError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def authorize(request: Request) -> None:
        verify_api_key(request, settings)
        if not limiter.allow(client_identifier(request)):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")

    @app.get("/healthz", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=True)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/embeddings/text", response_model=EmbeddingsResponse, dependencies=[Depends(authorize)], tags=["embeddings"])
    def embed_text(request: TextEmbeddingRequest) -> EmbeddingsResponse:
        values = projector.text_embeddings(request.texts).tolist()
        return EmbeddingsResponse(model="clip-projector-1", dimension=len(values[0]), embeddings=values)

    @app.post("/v1/embeddings/image", response_model=EmbeddingsResponse, dependencies=[Depends(authorize)], tags=["embeddings"])
    async def embed_image(image: UploadFile = File(...)) -> EmbeddingsResponse:
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "only JPEG, PNG, and WebP are allowed")
        payload = await image.read(settings.max_upload_bytes + 1)
        if not payload or len(payload) > settings.max_upload_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "image exceeds configured size limit")
        values = projector.image_embedding(payload).tolist()
        return EmbeddingsResponse(model="clip-projector-1", dimension=len(values[0]), embeddings=values)

    @app.post("/v1/similarity", response_model=SimilarityResponse, dependencies=[Depends(authorize)], tags=["similarity"])
    async def similarity(text: str, image: UploadFile = File(...)) -> SimilarityResponse:
        if not text.strip() or len(text) > 10_000:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "text must be nonempty and at most 10000 characters")
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "only JPEG, PNG, and WebP are allowed")
        payload = await image.read(settings.max_upload_bytes + 1)
        if not payload or len(payload) > settings.max_upload_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "image exceeds configured size limit")
        return SimilarityResponse(model="clip-projector-1", similarity=projector.similarity(payload, text))

    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run("clip_projector.main:app", host=settings.host, port=settings.port, log_level=settings.log_level.lower())
