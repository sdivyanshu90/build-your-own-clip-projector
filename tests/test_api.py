import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from clip_projector.config import Settings
from clip_projector.main import create_app


class FakeProjector:
    def text_embeddings(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor([[1.0, 0.0] for _ in texts])

    def image_embedding(self, _: bytes) -> torch.Tensor:
        return torch.tensor([[0.0, 1.0]])

    def similarity(self, _: bytes, __: str) -> float:
        return 0.25


def png() -> bytes:
    payload = io.BytesIO()
    Image.new("RGB", (2, 2)).save(payload, "PNG")
    return payload.getvalue()


def client() -> TestClient:
    app = create_app(Settings(environment="test", api_keys="secret", rate_limit_per_minute=20), FakeProjector())
    return TestClient(app)


def test_health_and_authorized_text_embedding() -> None:
    api = client()
    assert api.get("/healthz").status_code == 200
    response = api.post("/v1/embeddings/text", headers={"X-API-Key": "secret"}, json={"texts": ["hello"]})
    assert response.status_code == 200
    assert response.json()["embeddings"] == [[1.0, 0.0]]


def test_auth_validation_and_image_errors() -> None:
    api = client()
    assert api.post("/v1/embeddings/text", json={"texts": ["hello"]}).status_code == 401
    assert api.post("/v1/embeddings/text", headers={"X-API-Key": "secret"}, json={"texts": [""]}).status_code == 422
    assert api.post("/v1/embeddings/image", headers={"X-API-Key": "secret"}, files={"image": ("x.txt", b"x", "text/plain")}).status_code == 415


def test_image_similarity_and_metrics() -> None:
    api = client()
    headers, files = {"X-API-Key": "secret"}, {"image": ("image.png", png(), "image/png")}
    assert api.post("/v1/embeddings/image", headers=headers, files=files).status_code == 200
    assert api.post("/v1/similarity?text=cat", headers=headers, files=files).json()["similarity"] == 0.25
    assert "clip_projector_requests_total" in api.get("/metrics").text
