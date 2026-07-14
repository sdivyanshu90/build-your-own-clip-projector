from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from clip_projector.model import CLIPProjector, ModelConfig
from clip_projector.preprocessing import batch_tokens, preprocess_image


class ProjectorService:
    def __init__(self, device: str = "cpu", checkpoint_path: str = "") -> None:
        self.device = torch.device(device)
        self.model = CLIPProjector().to(self.device).eval()
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()

    @torch.inference_mode()
    def text_embeddings(self, texts: list[str]) -> Tensor:
        return self.model.encode_text(batch_tokens(texts, self.model.config).to(self.device)).cpu()

    @torch.inference_mode()
    def image_embedding(self, payload: bytes) -> Tensor:
        image = preprocess_image(payload, self.model.config.image_size).unsqueeze(0).to(self.device)
        return self.model.encode_image(image).cpu()

    def similarity(self, payload: bytes, text: str) -> float:
        image, text_embedding = self.image_embedding(payload), self.text_embeddings([text])
        return float((image * text_embedding).sum().item())
