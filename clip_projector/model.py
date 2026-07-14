from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    vocab_size: int = 49408
    context_length: int = 77
    width: int = 256
    layers: int = 6
    heads: int = 8
    projection_dim: int = 256


class VisionEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        patches = (config.image_size // config.patch_size) ** 2
        self.conv = nn.Conv2d(3, config.width, config.patch_size, config.patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(config.width) * config.width**-0.5)
        self.position_embedding = nn.Parameter(torch.randn(patches + 1, config.width) * 0.01)
        layer = nn.TransformerEncoderLayer(config.width, config.heads, config.width * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.layers)
        self.norm = nn.LayerNorm(config.width)

    def forward(self, image: Tensor) -> Tensor:
        tokens = self.conv(image).flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(image.size(0), 1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self.position_embedding
        return self.norm(self.transformer(tokens)[:, 0])


class TextEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Parameter(torch.randn(config.context_length, config.width) * 0.01)
        layer = nn.TransformerEncoderLayer(config.width, config.heads, config.width * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.layers)
        self.norm = nn.LayerNorm(config.width)

    def forward(self, tokens: Tensor) -> Tensor:
        padding_mask = tokens.eq(0)
        x = self.token_embedding(tokens) + self.position_embedding[: tokens.size(1)]
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        lengths = tokens.ne(0).sum(dim=1).clamp(min=1) - 1
        return self.norm(x[torch.arange(x.size(0), device=x.device), lengths])


class CLIPProjector(nn.Module):
    """A trainable contrastive image/text dual encoder."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.vision = VisionEncoder(self.config)
        self.text = TextEncoder(self.config)
        self.image_projection = nn.Linear(self.config.width, self.config.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.config.width, self.config.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: Tensor) -> Tensor:
        return F.normalize(self.image_projection(self.vision(images)), dim=-1)

    def encode_text(self, tokens: Tensor) -> Tensor:
        return F.normalize(self.text_projection(self.text(tokens)), dim=-1)

    def forward(self, images: Tensor, tokens: Tensor) -> tuple[Tensor, Tensor]:
        image_features, text_features = self.encode_image(images), self.encode_text(tokens)
        logits = self.logit_scale.exp().clamp(max=100) * image_features @ text_features.T
        return logits, logits.T

    def contrastive_loss(self, images: Tensor, tokens: Tensor) -> Tensor:
        image_logits, text_logits = self(images, tokens)
        labels = torch.arange(images.size(0), device=images.device)
        return (F.cross_entropy(image_logits, labels) + F.cross_entropy(text_logits, labels)) / 2
