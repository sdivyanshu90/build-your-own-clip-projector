from __future__ import annotations

import io
import re

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor

from clip_projector.model import ModelConfig

IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
TOKEN_PATTERN = re.compile(r"[\w']+|[^\s\w]", re.UNICODE)


class InvalidImageError(ValueError):
    pass


def preprocess_image(payload: bytes, image_size: int = 224) -> Tensor:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            image = source.convert("RGB")
            image.thumbnail((image_size, image_size))
            canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
            left, top = (image_size - image.width) // 2, (image_size - image.height) // 2
            canvas.paste(image, (left, top))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("file is not a valid decodable image") from exc
    data = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
    tensor = data.reshape(image_size, image_size, 3).permute(2, 0, 1).float().div(255)
    mean, std = torch.tensor(IMAGE_MEAN).view(3, 1, 1), torch.tensor(IMAGE_STD).view(3, 1, 1)
    return (tensor - mean) / std


def _token_id(piece: str, vocab_size: int) -> int:
    # Deterministic bounded hash; checkpoint training and serving share this tokenizer.
    value = 2166136261
    for byte in piece.encode("utf-8"):
        value = (value ^ byte) * 16777619 & 0xFFFFFFFF
    return 2 + value % (vocab_size - 2)


def tokenize(text: str, config: ModelConfig) -> Tensor:
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        raise ValueError("text must not be empty")
    pieces = TOKEN_PATTERN.findall(normalized)[: config.context_length]
    ids = [_token_id(piece, config.vocab_size) for piece in pieces]
    return torch.tensor(ids, dtype=torch.long)


def batch_tokens(texts: list[str], config: ModelConfig) -> Tensor:
    result = torch.zeros((len(texts), config.context_length), dtype=torch.long)
    for index, text in enumerate(texts):
        ids = tokenize(text, config)
        result[index, : ids.numel()] = ids
    return result
