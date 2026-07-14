import io

import pytest
from PIL import Image

from clip_projector.model import ModelConfig
from clip_projector.preprocessing import InvalidImageError, batch_tokens, preprocess_image, tokenize


def image_bytes() -> bytes:
    image = Image.new("RGB", (10, 20), "red")
    data = io.BytesIO()
    image.save(data, "PNG")
    return data.getvalue()


def test_image_preprocessing_returns_normalized_chw_tensor() -> None:
    assert preprocess_image(image_bytes(), 32).shape == (3, 32, 32)


def test_invalid_image_is_rejected() -> None:
    with pytest.raises(InvalidImageError):
        preprocess_image(b"not-an-image")


def test_tokenization_rejects_blank_and_batches() -> None:
    config = ModelConfig(context_length=4)
    with pytest.raises(ValueError):
        tokenize("   ", config)
    assert batch_tokens(["one two", "three"], config).shape == (2, 4)
