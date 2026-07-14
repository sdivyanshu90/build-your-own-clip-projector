import torch

from clip_projector.model import CLIPProjector, ModelConfig


def test_dual_encoder_returns_square_logits_and_loss() -> None:
    config = ModelConfig(image_size=32, patch_size=16, context_length=8, width=32, layers=1, heads=4, projection_dim=16)
    model = CLIPProjector(config)
    images, tokens = torch.randn(2, 3, 32, 32), torch.tensor([[1, 2, 0, 0, 0, 0, 0, 0], [3, 4, 5, 0, 0, 0, 0, 0]])
    image_logits, text_logits = model(images, tokens)
    assert image_logits.shape == text_logits.shape == (2, 2)
    assert model.contrastive_loss(images, tokens).isfinite()


def test_embeddings_are_unit_normalized() -> None:
    config = ModelConfig(image_size=32, patch_size=16, context_length=8, width=32, layers=1, heads=4, projection_dim=16)
    model = CLIPProjector(config)
    assert torch.allclose(model.encode_image(torch.randn(1, 3, 32, 32)).norm(dim=-1), torch.ones(1), atol=1e-5)
