import torch

from clip_projector.service import ProjectorService


def test_checkpoint_missing_fails_clearly() -> None:
    service = ProjectorService.__new__(ProjectorService)
    service.device = torch.device("cpu")
    try:
        service.load_checkpoint("/does/not/exist")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing checkpoint accepted")
