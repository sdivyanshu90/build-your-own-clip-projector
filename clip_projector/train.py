from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from clip_projector.model import CLIPProjector
from clip_projector.preprocessing import batch_tokens, preprocess_image


class CaptionDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, manifest: Path, image_root: Path) -> None:
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not records or any(not isinstance(row.get("image"), str) or not isinstance(row.get("text"), str) for row in records):
            raise ValueError("manifest must contain nonempty JSONL records with image and text strings")
        self.records, self.image_root = records, image_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.records[index]
        return preprocess_image((self.image_root / row["image"]).read_bytes()), row["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLIP projector from JSONL image/text pairs")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    model, device = CLIPProjector().to(args.device), torch.device(args.device)
    dataset = CaptionDataset(args.manifest, args.image_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.2)
    for _ in range(args.epochs):
        model.train()
        for images, texts in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.contrastive_loss(images.to(device), batch_tokens(list(texts), model.config).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict()}, args.output)


if __name__ == "__main__":
    main()
