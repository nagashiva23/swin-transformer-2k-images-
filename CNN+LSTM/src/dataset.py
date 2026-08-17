from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .vocabulary import Vocabulary, read_captions


def build_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class RoCoCaptionDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        vocabulary: Vocabulary,
        image_size: int = 224,
        max_length: int = 48,
        limit: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.rows = read_captions(self.data_root / f"{split}_captions.csv", limit)
        image_root = self.data_root / f"{split}_images"
        self.image_dir = image_root / split if (image_root / split).is_dir() else image_root
        self.transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def _find_image_path(self, image_id: str) -> Path:
        candidates = [
            self.image_dir / f"{image_id}.jpg",
            self.image_dir / f"{image_id}.jpeg",
            self.image_dir / f"{image_id}.png",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Image not found for ID {image_id} in {self.image_dir}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_id, caption = self.rows[index]
        image_path = self._find_image_path(image_id)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            pixels = self.transform(image)
        caption_ids = torch.tensor(self.vocabulary.encode(caption, self.max_length), dtype=torch.long)
        return pixels, caption_ids, image_id

