"""
BLOCK: Dataset
--------------
Loads ROCOv2 images + captions from your local paths, subsampled to
`max_samples` (default 2000) for a manageable training run.

NOTE: Check your actual CSV column names first:
    import pandas as pd
    print(pd.read_csv("train_captions.csv").columns)
Common ROCOv2 columns are something like ['ID', 'Caption'] -- adjust
IMAGE_COL / CAPTION_COL below to match exactly what you see.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMAGE_COL = "ID"        # <-- change if your csv uses a different column name
CAPTION_COL = "Caption"  # <-- change if your csv uses a different column name


class ROCODataset(Dataset):
    def __init__(self, csv_path, images_dir, vocab, max_len=40,
                 max_samples=2000, transform=None):
        df = pd.read_csv(csv_path)
        if max_samples is not None:
            df = df.iloc[:max_samples].reset_index(drop=True)
        self.df = df
        self.images_dir = images_dir
        self.vocab = vocab
        self.max_len = max_len

        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, image_id):
        image_id = str(image_id)
        if os.path.splitext(image_id)[1]:   # already has an extension
            return os.path.join(self.images_dir, image_id)
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(self.images_dir, image_id + ext)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(self.images_dir, image_id + ".jpg")  # fallback

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row[IMAGE_COL])
        caption = row[CAPTION_COL]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption_ids = torch.tensor(
            self.vocab.encode(caption, self.max_len), dtype=torch.long
        )
        return image, caption_ids


def build_vocab_from_csv(csv_path, min_freq=2, max_samples=2000):
    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.iloc[:max_samples]
    from vocab import Vocab
    return Vocab(df[CAPTION_COL].astype(str).tolist(), min_freq=min_freq)
