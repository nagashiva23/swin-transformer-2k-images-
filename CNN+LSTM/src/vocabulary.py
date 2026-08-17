from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\s\w]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower().strip())


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    document_frequency: dict[str, int]

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    @classmethod
    def build(cls, captions: Iterable[str], min_frequency: int = 2, max_size: int = 8000) -> "Vocabulary":
        frequency: Counter[str] = Counter()
        document_frequency: Counter[str] = Counter()
        for caption in captions:
            tokens = tokenize(caption)
            frequency.update(tokens)
            document_frequency.update(set(tokens))

        specials = [cls.PAD, cls.UNK, cls.BOS, cls.EOS]
        learned = [
            token
            for token, count in frequency.most_common()
            if count >= min_frequency and token not in specials
        ][: max_size - len(specials)]
        id_to_token = specials + learned
        return cls(
            token_to_id={token: index for index, token in enumerate(id_to_token)},
            id_to_token=id_to_token,
            document_frequency=dict(document_frequency),
        )

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    def encode(self, caption: str, max_length: int) -> list[int]:
        tokens = tokenize(caption)[: max_length - 2]
        ids = [self.bos_id] + [self.token_to_id.get(token, self.token_to_id[self.UNK]) for token in tokens] + [self.eos_id]
        return ids + [self.pad_id] * (max_length - len(ids))

    def decode(self, token_ids: list[int]) -> str:
        words: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token[token_id] if token_id < len(self.id_to_token) else self.UNK
            if token == self.EOS:
                break
            if token in (self.BOS, self.PAD):
                continue
            words.append(token)
        return " ".join(words)

    def save(self, path: Path) -> None:
        payload = {
            "id_to_token": self.id_to_token,
            "document_frequency": self.document_frequency,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        id_to_token = payload["id_to_token"]
        return cls(
            token_to_id={token: index for index, token in enumerate(id_to_token)},
            id_to_token=id_to_token,
            document_frequency=payload.get("document_frequency", {}),
        )


def read_captions(csv_path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        id_key, caption_key = None, None
        if reader.fieldnames:
            for field in reader.fieldnames:
                name = field.lower()
                if name == "id":
                    id_key = field
                elif name == "caption":
                    caption_key = field
        id_key = id_key or "ID"
        caption_key = caption_key or "Caption"
        for row in reader:
            image_id = row.get(id_key)
            caption = row.get(caption_key)
            if image_id and caption:
                rows.append((image_id, caption))
            if limit is not None and len(rows) >= limit:
                break
    return rows

