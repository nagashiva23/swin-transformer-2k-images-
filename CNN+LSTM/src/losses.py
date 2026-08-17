from __future__ import annotations

import torch
from torch import nn


class CaptionCrossEntropyLoss(nn.Module):
    def __init__(self, pad_id: int) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        vocab_size = logits.size(-1)
        loss = self.criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        return loss, {"caption": float(loss.detach())}

