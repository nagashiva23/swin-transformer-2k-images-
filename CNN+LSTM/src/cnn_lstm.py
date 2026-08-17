from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class CNNEncoder(nn.Module):
    def __init__(self, projected_dim: int = 512, freeze_backbone: bool = True) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Linear(2048, projected_dim)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = features.flatten(1)
        return self.projection(features)


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 512,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 1,
        pad_id: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.init_hidden = nn.Linear(feature_dim, hidden_dim * num_layers)
        self.init_cell = nn.Linear(feature_dim, hidden_dim * num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def _initial_state(self, image_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = image_features.size(0)
        hidden = torch.tanh(self.init_hidden(image_features)).view(batch_size, self.num_layers, self.hidden_dim)
        cell = torch.tanh(self.init_cell(image_features)).view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()
        cell = cell.permute(1, 0, 2).contiguous()
        return hidden, cell

    def forward(self, image_features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        inputs = captions[:, :-1]
        embeddings = self.embedding(inputs)
        initial_state = self._initial_state(image_features)
        outputs, _ = self.lstm(embeddings, initial_state)
        return self.output(outputs)

    @torch.no_grad()
    def greedy_decode(
        self,
        image_features: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_length: int = 48,
    ) -> torch.Tensor:
        batch_size = image_features.size(0)
        hidden, cell = self._initial_state(image_features)
        current_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=image_features.device)
        generated = [current_tokens]

        for _ in range(max_length - 1):
            embeddings = self.embedding(current_tokens[:, -1:])
            outputs, (hidden, cell) = self.lstm(embeddings, (hidden, cell))
            logits = self.output(outputs[:, -1, :])
            next_tokens = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_tokens)
            current_tokens = torch.cat([current_tokens, next_tokens], dim=1)
            if (next_tokens == eos_id).all():
                break

        return torch.cat(generated, dim=1)


class CNNLSTMCaptioner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 512,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        lstm_layers: int = 1,
        pad_id: int = 0,
        freeze_encoder: bool = True,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = CNNEncoder(projected_dim=feature_dim, freeze_backbone=freeze_encoder)
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            pad_id=pad_id,
            dropout=decoder_dropout,
        )

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encoder(images)
        logits = self.decoder(image_features, captions)
        return logits, image_features.unsqueeze(1)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, bos_id: int, eos_id: int, max_length: int = 48) -> torch.Tensor:
        image_features = self.encoder(images)
        return self.decoder.greedy_decode(image_features, bos_id=bos_id, eos_id=eos_id, max_length=max_length)

