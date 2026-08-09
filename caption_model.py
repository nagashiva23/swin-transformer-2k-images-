"""
Full model: Swin Encoder (image -> tokens) + Transformer Decoder
(tokens -> caption). This is the encoder-decoder captioning model.
"""

import torch
import torch.nn as nn
from swin_model import SwinEncoder
from decoder_model import CaptionDecoder


class SwinCaptioningModel(nn.Module):
    def __init__(self, vocab_size, img_size=224, window_size=7,
                 embed_dim=96, depths=(2, 2, 6, 2), enc_heads=(3, 6, 12, 24),
                 dec_heads=8, dec_layers=6, max_len=40, dropout=0.1):
        super().__init__()
        self.encoder = SwinEncoder(
            img_size=img_size, embed_dim=embed_dim, depths=depths,
            num_heads=enc_heads, window_size=window_size,
        )
        d_model = self.encoder.out_dim   # 768 with Swin-T defaults
        self.decoder = CaptionDecoder(
            vocab_size=vocab_size, d_model=d_model, num_heads=dec_heads,
            ff_dim=d_model * 4, num_layers=dec_layers, max_len=max_len, dropout=dropout,
        )

    def forward(self, images, captions):
        """
        images: (B, 3, 224, 224)
        captions: (B, T) full caption ids including <sos> ... <eos> <pad>...
        returns logits for predicting captions[:, 1:] given captions[:, :-1]
        """
        memory = self.encoder(images)          # (B, 49, d_model)
        tgt_in = captions[:, :-1]               # teacher forcing input
        logits = self.decoder(tgt_in, memory)   # (B, T-1, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, images, vocab, max_len=40, device="cpu"):
        """Greedy decoding for inference (batch size 1 recommended)."""
        self.eval()
        memory = self.encoder(images.to(device))
        B = images.size(0)
        ids = torch.full((B, 1), vocab.sos_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(ids, memory)
            next_id = logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if (next_id == vocab.eos_id).all():
                break
        return ids

