"""
Full model: Swin Encoder (image -> tokens) + Transformer Decoder
(tokens -> caption). This is the encoder-decoder captioning model.
"""

import torch
import torch.nn as nn
from swin_model import SwinEncoder
from decoder_model import CaptionDecoder


def _init_weights(module):
    # relative_position_bias_table already gets its own trunc_normal_ init
    # in WindowAttention.__init__ and isn't a Linear/LayerNorm, so it's untouched here.
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class SwinCaptioningModel(nn.Module):
    def __init__(self, vocab_size, img_size=224, window_size=7,
                 embed_dim=96, depths=(2, 2, 6, 2), enc_heads=(3, 6, 12, 24),
                 dec_heads=8, dec_layers=6, max_len=40, dropout=0.1):
        super().__init__()
        self.encoder = SwinEncoder(
            img_size=img_size, embed_dim=embed_dim, depths=depths,
            num_heads=enc_heads, window_size=window_size,
        )
        d_model = self.encoder.out_dim
        self.decoder = CaptionDecoder(
            vocab_size=vocab_size, d_model=d_model, num_heads=dec_heads,
            ff_dim=d_model * 4, num_layers=dec_layers, max_len=max_len, dropout=dropout,
        )
        self.apply(_init_weights)

    def forward(self, images, captions):
        memory = self.encoder(images)
        tgt_in = captions[:, :-1]
        logits = self.decoder(tgt_in, memory)
        return logits

    @torch.no_grad()
    def generate(self, images, vocab, max_len=40, device="cpu"):
        self.eval()
        memory = self.encoder(images.to(device))
        B = images.size(0)
        ids = torch.full((B, 1), vocab.sos_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(ids, memory)
            step_logits = logits[:, -1, :].clone()

            prev_token = ids[:, -1]
            step_logits.scatter_(1, prev_token.unsqueeze(1), float(-100.0))

            seq_len = ids.size(1)
            if seq_len >= 2:
                for b in range(ids.size(0)):
                    seq = ids[b].tolist()
                    seen_trigrams = set()
                    for i in range(len(seq) - 2):
                        seen_trigrams.add((seq[i], seq[i + 1], seq[i + 2]))
                    prefix = (seq[-2], seq[-1])
                    banned = {t[2] for t in seen_trigrams if (t[0], t[1]) == prefix}
                    for banned_next in banned:
                        step_logits[b, banned_next] = float(-100.0)

            next_id = step_logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if (next_id == vocab.eos_id).all():
                break
        return ids
