import math
import torch
import torch.nn as nn
import torchvision.models as models
from swin_model import SwinEncoder
from decoder_model import CaptionDecoder


def _init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class PretrainedSwinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.Swin_T_Weights.DEFAULT
        swin = models.swin_t(weights=weights)
        self.features = swin.features
        self.norm = swin.norm
        self.out_dim = 768

    def forward(self, x):
        # x: (B, 3, 224, 224)
        feat = self.features(x)        # (B, 7, 7, 768)
        normed = self.norm(feat)       # (B, 7, 7, 768)
        memory = normed.flatten(1, 2)  # (B, 49, 768)
        return memory


class SwinCaptioningModel(nn.Module):
    def __init__(self, vocab_size, img_size=224, window_size=7,
                 embed_dim=96, depths=(2, 2, 6, 2), enc_heads=(3, 6, 12, 24),
                 dec_heads=8, dec_layers=6, max_len=40, dropout=0.20, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        if pretrained:
            self.encoder = PretrainedSwinEncoder()
            d_model = self.encoder.out_dim
        else:
            self.encoder = SwinEncoder(
                img_size=img_size, embed_dim=embed_dim, depths=depths,
                num_heads=enc_heads, window_size=window_size,
            )
            d_model = self.encoder.out_dim
            self.encoder.apply(_init_weights)

        self.decoder = CaptionDecoder(
            vocab_size=vocab_size, d_model=d_model, num_heads=dec_heads,
            ff_dim=d_model * 4, num_layers=dec_layers, max_len=max_len, dropout=dropout,
        )
        self.decoder.apply(_init_weights)

    def forward(self, images, captions, word_drop_p=0.10, unk_id=3, pad_id=0, sos_id=1, eos_id=2):
        memory = self.encoder(images)
        tgt_in = captions[:, :-1].clone()

        if self.training and word_drop_p > 0.0:
            # Word dropout: randomly replace non-special tokens with <unk>
            mask = (torch.rand_like(tgt_in, dtype=torch.float) < word_drop_p) & \
                   (tgt_in != pad_id) & (tgt_in != sos_id) & (tgt_in != eos_id)
            tgt_in[mask] = unk_id

        logits = self.decoder(tgt_in, memory)
        return logits

    @torch.no_grad()
    def generate(self, images, vocab, max_len=40, device="cpu"):
        """Greedy decoding with n-gram blocking and repetition penalties."""
        self.eval()
        images = images.to(device)
        memory = self.encoder(images)
        B = images.size(0)
        ids = torch.full((B, 1), vocab.sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(ids, memory)
            step_logits = logits[:, -1, :].clone()

            prev_token = ids[:, -1]
            for b in range(B):
                if not finished[b]:
                    p = prev_token[b].item()
                    if p != vocab.eos_id and p != vocab.pad_id:
                        step_logits[b, p] = float(-100.0)

            seq_len = ids.size(1)
            if seq_len >= 2:
                for b in range(B):
                    if finished[b]:
                        continue
                    seq = ids[b].tolist()
                    seen_trigrams = set()
                    for i in range(len(seq) - 2):
                        seen_trigrams.add((seq[i], seq[i + 1], seq[i + 2]))
                    prefix = (seq[-2], seq[-1])
                    banned = {t[2] for t in seen_trigrams if (t[0], t[1]) == prefix}
                    for banned_next in banned:
                        step_logits[b, banned_next] = float(-100.0)

            next_id = step_logits.argmax(-1, keepdim=True)
            next_id = torch.where(finished.unsqueeze(1), torch.full_like(next_id, vocab.pad_id), next_id)
            finished = finished | (next_id.squeeze(1) == vocab.eos_id)
            ids = torch.cat([ids, next_id], dim=1)
            if finished.all():
                break
        return ids

    @torch.no_grad()
    def generate_beam(self, images, vocab, beam_width=3, max_len=40, length_penalty=0.7, device="cpu"):
        """Beam search decoding for higher fluency and metric performance."""
        self.eval()
        images = images.to(device)
        memory = self.encoder(images)
        B = images.size(0)
        final_sequences = []

        for b in range(B):
            mem_b = memory[b:b+1]  # (1, 49, d_model)
            # Each beam candidate: (score, token_list, finished)
            beams = [(0.0, [vocab.sos_id], False)]

            for step in range(max_len - 1):
                candidates = []
                all_finished = True

                for score, seq, fin in beams:
                    if fin:
                        candidates.append((score, seq, True))
                        continue

                    all_finished = False
                    inp = torch.tensor([seq], dtype=torch.long, device=device)
                    logits = self.decoder(inp, mem_b)[:, -1, :]  # (1, vocab_size)
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                    # Repetition penalty on previous token
                    if len(seq) > 0 and seq[-1] not in (vocab.sos_id, vocab.eos_id, vocab.pad_id):
                        log_probs[seq[-1]] -= 1.5

                    # Trigram blocking
                    if len(seq) >= 2:
                        seen_trigrams = set()
                        for i in range(len(seq) - 2):
                            seen_trigrams.add((seq[i], seq[i + 1], seq[i + 2]))
                        prefix = (seq[-2], seq[-1])
                        banned = {t[2] for t in seen_trigrams if (t[0], t[1]) == prefix}
                        for ban in banned:
                            log_probs[ban] = float("-inf")

                    topk_scores, topk_tokens = torch.topk(log_probs, beam_width)
                    for k in range(beam_width):
                        tok = topk_tokens[k].item()
                        next_score = score + topk_scores[k].item()
                        is_end = (tok == vocab.eos_id)
                        candidates.append((next_score, seq + [tok], is_end))

                if all_finished:
                    break

                # Sort by length-normalized score
                def score_fn(item):
                    s, q, f = item
                    # length penalty: (5 + len)^alpha / (5 + 1)^alpha
                    lp = ((5.0 + len(q)) / 6.0) ** length_penalty
                    return s / lp

                candidates.sort(key=score_fn, reverse=True)
                beams = candidates[:beam_width]

            best_seq = beams[0][1]
            while len(best_seq) < max_len:
                best_seq.append(vocab.pad_id)
            final_sequences.append(best_seq[:max_len])

        return torch.tensor(final_sequences, dtype=torch.long, device=device)
