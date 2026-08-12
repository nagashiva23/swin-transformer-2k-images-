import math
import torch
import torch.nn as nn


# BLOCK 1: Positional Encoding (sinusoidal, fixed -- not learned)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------
# BLOCK 2: Multi-Head Attention (manual, used for both self- and
# cross-attention below)
# --------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.q_proj(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # -100.0 instead of -inf: matches the convention already used in
            # swin_model.py's shift-window mask, which avoids a known PyTorch
            # MPS-backend issue where softmax over -inf can produce NaN.
            scores = scores.masked_fill(mask == 0, float(-100.0))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.out_proj(out)


# --------------------------------------------------------------------
# BLOCK 3: Decoder Layer
# masked self-attn (caption sees only past tokens)
#   -> cross-attn (caption tokens attend to Swin image tokens)
#   -> feed-forward
# each sub-layer wrapped in residual + LayerNorm
# --------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


# --------------------------------------------------------------------
# BLOCK 4: Full Caption Decoder (embedding + N decoder layers + output head)
# --------------------------------------------------------------------
class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=8, ff_dim=2048,
                 num_layers=6, max_len=40, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # tgt: (B, T) token ids (caption shifted right, i.e. input side)
        # memory: (B, N, d_model) Swin encoder tokens
        B, T = tgt.shape
        causal_mask = torch.tril(torch.ones(T, T, device=tgt.device)).view(1, 1, T, T)

        x = self.dropout(self.pos_enc(self.embed(tgt)))
        for layer in self.layers:
            x = layer(x, memory, causal_mask)
        return self.fc_out(x)   # (B, T, vocab_size) logits
