import torch
import torch.nn as nn


''' BLOCK 1: Patch Embedding
 Splits the image into non-overlapping patch_size x patch_size patches
 and linearly projects each patch to `embed_dim`.'''

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        # a stride-`patch_size` conv == "split into patches + linear project"
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                       # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)        # (B, N, embed_dim)
        x = self.norm(x)
        return x



''' BLOCK 2: Window partition / reverse helpers
Cuts the H x W token grid into non-overlapping window_size x window_size
 windows so attention can be computed locally (this is what makes Swin
 cheaper than full self-attention).'''


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



''' BLOCK 3: Window Attention
Multi-head self-attention computed *within each window*, plus a
 learned relative position bias (this replaces absolute positional
 encoding in Swin).'''

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias table: one bias value per head for
        # every possible relative (dy, dx) offset inside a window
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2,Wh,Ww
        coords_flatten = torch.flatten(coords, 1)                                # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x: (num_windows*B, N, C)  where N = window_size*window_size
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)                     # (B_, heads, N, N)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x




''' BLOCK 4: Swin Transformer Block
One block = (shifted-)window attention + MLP, each with a residual
 connection and pre-LayerNorm. Even-indexed blocks use regular windows
 (W-MSA), odd-indexed blocks shift the windows (SW-MSA) so information
 can flow across window boundaries.'''
class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class DropPath(nn.Module):
    """Stochastic depth: randomly drops the whole residual branch per-sample."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep_prob).floor_()
        return x.div(keep_prob) * mask


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        if min(input_resolution) <= window_size:
            # feature map smaller than a window: use one window, no shift
            self.shift_size = 0
            self.window_size = min(input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.register_buffer("attn_mask", self._build_shift_mask(dim), persistent=False)

    def _build_shift_mask(self, dim):
        if self.shift_size == 0:
            return None
        H, W = self.input_resolution
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        windows = window_partition(shifted_x, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --------------------------------------------------------------------
# BLOCK 5: Patch Merging
# Downsamples the token grid by 2x and doubles the channel dim --
# this is what creates the hierarchical (pyramid) feature maps.
# --------------------------------------------------------------------
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)                     # (B, H/2*W/2, 2C)
        return x


# --------------------------------------------------------------------
# BLOCK 6: Stage (a stack of SwinBlocks + optional PatchMerging)
# --------------------------------------------------------------------
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, downsample=None, drop_path=0.0):
        super().__init__()
        if not isinstance(drop_path, (list, tuple)):
            drop_path = [drop_path] * depth
        self.blocks = nn.ModuleList([
            SwinBlock(dim, input_resolution, num_heads, window_size,
                      shift_size=0 if (i % 2 == 0) else window_size // 2,
                      drop_path=drop_path[i])
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# --------------------------------------------------------------------
# BLOCK 7: Full Swin Encoder (Swin-T sized by default)
# --------------------------------------------------------------------
class SwinEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7,
                 drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        grid = self.patch_embed.grid_size

        # linearly increasing stochastic-depth rate across all encoder blocks
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.layers = nn.ModuleList()
        dim, res = embed_dim, grid
        block_idx = 0
        for i in range(len(depths)):
            downsample = PatchMerging if i < len(depths) - 1 else None
            self.layers.append(BasicLayer(dim, (res, res), depths[i], num_heads[i], window_size, downsample,
                                           drop_path=dpr[block_idx:block_idx + depths[i]]))
            block_idx += depths[i]
            if downsample is not None:
                dim *= 2
                res //= 2

        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim      # 768 for Swin-T defaults
        self.out_res = res      # 7  -> 49 tokens

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x   # (B, out_res*out_res, out_dim) e.g. (B, 49, 768)


if __name__ == "__main__":
    m = SwinEncoder()
    dummy = torch.randn(2, 3, 224, 224)
    out = m(dummy)
    print("encoder output shape:", out.shape)   # -> (2, 49, 768)
