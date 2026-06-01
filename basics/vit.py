"""Vision Transformer — §2 (with §5 return_all_tokens and §6 RoPE options).

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from basics.model import Block
from basics.rope import RoPE1D, RoPE2D


class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # A strided Conv2d with kernel=stride=patch_size extracts each
        # non-overlapping patch and linearly projects it to d_model.
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.conv(x)            # (B, d_model, H/patch, W/patch)
        patches = patches.flatten(2)      # (B, d_model, num_patches)
        patches = patches.transpose(1, 2)  # (B, num_patches, d_model)
        return patches


# ---------------------------------------------------------------------------
# RoPE-aware attention/block for the §6 ablations.
# (basics/model.py is PROVIDED and not modified; for RoPE we use these instead.)
# ---------------------------------------------------------------------------


class _RoPEAttention(nn.Module):
    """Multi-head self-attention that applies a rotary embedding to q and k.

    The rotation is supplied by the caller as `apply_rope(t)` so the same module
    works for 1D and 2D RoPE.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,hd)

    def forward(self, x: torch.Tensor, apply_rope) -> torch.Tensor:
        B, T, _ = x.shape
        q = apply_rope(self._split(self.q_proj(x)))
        k = apply_rope(self._split(self.k_proj(x)))
        v = self._split(self.v_proj(x))
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = attn @ v                                               # (B,H,T,hd)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.dropout(self.out_proj(out))


class _RoPEBlock(nn.Module):
    """Pre-LN transformer block using `_RoPEAttention` (bidirectional)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _RoPEAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, apply_rope) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), apply_rope)
        x = x + self.dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x


class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. Add a learnable positional embedding (pos_encoding="learned"), or apply
         RoPE to q/k inside attention (pos_encoding="rope_1d"/"rope_2d", §6).
      4. Pass the sequence through `num_blocks` Transformer Blocks (is_decoder=False).
      5. Apply a final LayerNorm.
      6. Return the [CLS] slice (B, d_model), or all tokens if return_all_tokens.

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
        pos_encoding: "learned" (default), "rope_1d", or "rope_2d" (§6 ablation).
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
        pos_encoding: str = "learned",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2
        self.grid = img_size // patch_size
        self.pos_encoding = pos_encoding

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)

        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.blocks = nn.ModuleList(
                Block(d_model, num_heads, block_size=self.num_patches + 1,
                      dropout=dropout, is_decoder=False)
                for _ in range(num_blocks)
            )
            self.rope = None
        else:
            head_dim = d_model // num_heads
            self.blocks = nn.ModuleList(
                _RoPEBlock(d_model, num_heads, dropout) for _ in range(num_blocks)
            )
            if pos_encoding == "rope_1d":
                # CLS at index 0, patches at 1..N (1D raster order).
                self.rope = RoPE1D(head_dim, max_seq_len=self.num_patches + 1)
                positions = torch.arange(self.num_patches + 1)
                self.register_buffer("positions", positions, persistent=False)
            elif pos_encoding == "rope_2d":
                # CLS at (0,0); patches shifted to 1..grid so they don't collide.
                self.rope = RoPE2D(head_dim, grid_size=self.grid + 1)
                ys, xs = torch.meshgrid(
                    torch.arange(self.grid), torch.arange(self.grid), indexing="ij"
                )
                x_coords = torch.cat([torch.zeros(1, dtype=torch.long), xs.flatten() + 1])
                y_coords = torch.cat([torch.zeros(1, dtype=torch.long), ys.flatten() + 1])
                self.register_buffer("x_coords", x_coords, persistent=False)
                self.register_buffer("y_coords", y_coords, persistent=False)
            else:
                raise ValueError(f"unknown pos_encoding: {pos_encoding}")

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.final_ln = nn.LayerNorm(d_model)

    def _apply_rope(self, t: torch.Tensor) -> torch.Tensor:
        if self.pos_encoding == "rope_1d":
            return self.rope(t, self.positions)
        return self.rope(t, self.x_coords, self.y_coords)

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                               # (B, N, d_model)
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                        # (B, N+1, d_model)

        if self.pos_encoding == "learned":
            x = x + self.pos_embed
            for block in self.blocks:
                x = block(x)
        else:
            for block in self.blocks:
                x = block(x, self._apply_rope)

        x = self.final_ln(x)                                  # (B, N+1, d_model)
        if return_all_tokens:
            return x
        return x[:, 0]                                        # (B, d_model)
