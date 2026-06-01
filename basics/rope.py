"""Rotary Position Embeddings — §6.

You implement: RoPE1D, RoPE2D.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the 2x2 rotation to adjacent dimension pairs (x_{2i}, x_{2i+1}).

    Following Eq. (3) of the writeup:
        out_{2i}   = x_{2i} cos - x_{2i+1} sin
        out_{2i+1} = x_{2i} sin + x_{2i+1} cos

    Args:
        x:   (..., d) with d even.
        cos: (..., d/2) cosines of the per-pair angles, already broadcastable to x[...,::2].
        sin: (..., d/2) sines.

    Returns:
        (..., d) rotated tensor, interleaved back into the original layout.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    # Interleave the two halves back to [e0, o0, e1, o1, ...].
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


class RoPE1D(nn.Module):
    """1D Rotary Position Embedding.

    For a vector x at position m, RoPE groups dimensions into d/2 pairs and
    rotates each pair (x_{2i}, x_{2i+1}) by angle m * theta_i, where
        theta_i = base ** (-2i / head_dim).

    Apply RoPE to queries and keys (not values) inside attention, before
    computing q @ k^T.

    Args:
        head_dim:    Dimensionality of each attention head. Must be even.
        max_seq_len: Maximum sequence length to precompute angles for.
        base:        Base of the geometric progression (typically 10_000).

    Forward:
        x:         (B, num_heads, T, head_dim)
        positions: (T,) integer tensor of token positions.
        returns:   (B, num_heads, T, head_dim) with RoPE applied.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # theta_i = base ** (-2i / head_dim), i = 0 .. head_dim/2 - 1.
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)  # (head_dim/2,)
        t = torch.arange(max_seq_len).float()                                 # (max_seq_len,)
        freqs = torch.outer(t, inv_freq)                                      # (max_seq_len, head_dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.to(self.cos_cached.device)
        cos = self.cos_cached[positions].to(x.dtype)  # (T, head_dim/2)
        sin = self.sin_cached[positions].to(x.dtype)
        # Broadcast over (B, num_heads): (T, d/2) -> (1, 1, T, d/2).
        cos = cos[None, None]
        sin = sin[None, None]
        return _rotate_pairs(x, cos, sin)


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for image patches.

    Splits head_dim in half. The first half rotates by the patch's x-coordinate
    using 1D RoPE; the second half rotates by the patch's y-coordinate. After
    rotation, dot products depend on the 2D *relative* offset between patches.

    Args:
        head_dim:  Must be divisible by 4 (since each half is split into
                   real/imaginary pairs).
        grid_size: Maximum grid side (patches per row).
        base:      Base of the geometric progression.

    Forward:
        x:        (B, num_heads, T, head_dim)
        x_coords: (T,) integer tensor of x positions on the grid.
        y_coords: (T,) integer tensor of y positions on the grid.
        returns:  (B, num_heads, T, head_dim) with 2D RoPE applied.
    """

    def __init__(self, head_dim: int, grid_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.head_dim = head_dim
        self.grid_size = grid_size
        self.base = base
        self.half = head_dim // 2  # dims allocated to each spatial axis

        # Each axis runs 1D RoPE over `half` dims -> half/2 = head_dim/4 frequency pairs.
        inv_freq = base ** (-torch.arange(0, self.half, 2).float() / self.half)  # (head_dim/4,)
        t = torch.arange(grid_size).float()
        freqs = torch.outer(t, inv_freq)  # (grid_size, head_dim/4)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def _apply_axis(self, x_half: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.to(self.cos_cached.device)
        cos = self.cos_cached[coords].to(x_half.dtype)[None, None]  # (1,1,T,head_dim/4)
        sin = self.sin_cached[coords].to(x_half.dtype)[None, None]
        return _rotate_pairs(x_half, cos, sin)

    def forward(
        self,
        x: torch.Tensor,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
    ) -> torch.Tensor:
        x_part = x[..., : self.half]   # rotated by the x-coordinate
        y_part = x[..., self.half :]   # rotated by the y-coordinate
        out_x = self._apply_axis(x_part, x_coords)
        out_y = self._apply_axis(y_part, y_coords)
        return torch.cat([out_x, out_y], dim=-1)
