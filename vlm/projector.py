"""Vision-Language Projector — §5.

You implement: VisionLanguageProjector.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionLanguageProjector(nn.Module):
    """2-layer MLP that maps image features into the decoder's embedding space.

    Architecture:
        Linear(d_image, expansion * d_image) -> GELU -> Linear(expansion * d_image, d_decoder)

    Must handle both:
      - A single pooled image vector:  input (B, d_image)         -> output (B, 1, d_decoder)
      - A sequence of patch vectors:   input (B, N_vis, d_image)  -> output (B, N_vis, d_decoder)

    Args:
        d_image:   Image-encoder embedding dim (your ViT's d_model).
        d_decoder: Decoder embedding dim (e.g., 960 for SmolLM2-360M).
        expansion: MLP hidden expansion factor (4 by default, à la LLaVA).
    """

    def __init__(self, d_image: int, d_decoder: int, expansion: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_image, expansion * d_image)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expansion * d_image, d_decoder)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # Promote a single pooled vector (B, d_image) to a length-1 sequence so
        # the same code path handles (B, N_vis, d_image) inputs.
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)  # (B, 1, d_image)
        return self.fc2(self.act(self.fc1(image_features)))  # (B, N_vis, d_decoder)
