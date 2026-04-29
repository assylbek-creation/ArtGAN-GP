"""Image-grid helpers for sample visualization."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torchvision.utils import make_grid, save_image


def denormalize(images: Tensor, mean: float = 0.5, std: float = 0.5) -> Tensor:
    """Reverse ``Normalize(mean, std)`` so values land back in ``[0, 1]``."""
    return (images * std + mean).clamp(0.0, 1.0)


def make_sample_grid(images: Tensor, nrow: int = 8) -> Tensor:
    """Build a single image tensor (3, H, W) tiling ``images`` (B, 3, h, w) into a grid."""
    return make_grid(denormalize(images), nrow=nrow, padding=2)


def save_sample_grid(images: Tensor, path: str | Path, nrow: int = 8) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images), str(out), nrow=nrow, padding=2)
    return out


@torch.no_grad()
def generate_grid(generator, latent: Tensor, nrow: int = 8) -> Tensor:
    """Run ``generator`` in eval mode on ``latent`` and tile the output."""
    was_training = generator.training
    generator.eval()
    try:
        samples = generator(latent)
    finally:
        generator.train(was_training)
    return make_sample_grid(samples, nrow=nrow)
