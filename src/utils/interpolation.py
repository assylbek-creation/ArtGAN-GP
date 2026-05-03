"""Latent-space interpolation utilities.

GAN latents drawn from ``N(0, I)`` in high dimensions concentrate on the
surface of a sphere of radius ``sqrt(latent_dim)``. Linear interpolation
between two such samples passes through the low-density interior — the
generator was never trained on those inputs. Spherical linear
interpolation (slerp) stays on the manifold and tends to produce smoother
visual transitions, so we expose both and default to slerp.
"""

from __future__ import annotations

import torch
from torch import Tensor


def lerp_path(z1: Tensor, z2: Tensor, n_steps: int) -> Tensor:
    """Linear interpolation. Returns ``(n_steps, latent_dim)``."""
    if z1.shape != z2.shape:
        raise ValueError(f"shape mismatch: {tuple(z1.shape)} vs {tuple(z2.shape)}")
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    ts = torch.linspace(0.0, 1.0, n_steps, device=z1.device, dtype=z1.dtype).unsqueeze(-1)
    return (1.0 - ts) * z1 + ts * z2


def slerp_path(z1: Tensor, z2: Tensor, n_steps: int, eps: float = 1e-6) -> Tensor:
    """Spherical linear interpolation. Returns ``(n_steps, latent_dim)``.

    Falls back to ``lerp_path`` when the two vectors are nearly colinear,
    since the slerp formula divides by ``sin(omega)`` which goes to zero.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"shape mismatch: {tuple(z1.shape)} vs {tuple(z2.shape)}")
    if z1.dim() != 1:
        raise ValueError(f"slerp_path expects 1-D latents, got {tuple(z1.shape)}")
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")

    cos_omega = torch.dot(z1, z2) / (z1.norm() * z2.norm() + eps)
    cos_omega = cos_omega.clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)

    if sin_omega.abs().item() < eps:
        return lerp_path(z1, z2, n_steps)

    ts = torch.linspace(0.0, 1.0, n_steps, device=z1.device, dtype=z1.dtype)
    a = (torch.sin((1.0 - ts) * omega) / sin_omega).unsqueeze(-1)
    b = (torch.sin(ts * omega) / sin_omega).unsqueeze(-1)
    return a * z1 + b * z2


def slerp_grid(
    corners: tuple[Tensor, Tensor, Tensor, Tensor], n_steps: int
) -> Tensor:
    """Bilinear-style slerp over four corner latents.

    Returns ``(n_steps * n_steps, latent_dim)`` rasterized row-by-row.
    Useful for a 2-D walk grid: top edge interpolates between corners 0
    and 1, bottom edge between 2 and 3, then each column slerps top->bot.
    """
    z00, z01, z10, z11 = corners
    top = slerp_path(z00, z01, n_steps)
    bot = slerp_path(z10, z11, n_steps)
    rows = []
    for j in range(n_steps):
        rows.append(slerp_path(top[j], bot[j], n_steps))
    return torch.stack(rows, dim=0).reshape(n_steps * n_steps, -1)
