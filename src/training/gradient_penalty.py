"""Gradient penalty for WGAN-GP (Gulrajani et al., 2017).

Implements ``lambda * E[(||grad_x_hat D(x_hat)||_2 - 1)^2]`` where each
``x_hat = eps * real + (1 - eps) * fake`` and ``eps ~ U(0, 1)`` is sampled
**per-sample** (one scalar per item in the batch, broadcast across the
spatial and channel dims).

Two details that quietly break training if violated:

- ``create_graph=True`` on ``torch.autograd.grad``: the penalty itself must
  be differentiable w.r.t. the critic parameters, otherwise the GP term
  contributes no gradient to the critic update.
- ``eps`` is per-sample (shape ``(B, 1, 1, 1)``), NOT a single scalar for
  the whole batch. Using a scalar collapses the penalty.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def gradient_penalty(critic: nn.Module, real: Tensor, fake: Tensor) -> Tensor:
    if real.shape != fake.shape:
        raise ValueError(f"real {tuple(real.shape)} and fake {tuple(fake.shape)} must match")

    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=real.device, dtype=real.dtype)
    interpolated = eps * real + (1.0 - eps) * fake
    interpolated.requires_grad_(True)

    score = critic(interpolated)
    grads = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads_flat = grads.view(batch_size, -1)
    grad_norm = grads_flat.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()
