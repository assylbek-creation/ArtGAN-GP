"""Unit tests for the WGAN-GP gradient penalty.

These nail down the load-bearing properties:

- Output is a non-negative scalar.
- Gradients flow back to the critic's parameters (i.e. ``create_graph=True``
  is doing its job — without it the GP would not influence the critic
  update).
- For a critic whose Jacobian w.r.t. its input is exactly 1 everywhere,
  the penalty is zero. A small offset increases it predictably.
- Mismatched shapes raise instead of silently broadcasting.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.training.gradient_penalty import gradient_penalty


class _UnitJacobianCritic(nn.Module):
    """f(x) = sum(x), so ||grad_x f(x)|| = sqrt(numel(x_per_sample))."""

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(1).sum(dim=1)


class _ScaledCritic(nn.Module):
    """f(x) = scale * sum(x). Jacobian norm = scale * sqrt(numel)."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x.flatten(1).sum(dim=1)


def _fake_batch(shape=(4, 3, 8, 8)) -> tuple[Tensor, Tensor]:
    return torch.randn(*shape), torch.randn(*shape)


def test_output_is_non_negative_scalar() -> None:
    critic = _UnitJacobianCritic()
    real, fake = _fake_batch()
    gp = gradient_penalty(critic, real, fake)
    assert gp.dim() == 0
    assert gp.item() >= 0.0


def test_zero_when_jacobian_norm_equals_one() -> None:
    """Pick a scale so that ||grad|| == 1 -> penalty == 0."""
    numel = 3 * 8 * 8  # per-sample numel for the (4, 3, 8, 8) batch
    scale = 1.0 / (numel**0.5)
    critic = _ScaledCritic(scale=scale)
    real, fake = _fake_batch()
    gp = gradient_penalty(critic, real, fake)
    assert gp.item() == pytest.approx(0.0, abs=1e-6)


def test_nonzero_when_jacobian_norm_differs_from_one() -> None:
    critic = _UnitJacobianCritic()  # ||grad|| = sqrt(192) >> 1
    real, fake = _fake_batch()
    gp = gradient_penalty(critic, real, fake)
    expected = (192**0.5 - 1.0) ** 2
    assert gp.item() == pytest.approx(expected, rel=1e-5)


def test_gradients_reach_critic_parameters() -> None:
    """Without create_graph=True, this test would silently get zero grads."""
    critic = _ScaledCritic(scale=1.0)
    real, fake = _fake_batch()
    gp = gradient_penalty(critic, real, fake)
    gp.backward()
    assert critic.scale.grad is not None
    assert critic.scale.grad.abs().item() > 0


def test_shape_mismatch_raises() -> None:
    critic = _UnitJacobianCritic()
    real = torch.randn(4, 3, 8, 8)
    fake = torch.randn(4, 3, 16, 16)
    with pytest.raises(ValueError, match="must match"):
        gradient_penalty(critic, real, fake)


class _SumOfSquaresCritic(nn.Module):
    """f(x_i) = sum(x_i ** 2). ||grad_x_i f(x_i)|| = 2 * ||x_i||, so the
    penalty depends on the interpolated point and therefore on eps."""

    def forward(self, x: Tensor) -> Tensor:
        return (x.flatten(1) ** 2).sum(dim=1)


def test_per_sample_eps_is_actually_random() -> None:
    """Two consecutive calls with the same data should give different GP
    values, proving that eps is sampled fresh per call. We pick a critic
    whose Jacobian norm depends on the input (sum-of-squares); a critic
    with a constant Jacobian like _UnitJacobianCritic would mask the
    randomness."""
    torch.manual_seed(0)
    critic = _SumOfSquaresCritic()
    real, fake = _fake_batch()
    gp1 = gradient_penalty(critic, real, fake).item()
    gp2 = gradient_penalty(critic, real, fake).item()
    assert gp1 != gp2
