"""Unit tests for Generator and Critic.

These verify load-bearing WGAN-GP invariants:

- Generator output shape and ``[-1, 1]`` range from ``tanh``.
- Critic output is a per-sample scalar (no sigmoid, no extra dims).
- Critic contains **no BatchNorm** layers — they are silently incompatible
  with the gradient penalty and would only show up as bad training.
- Both networks accept gradients end-to-end.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from src.models import build_models
from src.models.critic import Critic
from src.models.generator import Generator


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {
            "model": {
                "latent_dim": 128,
                "generator": {
                    "base_channels": 64,
                    "output_channels": 3,
                    "output_size": 64,
                },
                "critic": {
                    "base_channels": 64,
                    "input_channels": 3,
                    "input_size": 64,
                    "norm": "layer",
                },
            }
        }
    )


def test_generator_output_shape_and_range() -> None:
    g = Generator(latent_dim=128, base_channels=64)
    z = torch.randn(2, 128)
    out = g(z)
    assert out.shape == (2, 3, 64, 64)
    assert out.min() >= -1.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5


def test_generator_accepts_4d_latent() -> None:
    g = Generator(latent_dim=128, base_channels=64)
    z = torch.randn(2, 128, 1, 1)
    assert g(z).shape == (2, 3, 64, 64)


def test_critic_output_is_per_sample_scalar() -> None:
    c = Critic()
    x = torch.randn(4, 3, 64, 64)
    out = c(x)
    assert out.shape == (4,)
    assert out.dtype == torch.float32


def test_critic_has_no_batchnorm() -> None:
    """Load-bearing WGAN-GP invariant — see src/models/critic.py docstring."""
    c = Critic()
    bn_layers = [m for m in c.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert bn_layers == [], f"Critic must not contain BatchNorm; found {bn_layers}"


def test_critic_rejects_batch_norm_kind() -> None:
    with pytest.raises(ValueError, match="BatchNorm is incompatible"):
        Critic(norm="batch")


def test_critic_supports_instance_norm() -> None:
    c = Critic(norm="instance")
    out = c(torch.randn(2, 3, 64, 64))
    assert out.shape == (2,)


def test_gradients_flow_through_generator_and_critic() -> None:
    g = Generator()
    c = Critic()
    z = torch.randn(2, 128, requires_grad=False)
    fake = g(z)
    score = c(fake).mean()
    score.backward()

    g_grads = [p.grad for p in g.parameters() if p.requires_grad]
    c_grads = [p.grad for p in c.parameters() if p.requires_grad]
    assert all(grad is not None for grad in g_grads)
    assert all(grad is not None for grad in c_grads)
    assert any(grad.abs().sum() > 0 for grad in g_grads)
    assert any(grad.abs().sum() > 0 for grad in c_grads)


def test_build_models_factory(cfg) -> None:
    g, c = build_models(cfg)
    assert isinstance(g, Generator)
    assert isinstance(c, Critic)
    assert g(torch.randn(2, 128)).shape == (2, 3, 64, 64)
    assert c(torch.randn(2, 3, 64, 64)).shape == (2,)


def test_sample_latent_helper() -> None:
    g = Generator(latent_dim=128)
    z = g.sample_latent(8)
    assert z.shape == (8, 128)
