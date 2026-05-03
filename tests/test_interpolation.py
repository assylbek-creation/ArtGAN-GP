"""Tests for latent interpolation helpers."""

from __future__ import annotations

import pytest
import torch

from src.utils.interpolation import lerp_path, slerp_grid, slerp_path


def test_lerp_endpoints_match_inputs() -> None:
    z1 = torch.randn(128)
    z2 = torch.randn(128)
    path = lerp_path(z1, z2, n_steps=10)
    assert path.shape == (10, 128)
    assert torch.allclose(path[0], z1, atol=1e-6)
    assert torch.allclose(path[-1], z2, atol=1e-6)


def test_lerp_midpoint_is_average() -> None:
    z1 = torch.zeros(8)
    z2 = torch.ones(8) * 4.0
    path = lerp_path(z1, z2, n_steps=3)
    assert torch.allclose(path[1], torch.full((8,), 2.0))


def test_slerp_endpoints_match_inputs() -> None:
    torch.manual_seed(0)
    z1 = torch.randn(128)
    z2 = torch.randn(128)
    path = slerp_path(z1, z2, n_steps=10)
    assert path.shape == (10, 128)
    assert torch.allclose(path[0], z1, atol=1e-4)
    assert torch.allclose(path[-1], z2, atol=1e-4)


def test_slerp_preserves_norm_for_unit_vectors() -> None:
    """For two unit-norm vectors slerp keeps the norm at 1 along the path."""
    torch.manual_seed(0)
    z1 = torch.randn(128)
    z1 = z1 / z1.norm()
    z2 = torch.randn(128)
    z2 = z2 / z2.norm()
    path = slerp_path(z1, z2, n_steps=11)
    norms = path.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_slerp_falls_back_to_lerp_when_colinear() -> None:
    z1 = torch.ones(8)
    z2 = torch.ones(8) * 2.0  # same direction, different magnitude -> sin(omega)=0
    s = slerp_path(z1, z2, n_steps=5)
    l = lerp_path(z1, z2, n_steps=5)
    assert torch.allclose(s, l, atol=1e-5)


def test_slerp_rejects_batch_inputs() -> None:
    with pytest.raises(ValueError, match="1-D latents"):
        slerp_path(torch.randn(2, 128), torch.randn(2, 128), n_steps=10)


def test_paths_reject_n_steps_below_two() -> None:
    z = torch.randn(8)
    with pytest.raises(ValueError, match="n_steps"):
        lerp_path(z, z, n_steps=1)
    with pytest.raises(ValueError, match="n_steps"):
        slerp_path(z, z, n_steps=1)


def test_paths_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        lerp_path(torch.randn(8), torch.randn(16), n_steps=4)
    with pytest.raises(ValueError, match="shape mismatch"):
        slerp_path(torch.randn(8), torch.randn(16), n_steps=4)


def test_slerp_grid_shape() -> None:
    torch.manual_seed(0)
    corners = tuple(torch.randn(64) for _ in range(4))
    grid = slerp_grid(corners, n_steps=5)  # type: ignore[arg-type]
    assert grid.shape == (25, 64)
