"""End-to-end smoke test for the WGAN-GP training loop.

Runs one epoch on tiny synthetic data and a stripped-down 16x16 critic /
generator pair. It does **not** check that the model learns anything — it
checks that the loop runs without exceptions, produces finite stats, and
updates parameters.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.training.loop import train_one_epoch


class _TinyGenerator(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 3 * 16 * 16),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() > 2:
            z = z.flatten(1)
        return self.net(z).view(-1, 3, 16, 16)


class _TinyCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def test_loop_runs_one_epoch_and_updates_params() -> None:
    torch.manual_seed(0)
    latent_dim = 8

    g = _TinyGenerator(latent_dim)
    c = _TinyCritic()

    fake_data = torch.randn(24, 3, 16, 16)
    loader = DataLoader(TensorDataset(fake_data), batch_size=4, shuffle=False)

    # Wrap loader so the dataloader yields tensors (not 1-tuples).
    class _Unwrap:
        def __init__(self, dl):
            self.dl = dl

        def __iter__(self):
            for batch in self.dl:
                yield batch[0]

        def __len__(self):
            return len(self.dl)

    opt_g = Adam(g.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_c = Adam(c.parameters(), lr=1e-4, betas=(0.0, 0.9))

    g_params_before = [p.detach().clone() for p in g.parameters()]
    c_params_before = [p.detach().clone() for p in c.parameters()]

    stats = train_one_epoch(
        generator=g,
        critic=c,
        opt_g=opt_g,
        opt_c=opt_c,
        dataloader=_Unwrap(loader),
        latent_dim=latent_dim,
        n_critic=3,
        lambda_gp=10.0,
        grad_clip_critic=None,
        device=torch.device("cpu"),
    )

    for k in ["loss_critic", "loss_generator", "wasserstein_estimate", "gradient_penalty"]:
        assert k in stats
        assert torch.isfinite(torch.tensor(stats[k])), f"{k}={stats[k]}"

    assert any(
        not torch.allclose(a, b) for a, b in zip(g_params_before, list(g.parameters()))
    ), "generator parameters did not move"
    assert any(
        not torch.allclose(a, b) for a, b in zip(c_params_before, list(c.parameters()))
    ), "critic parameters did not move"
