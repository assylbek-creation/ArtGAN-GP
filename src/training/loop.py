"""WGAN-GP training loop.

One epoch = one full pass over the dataloader. For each round we run
``n_critic`` critic updates followed by one generator update. This 5:1
schedule is the WGAN-GP default and matters for stability — making the
critic too weak destroys the Wasserstein estimate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.training.gradient_penalty import gradient_penalty


@dataclass
class StepStats:
    loss_critic: float
    loss_generator: float | None
    wasserstein_estimate: float
    gradient_penalty: float
    critic_grad_norm: float


def _generate_fake(
    generator: nn.Module, batch_size: int, latent_dim: int, device: torch.device
) -> Tensor:
    z = torch.randn(batch_size, latent_dim, device=device)
    return generator(z)


def critic_step(
    *,
    generator: nn.Module,
    critic: nn.Module,
    opt_c: Optimizer,
    real: Tensor,
    latent_dim: int,
    lambda_gp: float,
    grad_clip: float | None,
) -> tuple[float, float, float, float]:
    device = real.device
    batch_size = real.size(0)

    with torch.no_grad():
        fake = _generate_fake(generator, batch_size, latent_dim, device)

    score_real = critic(real)
    score_fake = critic(fake)
    gp = gradient_penalty(critic, real, fake)

    wasserstein = score_real.mean() - score_fake.mean()
    loss = -wasserstein + lambda_gp * gp

    opt_c.zero_grad(set_to_none=True)
    loss.backward()

    if grad_clip is not None:
        grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip)
    else:
        grad_norm = _global_grad_norm(critic.parameters())

    opt_c.step()

    return loss.item(), wasserstein.item(), gp.item(), float(grad_norm)


def generator_step(
    *,
    generator: nn.Module,
    critic: nn.Module,
    opt_g: Optimizer,
    batch_size: int,
    latent_dim: int,
    device: torch.device,
) -> float:
    fake = _generate_fake(generator, batch_size, latent_dim, device)
    loss = -critic(fake).mean()

    opt_g.zero_grad(set_to_none=True)
    loss.backward()
    opt_g.step()
    return loss.item()


def _global_grad_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.norm(torch.stack(norms), 2)


def train_one_epoch(
    *,
    generator: nn.Module,
    critic: nn.Module,
    opt_g: Optimizer,
    opt_c: Optimizer,
    dataloader: DataLoader,
    latent_dim: int,
    n_critic: int,
    lambda_gp: float,
    grad_clip_critic: float | None,
    device: torch.device,
    on_step: Callable[[StepStats], None] | None = None,
) -> dict[str, float]:
    generator.train()
    critic.train()

    sum_loss_c = sum_wass = sum_gp = sum_loss_g = 0.0
    n_c_steps = 0
    n_g_steps = 0
    last_loss_g: float | None = None
    data_iter = iter(dataloader)

    while True:
        last_real: Tensor | None = None
        last_loss_c = last_wass = last_gp = last_grad_c = 0.0
        completed_critic = 0

        for _ in range(n_critic):
            try:
                real = next(data_iter)
            except StopIteration:
                real = None
                break

            real = real.to(device, non_blocking=True)
            loss_c, wass, gp, grad_c = critic_step(
                generator=generator,
                critic=critic,
                opt_c=opt_c,
                real=real,
                latent_dim=latent_dim,
                lambda_gp=lambda_gp,
                grad_clip=grad_clip_critic,
            )
            sum_loss_c += loss_c
            sum_wass += wass
            sum_gp += gp
            n_c_steps += 1
            completed_critic += 1
            last_real = real
            last_loss_c, last_wass, last_gp, last_grad_c = loss_c, wass, gp, grad_c

            if on_step is not None:
                on_step(
                    StepStats(
                        loss_critic=loss_c,
                        loss_generator=None,
                        wasserstein_estimate=wass,
                        gradient_penalty=gp,
                        critic_grad_norm=grad_c,
                    )
                )

        if completed_critic < n_critic or last_real is None:
            break

        loss_g = generator_step(
            generator=generator,
            critic=critic,
            opt_g=opt_g,
            batch_size=last_real.size(0),
            latent_dim=latent_dim,
            device=device,
        )
        sum_loss_g += loss_g
        n_g_steps += 1
        last_loss_g = loss_g

        if on_step is not None:
            on_step(
                StepStats(
                    loss_critic=last_loss_c,
                    loss_generator=loss_g,
                    wasserstein_estimate=last_wass,
                    gradient_penalty=last_gp,
                    critic_grad_norm=last_grad_c,
                )
            )

    n_c = max(n_c_steps, 1)
    return {
        "loss_critic": sum_loss_c / n_c,
        "loss_generator": sum_loss_g / n_g_steps if n_g_steps else float("nan"),
        "wasserstein_estimate": sum_wass / n_c,
        "gradient_penalty": sum_gp / n_c,
        "last_loss_generator": last_loss_g if last_loss_g is not None else float("nan"),
    }
