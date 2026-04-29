"""Training entry point.

Usage::

    python -m src.training.train
    python -m src.training.train training.epochs=10 data.batch_size=32

The Hydra config is rooted at ``src/config/baseline.yaml``. To disable W&B
for a quick local smoke run, pass ``wandb.mode=disabled``.
"""

from __future__ import annotations

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam

from src.data.loader import build_dataloader
from src.models import build_models
from src.training.loop import train_one_epoch
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import build_logger
from src.utils.visualize import generate_grid


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


@hydra.main(version_base=None, config_path="../config", config_name="baseline")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)

    dataloader = build_dataloader(cfg, train=True)
    generator, critic = build_models(cfg)
    generator.to(device)
    critic.to(device)

    train_cfg = cfg.training
    opt_g = Adam(
        generator.parameters(),
        lr=train_cfg.lr_generator,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
    )
    opt_c = Adam(
        critic.parameters(),
        lr=train_cfg.lr_critic,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
    )

    logger = build_logger(cfg)

    fixed_z = torch.randn(train_cfg.fixed_sample_size, cfg.model.latent_dim, device=device)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    sample_dir = Path(cfg.output_dir) / "samples"
    global_step = 0

    def _on_step(stats) -> None:
        nonlocal global_step
        global_step += 1
        if global_step % train_cfg.log_every_n_steps != 0:
            return
        payload = {
            "loss_critic": stats.loss_critic,
            "wasserstein_estimate": stats.wasserstein_estimate,
            "gradient_penalty": stats.gradient_penalty,
            "critic_grad_norm": stats.critic_grad_norm,
        }
        if stats.loss_generator is not None:
            payload["loss_generator"] = stats.loss_generator
        logger.log(payload, step=global_step)

    try:
        for epoch in range(1, train_cfg.epochs + 1):
            epoch_stats = train_one_epoch(
                generator=generator,
                critic=critic,
                opt_g=opt_g,
                opt_c=opt_c,
                dataloader=dataloader,
                latent_dim=cfg.model.latent_dim,
                n_critic=train_cfg.n_critic,
                lambda_gp=train_cfg.lambda_gp,
                grad_clip_critic=train_cfg.grad_clip_critic,
                device=device,
                on_step=_on_step,
            )
            logger.log({f"epoch/{k}": v for k, v in epoch_stats.items()}, step=global_step)
            print(f"epoch {epoch}: {epoch_stats}")

            if epoch % train_cfg.sample_every_n_epochs == 0:
                grid = generate_grid(generator, fixed_z, nrow=8)
                logger.log_images("samples/fixed_z", grid, step=global_step)
                from src.utils.visualize import save_sample_grid

                save_sample_grid(
                    generator(fixed_z).detach().cpu(),
                    sample_dir / f"epoch_{epoch:04d}.png",
                    nrow=8,
                )

            if epoch % train_cfg.checkpoint_every_n_epochs == 0:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.pt",
                    epoch=epoch,
                    generator=generator,
                    critic=critic,
                    opt_g=opt_g,
                    opt_c=opt_c,
                    extra={"config": OmegaConf.to_container(cfg, resolve=True)},
                )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
