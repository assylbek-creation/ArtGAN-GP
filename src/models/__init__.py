"""Model factory wired to the Hydra ``model`` config group."""

from __future__ import annotations

from omegaconf import DictConfig

from src.models.critic import Critic
from src.models.generator import Generator
from src.models.init import dcgan_weights_init

__all__ = ["Critic", "Generator", "build_models", "dcgan_weights_init"]


def build_models(cfg: DictConfig) -> tuple[Generator, Critic]:
    model_cfg = cfg.model
    generator = Generator(
        latent_dim=model_cfg.latent_dim,
        base_channels=model_cfg.generator.base_channels,
        output_channels=model_cfg.generator.output_channels,
        output_size=model_cfg.generator.output_size,
    )
    critic = Critic(
        input_channels=model_cfg.critic.input_channels,
        input_size=model_cfg.critic.input_size,
        base_channels=model_cfg.critic.base_channels,
        norm=model_cfg.critic.norm,
    )
    generator.apply(dcgan_weights_init)
    critic.apply(dcgan_weights_init)
    return generator, critic
