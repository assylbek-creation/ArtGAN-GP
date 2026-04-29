"""Thin W&B wrapper.

A no-op fallback when ``mode='disabled'`` lets unit tests and quick local
runs avoid the W&B dependency at runtime. The full MLOps integration
(artifacts, sweeps) lives on top of this in Phase 4.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from omegaconf import DictConfig, OmegaConf


class Logger(Protocol):
    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None: ...
    def log_images(self, key: str, images: Any, step: int | None = None) -> None: ...
    def finish(self) -> None: ...


class _NoopLogger:
    def __init__(self) -> None:
        self._buffer: list[Mapping[str, Any]] = []

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        suffix = "" if step is None else f" [step={step}]"
        scalar_summary = " ".join(
            f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(f"log{suffix}: {scalar_summary}")

    def log_images(self, key: str, images: Any, step: int | None = None) -> None:
        del images
        print(f"log_images: {key}{'' if step is None else f' [step={step}]'}")

    def finish(self) -> None:
        pass


class _WandbLogger:
    def __init__(self, cfg: DictConfig) -> None:
        import wandb

        self._wandb = wandb
        wandb_cfg = cfg.wandb
        wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            mode=wandb_cfg.mode,
            tags=list(wandb_cfg.tags) if wandb_cfg.tags else None,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        self._wandb.log(dict(metrics), step=step)

    def log_images(self, key: str, images: Any, step: int | None = None) -> None:
        self._wandb.log({key: self._wandb.Image(images)}, step=step)

    def finish(self) -> None:
        self._wandb.finish()


def build_logger(cfg: DictConfig) -> Logger:
    mode = cfg.wandb.mode
    if mode == "disabled":
        return _NoopLogger()
    return _WandbLogger(cfg)
