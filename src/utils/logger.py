"""Thin W&B wrapper.

A no-op fallback when ``mode='disabled'`` lets unit tests and quick local
runs avoid the W&B dependency at runtime. Beyond ``log``/``log_images``,
this module exposes ``log_artifact`` and ``download_artifact`` so the
training script and the data pipeline can persist datasets, checkpoints,
and the resolved config through W&B Artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from omegaconf import DictConfig, OmegaConf


class Logger(Protocol):
    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None: ...
    def log_images(self, key: str, images: Any, step: int | None = None) -> None: ...
    def log_artifact(
        self,
        path: str | Path,
        *,
        name: str,
        artifact_type: str,
        metadata: Mapping[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None: ...
    def download_artifact(self, name: str, root: str | Path | None = None) -> Path: ...
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

    def log_artifact(
        self,
        path: str | Path,
        *,
        name: str,
        artifact_type: str,
        metadata: Mapping[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        del metadata, aliases
        print(f"log_artifact: name={name} type={artifact_type} path={path} (noop)")

    def download_artifact(self, name: str, root: str | Path | None = None) -> Path:
        raise RuntimeError(
            "download_artifact is unavailable in disabled mode; provide local files instead."
        )

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

    def log_artifact(
        self,
        path: str | Path,
        *,
        name: str,
        artifact_type: str,
        metadata: Mapping[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        target = Path(path)
        artifact = self._wandb.Artifact(
            name=name, type=artifact_type, metadata=dict(metadata) if metadata else None
        )
        if target.is_dir():
            artifact.add_dir(str(target))
        else:
            artifact.add_file(str(target))
        self._wandb.log_artifact(artifact, aliases=aliases or None)

    def download_artifact(self, name: str, root: str | Path | None = None) -> Path:
        run = self._wandb.run
        if run is None:
            raise RuntimeError("W&B run not initialized; cannot download artifact.")
        artifact = run.use_artifact(name)
        return Path(artifact.download(root=str(root) if root else None))

    def finish(self) -> None:
        self._wandb.finish()


def build_logger(cfg: DictConfig) -> Logger:
    mode = cfg.wandb.mode
    if mode == "disabled":
        return _NoopLogger()
    return _WandbLogger(cfg)
