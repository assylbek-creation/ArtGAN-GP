"""W&B Sweep launcher (Bayesian search).

Registers the sweep defined in ``src/config/sweep.yaml`` (or joins an
existing one with ``--sweep-id``) and runs an agent that, per trial,
maps the sweep parameters into Hydra overrides and calls
``src.training.train.run`` directly. This keeps everything in-process
so the current Python environment, imports, and CWD are reused.

Usage::

    python -m scripts.run_sweep --count 20
    python -m scripts.run_sweep --sweep-id <existing_id> --count 5
    python -m scripts.run_sweep --project artgan-gp --entity my-team
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "src" / "config"
SWEEP_FILE = CONFIG_DIR / "sweep.yaml"


def load_sweep_config(path: Path = SWEEP_FILE) -> dict[str, Any]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Sweep config at {path} must be a mapping.")
    # 'program' is a CLI-mode artifact; we run inline via wandb.agent(function=...)
    cfg.pop("program", None)
    return cfg


def overrides_from_wandb_config(wandb_config: dict[str, Any]) -> list[str]:
    """Flatten ``wandb.config`` into Hydra dotted-key overrides.

    The sweep yaml uses dotted keys like ``training.lr_generator``; Hydra
    accepts the same syntax for overrides, so this is a near-passthrough.
    """
    return [f"{k}={v}" for k, v in wandb_config.items()]


def build_cfg(extra_overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="baseline", overrides=extra_overrides)


def sweep_iteration() -> None:
    """Single trial body, called once per agent step."""
    import wandb

    from src.training.train import run

    if wandb.run is None:
        # Belt-and-braces; the agent normally calls wandb.init() for us.
        wandb.init()

    overrides = overrides_from_wandb_config(dict(wandb.run.config))
    cfg = build_cfg(overrides)
    try:
        run(cfg)
    finally:
        # Logger.finish() already called inside run; but if run aborts before
        # building the logger, make sure the W&B run is closed.
        if wandb.run is not None:
            wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a W&B sweep for ArtGAN-GP.")
    parser.add_argument("--count", type=int, default=20, help="trials this agent will run")
    parser.add_argument("--sweep-id", type=str, default=None, help="join an existing sweep")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)
    args = parser.parse_args()

    import wandb

    if args.sweep_id is None:
        sweep_cfg = load_sweep_config()
        sweep_id = wandb.sweep(sweep_cfg, project=args.project, entity=args.entity)
        print(f"Created sweep {sweep_id} (project={args.project}).")
    else:
        sweep_id = args.sweep_id
        print(f"Joining existing sweep {sweep_id}.")

    wandb.agent(sweep_id, function=sweep_iteration, count=args.count)


if __name__ == "__main__":
    main()
