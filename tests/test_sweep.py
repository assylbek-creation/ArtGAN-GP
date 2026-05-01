"""Tests for the W&B sweep launcher.

These do not require ``wandb`` itself — they exercise the pure-Python
glue: that the sweep YAML parses to the expected shape, that flattening
``wandb.config`` produces valid Hydra overrides, and that those overrides
actually land in the composed config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_sweep import (
    SWEEP_FILE,
    build_cfg,
    load_sweep_config,
    overrides_from_wandb_config,
)


def test_sweep_yaml_has_required_top_level_keys() -> None:
    cfg = load_sweep_config()
    for key in ("method", "metric", "parameters"):
        assert key in cfg, f"sweep.yaml missing required key: {key}"
    assert cfg["method"] in {"bayes", "grid", "random"}
    assert cfg["metric"]["name"] == "fid"
    assert cfg["metric"]["goal"] == "minimize"


def test_sweep_yaml_program_key_is_stripped() -> None:
    """We run inline via wandb.agent(function=...); a 'program' key would
    confuse the agent into trying to spawn a subprocess instead."""
    cfg = load_sweep_config()
    assert "program" not in cfg


def test_overrides_from_wandb_config_flattens_dotted_keys() -> None:
    wb = {
        "training.lr_generator": 1e-4,
        "training.lr_critic": 2e-4,
        "data.batch_size": 64,
    }
    out = sorted(overrides_from_wandb_config(wb))
    assert out == [
        "data.batch_size=64",
        "training.lr_critic=0.0002",
        "training.lr_generator=0.0001",
    ]


def test_overrides_apply_to_composed_config() -> None:
    overrides = [
        "training.lr_generator=0.0003",
        "training.n_critic=7",
        "data.batch_size=128",
        "wandb.mode=disabled",
    ]
    cfg = build_cfg(overrides)
    assert cfg.training.lr_generator == pytest.approx(0.0003)
    assert cfg.training.n_critic == 7
    assert cfg.data.batch_size == 128
    assert cfg.wandb.mode == "disabled"


def test_sweep_file_path_resolves() -> None:
    assert SWEEP_FILE.is_file(), f"sweep.yaml missing at {SWEEP_FILE}"
    assert SWEEP_FILE.parent.name == "config"
    assert SWEEP_FILE.parent.parent.name == "src"
