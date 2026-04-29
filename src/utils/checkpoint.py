"""Checkpoint save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    generator: nn.Module,
    critic: nn.Module,
    opt_g: Optimizer,
    opt_c: Optimizer,
    extra: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "critic": critic.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_c": opt_c.state_dict(),
    }
    if extra:
        state.update(extra)
    torch.save(state, out)
    return out


def load_checkpoint(
    path: str | Path,
    *,
    generator: nn.Module,
    critic: nn.Module,
    opt_g: Optimizer | None = None,
    opt_c: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    state = torch.load(str(path), map_location=map_location)
    generator.load_state_dict(state["generator"])
    critic.load_state_dict(state["critic"])
    if opt_g is not None and "opt_g" in state:
        opt_g.load_state_dict(state["opt_g"])
    if opt_c is not None and "opt_c" in state:
        opt_c.load_state_dict(state["opt_c"])
    return state
