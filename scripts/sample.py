"""Generate samples from a trained checkpoint.

Three modes:

- ``random`` (default): sample N latents from N(0, I) and tile into a grid.
- ``interpolate``: slerp between two seeds and emit a strip of N images.
- ``grid``: bilinear slerp over four corner seeds, output an NxN tile.

Usage::

    python -m scripts.sample --checkpoint checkpoints/epoch_0100.pt
    python -m scripts.sample --checkpoint <path> --mode interpolate \\
        --seed-a 0 --seed-b 1 --n 16 --out outputs/interp.png
    python -m scripts.sample --checkpoint <path> --mode grid \\
        --seeds 0 1 2 3 --n 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.models import build_models
from src.utils.checkpoint import load_checkpoint
from src.utils.interpolation import slerp_grid, slerp_path
from src.utils.visualize import save_sample_grid


def _seeded_z(seed: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(latent_dim, device=device, generator=g)


def _load_generator_from_checkpoint(path: Path, device: torch.device):
    state = torch.load(str(path), map_location=device)
    cfg_dict = state.get("config")
    if cfg_dict is None:
        raise RuntimeError(
            f"Checkpoint {path} has no embedded 'config'. Re-train with the current "
            "save_checkpoint to embed it, or pass --config-path explicitly."
        )
    cfg: DictConfig = OmegaConf.create(cfg_dict)
    generator, _ = build_models(cfg)
    generator.to(device)
    load_checkpoint(path, generator=generator, critic=_DummyCritic(), map_location=device)
    generator.eval()
    return generator, cfg


class _DummyCritic(torch.nn.Module):
    """Stand-in so load_checkpoint can populate critic state without us caring."""

    def state_dict(self, *args, **kwargs):  # noqa: D401
        return {}

    def load_state_dict(self, *args, **kwargs):  # noqa: D401
        return None


@torch.no_grad()
def _generate(generator, latents: torch.Tensor) -> torch.Tensor:
    return generator(latents).detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample images from an ArtGAN-GP checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["random", "interpolate", "grid"], default="random"
    )
    parser.add_argument("--n", type=int, default=64, help="grid side OR number of steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-a", type=int, default=0)
    parser.add_argument("--seed-b", type=int, default=1)
    parser.add_argument(
        "--seeds", type=int, nargs=4, default=(0, 1, 2, 3),
        help="four corner seeds for --mode grid",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/sample.png"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator, cfg = _load_generator_from_checkpoint(args.checkpoint, device)
    latent_dim = cfg.model.latent_dim

    if args.mode == "random":
        torch.manual_seed(args.seed)
        z = torch.randn(args.n, latent_dim, device=device)
        nrow = max(1, int(args.n**0.5))
    elif args.mode == "interpolate":
        z1 = _seeded_z(args.seed_a, latent_dim, device)
        z2 = _seeded_z(args.seed_b, latent_dim, device)
        z = slerp_path(z1, z2, n_steps=args.n)
        nrow = args.n
    else:  # grid
        corners = tuple(_seeded_z(s, latent_dim, device) for s in args.seeds)
        z = slerp_grid(corners, n_steps=args.n)  # type: ignore[arg-type]
        nrow = args.n

    images = _generate(generator, z)
    out_path = save_sample_grid(images, args.out, nrow=nrow)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
