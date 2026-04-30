"""Frechet Inception Distance (FID) computation.

torchmetrics' ``FrechetInceptionDistance`` expects ``uint8`` tensors in
``[0, 255]``. Our pipeline carries ``[-1, 1]`` floats, so we denormalize
and quantize before feeding it.

FID is expensive (Inception forward + covariance over Gaussians); we only
call it every ``training.fid_every_n_epochs`` epochs and on a bounded
sample budget. It is the metric the W&B Sweep optimizes against.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def _to_uint8(images: Tensor) -> Tensor:
    """Convert ``[-1, 1]`` floats to ``uint8`` ``[0, 255]``."""
    images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)
    return (images * 255.0).to(torch.uint8)


@torch.no_grad()
def compute_fid(
    *,
    generator: nn.Module,
    dataloader: DataLoader,
    latent_dim: int,
    num_samples: int,
    device: torch.device,
    feature_dim: int = 2048,
) -> float:
    """Compute FID between ``num_samples`` real and fake images.

    Real images are drawn from ``dataloader`` (looped if needed). Fake
    images come from the generator with ``z ~ N(0, I)``.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=feature_dim, normalize=False).to(device)

    was_training = generator.training
    generator.eval()
    try:
        seen_real = 0
        loop = iter(dataloader)
        while seen_real < num_samples:
            try:
                real = next(loop)
            except StopIteration:
                loop = iter(dataloader)
                real = next(loop)
            real = real.to(device, non_blocking=True)
            take = min(real.size(0), num_samples - seen_real)
            fid.update(_to_uint8(real[:take]), real=True)
            seen_real += take

        seen_fake = 0
        batch_size = max(min(num_samples, dataloader.batch_size or 64), 1)
        while seen_fake < num_samples:
            take = min(batch_size, num_samples - seen_fake)
            z = torch.randn(take, latent_dim, device=device)
            fake = generator(z)
            fid.update(_to_uint8(fake), real=False)
            seen_fake += take

        return float(fid.compute().item())
    finally:
        generator.train(was_training)
