"""DCGAN-style Generator for 64x64 RGB images.

Maps a latent vector ``z ~ N(0, I)`` of dimension ``latent_dim`` to a 3x64x64
image in roughly ``[-1, 1]`` (final activation is ``tanh``, matching the
``Normalize(mean=0.5, std=0.5)`` used on the data side).

The channel schedule is ``base_channels * {8, 4, 2, 1}`` followed by
``output_channels``. With ``base_channels=64`` that gives 512 -> 256 -> 128
-> 64 -> 3.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 64,
        output_channels: int = 3,
        output_size: int = 64,
    ) -> None:
        super().__init__()
        if output_size != 64:
            raise NotImplementedError("Generator currently targets 64x64 output only.")
        self.latent_dim = latent_dim

        c8, c4, c2, c1 = (
            base_channels * 8,
            base_channels * 4,
            base_channels * 2,
            base_channels,
        )

        # 1x1 -> 4x4
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, c8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c8),
            nn.ReLU(inplace=True),
        )
        # 4x4 -> 8x8
        self.up1 = self._block(c8, c4)
        # 8x8 -> 16x16
        self.up2 = self._block(c4, c2)
        # 16x16 -> 32x32
        self.up3 = self._block(c2, c1)
        # 32x32 -> 64x64, no BN on output layer
        self.to_image = nn.Sequential(
            nn.ConvTranspose2d(c1, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: Tensor) -> Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)  # (B, latent_dim, 1, 1)
        x = self.proj(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.to_image(x)

    @torch.no_grad()
    def sample_latent(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)
