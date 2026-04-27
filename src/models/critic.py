"""WGAN-GP Critic (NOT a Discriminator) for 64x64 RGB images.

Outputs a real-valued scalar per sample. There is no sigmoid: the critic
estimates the Wasserstein-1 distance, not a probability.

Two non-obvious WGAN-GP requirements enforced here:

1. **No BatchNorm.** The gradient penalty is computed per-sample on
   interpolated inputs; BN couples samples in a batch and breaks the
   penalty. We use LayerNorm or InstanceNorm instead.
2. **No bias when followed by a normalization layer.** Standard practice;
   bias is absorbed by the affine parameters of the norm.
"""

from __future__ import annotations

from torch import Tensor, nn


def _norm_layer(kind: str, channels: int, spatial: int) -> nn.Module:
    kind = kind.lower()
    if kind == "layer":
        # LayerNorm over (C, H, W) — needs explicit shape because conv tensors are 4D.
        return nn.LayerNorm([channels, spatial, spatial])
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "none":
        return nn.Identity()
    if kind == "batch":
        raise ValueError(
            "BatchNorm is incompatible with WGAN-GP gradient penalty. "
            "Use 'layer' or 'instance' instead."
        )
    raise ValueError(f"Unknown norm kind: {kind!r}")


class Critic(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        input_size: int = 64,
        base_channels: int = 64,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        if input_size != 64:
            raise NotImplementedError("Critic currently expects 64x64 input only.")

        c1, c2, c4, c8 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        # 64 -> 32 -> 16 -> 8 -> 4
        self.block1 = self._block(input_channels, c1, spatial=32, norm=norm, first=True)
        self.block2 = self._block(c1, c2, spatial=16, norm=norm)
        self.block3 = self._block(c2, c4, spatial=8, norm=norm)
        self.block4 = self._block(c4, c8, spatial=4, norm=norm)
        # 4x4 -> 1x1, scalar; no sigmoid, no norm
        self.to_score = nn.Conv2d(c8, 1, kernel_size=4, stride=1, padding=0)

    @staticmethod
    def _block(
        in_ch: int, out_ch: int, spatial: int, norm: str, first: bool = False
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=first),
        ]
        if not first:
            layers.append(_norm_layer(norm, out_ch, spatial))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.to_score(x)
        return x.view(x.size(0))
