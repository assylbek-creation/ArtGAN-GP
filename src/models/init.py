"""DCGAN-style weight initialization.

Conv and ConvTranspose weights ~ N(0, 0.02), BN weights ~ N(1, 0.02), biases 0.
LayerNorm/InstanceNorm affine parameters keep their PyTorch defaults.
"""

from __future__ import annotations

from torch import nn


def dcgan_weights_init(module: nn.Module) -> None:
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias.data)
    elif "BatchNorm" in name and module.weight is not None:
        nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias.data)
