"""Image transforms applied at training time.

Pre-processing (resize + center crop) lives in ``scripts/download_data.py``;
this module only handles per-batch augmentation and tensorization, so each
batch read is cheap.
"""

from __future__ import annotations

from collections.abc import Sequence

from torchvision import transforms as T


def build_transform(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    train: bool,
    horizontal_flip: bool = True,
) -> T.Compose:
    steps: list = []
    if train and horizontal_flip:
        steps.append(T.RandomHorizontalFlip(p=0.5))
    steps.append(T.CenterCrop(image_size))  # safety net; preprocessed PNGs already match
    steps.append(T.ToTensor())  # -> [0, 1]
    steps.append(T.Normalize(list(mean), list(std)))  # -> roughly [-1, 1] with mean=std=0.5
    return T.Compose(steps)
