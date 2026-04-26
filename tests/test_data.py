"""Smoke tests for the dataset / transforms / dataloader stack.

Generates a handful of fake PNGs in a tmpdir so the test does not depend on the
real WikiArt download. Verifies tensor shape, dtype, and value range — not pixel
content.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.data.dataset import WikiArtAbstractDataset
from src.data.loader import build_dataloader
from src.data.transforms import build_transform


def _make_fake_dataset(root: Path, n: int, size: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (size, size), color=(i * 7 % 256, i * 13 % 256, i * 19 % 256)).save(
            root / f"{i:06d}.png"
        )


def test_transform_outputs_normalized_tensor() -> None:
    transform = build_transform(64, mean=[0.5] * 3, std=[0.5] * 3, train=False)
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    out = transform(img)
    assert out.shape == (3, 64, 64)
    assert out.dtype == torch.float32
    # white -> 1.0 in [0,1] -> (1 - 0.5) / 0.5 == 1.0 after normalize
    assert torch.isclose(out.max(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(out.min(), torch.tensor(1.0), atol=1e-5)


def test_dataset_reads_images(tmp_path: Path) -> None:
    _make_fake_dataset(tmp_path, n=4, size=64)
    transform = build_transform(64, mean=[0.5] * 3, std=[0.5] * 3, train=False)
    ds = WikiArtAbstractDataset(tmp_path, transform=transform)
    assert len(ds) == 4
    sample = ds[0]
    assert sample.shape == (3, 64, 64)
    assert sample.min() >= -1.0 - 1e-5
    assert sample.max() <= 1.0 + 1e-5


def test_dataset_raises_when_empty(tmp_path: Path) -> None:
    transform = build_transform(64, mean=[0.5] * 3, std=[0.5] * 3, train=False)
    with pytest.raises(RuntimeError):
        WikiArtAbstractDataset(tmp_path, transform=transform)


def test_dataloader_factory_yields_correct_batch(tmp_path: Path) -> None:
    _make_fake_dataset(tmp_path, n=8, size=64)
    cfg = OmegaConf.create(
        {
            "data": {
                "data_root": str(tmp_path),
                "image_size": 64,
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "horizontal_flip": True,
                "normalize_mean": [0.5, 0.5, 0.5],
                "normalize_std": [0.5, 0.5, 0.5],
            }
        }
    )
    loader = build_dataloader(cfg, train=True)
    batch = next(iter(loader))
    assert batch.shape == (4, 3, 64, 64)
    assert batch.dtype == torch.float32
