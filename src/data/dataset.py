"""Folder-of-PNGs dataset for the preprocessed WikiArt subset."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class WikiArtAbstractDataset(Dataset[Tensor]):
    """Reads ``*.png`` files in ``root`` produced by ``scripts/download_data.py``."""

    def __init__(self, root: str | Path, transform: Callable[[Image.Image], Tensor]) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Data root {self.root} does not exist. Run scripts/download_data.py first."
            )
        self.paths = sorted(self.root.glob("*.png"))
        if not self.paths:
            raise RuntimeError(f"No PNG files found under {self.root}.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
            return self.transform(img)
