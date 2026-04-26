"""DataLoader factory wired to the Hydra ``data`` config group."""

from __future__ import annotations

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.dataset import WikiArtAbstractDataset
from src.data.transforms import build_transform


def build_dataloader(cfg: DictConfig, train: bool = True) -> DataLoader:
    data_cfg = cfg.data
    transform = build_transform(
        image_size=data_cfg.image_size,
        mean=data_cfg.normalize_mean,
        std=data_cfg.normalize_std,
        train=train,
        horizontal_flip=bool(data_cfg.get("horizontal_flip", True)),
    )
    dataset = WikiArtAbstractDataset(root=data_cfg.data_root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=train,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=train,
        persistent_workers=data_cfg.num_workers > 0,
    )
