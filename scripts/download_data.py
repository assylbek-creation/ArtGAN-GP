"""Download a subset of WikiArt and pre-process to fixed-size square PNGs.

Streams ``huggan/wikiart`` from Hugging Face, keeps only the genres listed in the
data config, and writes ``{image_size}x{image_size}`` RGB PNGs into ``data_root``.

Usage::

    python -m scripts.download_data
    python -m scripts.download_data data.image_size=64 data.data_root=data/wikiart_abstract
"""

from __future__ import annotations

from pathlib import Path

import hydra
from datasets import load_dataset, load_dataset_builder
from omegaconf import DictConfig
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm


def _resolve_genre_ids(builder_genre_names: list[str], wanted: list[str]) -> set[int]:
    """Map human-readable genre names to integer label ids, tolerating spaces vs underscores."""
    normalized = {name.lower().replace(" ", "_"): idx for idx, name in enumerate(builder_genre_names)}
    ids: set[int] = set()
    missing: list[str] = []
    for name in wanted:
        key = name.lower().replace(" ", "_")
        if key in normalized:
            ids.add(normalized[key])
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Genres not found in dataset: {missing}. Available: {builder_genre_names}"
        )
    return ids


def _square_resize(image: Image.Image, size: int) -> Image.Image:
    """Resize so the shortest side == size, then center-crop to size x size."""
    image = image.convert("RGB")
    image = TF.resize(image, size, antialias=True)
    return TF.center_crop(image, [size, size])


@hydra.main(version_base=None, config_path="../src/config", config_name="baseline")
def main(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    out_dir = Path(data_cfg.data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = load_dataset_builder(data_cfg.hf_dataset)
    genre_feature = builder.info.features["genre"]
    genre_ids = _resolve_genre_ids(genre_feature.names, list(data_cfg.genres))
    print(f"Filtering to genre ids {sorted(genre_ids)} -> {list(data_cfg.genres)}")

    stream = load_dataset(data_cfg.hf_dataset, split="train", streaming=True)

    saved = 0
    for row in tqdm(stream, desc="streaming wikiart"):
        if row["genre"] not in genre_ids:
            continue
        try:
            img = _square_resize(row["image"], data_cfg.image_size)
        except (OSError, ValueError) as exc:  # corrupt or non-RGB-convertible images
            print(f"skip image (idx {saved}): {exc}")
            continue
        img.save(out_dir / f"{saved:06d}.png", format="PNG")
        saved += 1

    print(f"Saved {saved} images to {out_dir}")


if __name__ == "__main__":
    main()
