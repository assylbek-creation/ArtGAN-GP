"""Download a subset of WikiArt and pre-process to fixed-size square PNGs.

Streams ``huggan/wikiart`` from Hugging Face, keeps only the rows whose
``data.filter_field`` value matches one of ``data.genres``, and writes
``{image_size}x{image_size}`` RGB PNGs into ``data_root``. With
``data.upload_artifact=true`` the resulting directory is also published
as a W&B Artifact so sweep workers can pull an identical snapshot
instead of re-downloading from HF.

Note on ``filter_field``: ``huggan/wikiart`` exposes both ``genre``
(coarse, 11 values like ``abstract_painting``, ``landscape``) and
``style`` (fine, 27 values including ``Abstract_Expressionism`` and
``Color_Field_Painting``). Our project filters on **style** by default,
because that is the granularity that defines our subset.

Usage::

    python -m scripts.download_data
    python -m scripts.download_data data.upload_artifact=true
    python -m scripts.download_data data.max_images=200  # quick smoke run
    python -m scripts.download_data data.filter_field=genre data.genres=[abstract_painting]
"""

from __future__ import annotations

from pathlib import Path

import hydra
from datasets import load_dataset, load_dataset_builder
from omegaconf import DictConfig
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.utils.logger import build_logger


def _resolve_label_ids(label_names: list[str], wanted: list[str], field: str) -> set[int]:
    """Map human-readable label names to integer ids, tolerating spaces vs underscores."""
    normalized = {name.lower().replace(" ", "_"): idx for idx, name in enumerate(label_names)}
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
            f"Values for field {field!r} not found in dataset: {missing}. "
            f"Available {field}s: {label_names}"
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

    field = data_cfg.get("filter_field", "style")
    builder = load_dataset_builder(data_cfg.hf_dataset)
    if field not in builder.info.features:
        raise ValueError(
            f"Dataset {data_cfg.hf_dataset!r} has no feature {field!r}. "
            f"Available features: {list(builder.info.features)}"
        )
    label_feature = builder.info.features[field]
    label_ids = _resolve_label_ids(label_feature.names, list(data_cfg.genres), field=field)
    print(f"Filtering by {field}={sorted(label_ids)} -> {list(data_cfg.genres)}")

    stream = load_dataset(data_cfg.hf_dataset, split="train", streaming=True)
    cap = data_cfg.get("max_images")

    saved = 0
    for row in tqdm(stream, desc="streaming wikiart"):
        if cap is not None and saved >= cap:
            break
        if row[field] not in label_ids:
            continue
        try:
            img = _square_resize(row["image"], data_cfg.image_size)
        except (OSError, ValueError) as exc:  # corrupt or non-RGB-convertible images
            print(f"skip image (idx {saved}): {exc}")
            continue
        img.save(out_dir / f"{saved:06d}.png", format="PNG")
        saved += 1

    print(f"Saved {saved} images to {out_dir}")

    if data_cfg.get("upload_artifact"):
        logger = build_logger(cfg)
        try:
            logger.log_artifact(
                out_dir,
                name=data_cfg.artifact_name,
                artifact_type="dataset",
                metadata={
                    "image_size": data_cfg.image_size,
                    "filter_field": field,
                    "labels": list(data_cfg.genres),
                    "num_images": saved,
                    "source": data_cfg.hf_dataset,
                },
                aliases=["latest"],
            )
            print(f"Uploaded artifact '{data_cfg.artifact_name}' with {saved} images.")
        finally:
            logger.finish()


if __name__ == "__main__":
    main()
