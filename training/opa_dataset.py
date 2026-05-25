from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class OPASample:
    image_path: Path
    mask_path: Path
    label: int


def _split_dir_name(split: str) -> str:
    if split in {"train", "training"}:
        return "train_set"
    if split in {"val", "valid", "validation", "test"}:
        return "test_set"
    return split


def _candidate_paths(root: Path, split_dir: str, raw_path: str) -> Iterable[Path]:
    raw = Path(raw_path)
    if raw.is_absolute():
        yield raw
    yield root / raw
    yield root / "composite" / split_dir / raw
    yield root / "composite" / split_dir / raw.name
    yield root / "composite" / raw
    yield root / split_dir / raw
    yield root / split_dir / raw.name


def _resolve_existing(root: Path, split_dir: str, raw_path: str) -> Path | None:
    for candidate in _candidate_paths(root, split_dir, raw_path):
        if candidate.exists():
            return candidate
    return None


def _mask_candidates(image_path: Path) -> Iterable[Path]:
    name = image_path.name
    stem = image_path.stem
    suffix = image_path.suffix
    yield image_path.with_name(f"mask_{name}")
    yield image_path.with_name(f"mask_{stem}{suffix}")
    for ext in IMAGE_EXTS:
        yield image_path.with_name(f"mask_{stem}{ext}")


def _label_from_name(path: Path) -> int | None:
    stem = path.stem
    for token in reversed(stem.replace("-", "_").split("_")):
        if token in {"0", "1"}:
            return int(token)
    return None


def _read_csv_samples(root: Path, split: str) -> list[OPASample]:
    split_dir = _split_dir_name(split)
    csv_candidates = [
        root / f"{split_dir}.csv",
        root / f"{split}.csv",
        root / f"{split}_set.csv",
    ]
    csv_path = next((path for path in csv_candidates if path.exists()), None)
    if csv_path is None:
        return []

    samples: list[OPASample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                label = int(float(row[-3]))
                image_raw = row[-2]
                mask_raw = row[-1]
            except ValueError:
                continue
            image_path = _resolve_existing(root, split_dir, image_raw)
            mask_path = _resolve_existing(root, split_dir, mask_raw)
            if image_path is None or mask_path is None:
                continue
            samples.append(OPASample(image_path=image_path, mask_path=mask_path, label=label))
    return samples


def _scan_samples(root: Path, split: str) -> list[OPASample]:
    split_dir = _split_dir_name(split)
    scan_roots = [
        root / "composite" / split_dir,
        root / split_dir,
        root,
    ]
    base = next((path for path in scan_roots if path.exists()), None)
    if base is None:
        return []

    samples: list[OPASample] = []
    for image_path in sorted(base.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if image_path.name.startswith("mask_"):
            continue
        label = _label_from_name(image_path)
        if label is None:
            continue
        mask_path = next((path for path in _mask_candidates(image_path) if path.exists()), None)
        if mask_path is None:
            continue
        samples.append(OPASample(image_path=image_path, mask_path=mask_path, label=label))
    return samples


def discover_opa_samples(root: str | Path, split: str) -> list[OPASample]:
    dataset_root = Path(root)
    samples = _read_csv_samples(dataset_root, split)
    if samples:
        return samples
    return _scan_samples(dataset_root, split)


class OPAAssessmentDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 128,
        augment: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.augment = augment
        self.samples = discover_opa_samples(self.root, split)
        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]
        if not self.samples:
            raise FileNotFoundError(
                f"No OPA samples found under {self.root} for split={split}. "
                "Expected train_set.csv/test_set.csv or composite/train_set images with mask_* files."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = ImageOps.exif_transpose(Image.open(sample.image_path)).convert("RGB")
        mask = ImageOps.exif_transpose(Image.open(sample.mask_path)).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        rgb = np.asarray(image).astype(np.float32) / 255.0
        mask_np = np.asarray(mask).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.dstack([rgb, mask_np])).permute(2, 0, 1).contiguous()
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return tensor, label

    def label_counts(self) -> dict[int, int]:
        counts = {0: 0, 1: 0}
        for sample in self.samples:
            counts[int(sample.label)] = counts.get(int(sample.label), 0) + 1
        return counts

