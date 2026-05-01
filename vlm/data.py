"""Dataset loaders for EuroSAT (§3), RESISC45 (§4), and CLEVR (§5).

All loaders return torch DataLoaders. Images are resized to 64x64 and
normalized to ImageNet stats unless otherwise specified.

DO NOT MODIFY THIS FILE (you may extend it, but the staff tests rely on the
provided functions).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ImageNet normalization used by most pretrained vision encoders.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_image_transform(img_size: int = 64) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# EuroSAT — §3 (CLIP pretraining)
# ---------------------------------------------------------------------------


EUROSAT_CLASSES = [
    "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway",
    "Industrial Buildings", "Pasture", "Permanent Crop", "Residential Buildings",
    "River", "Sea or Lake",
]


class EuroSATCLIPDataset(Dataset):
    """EuroSAT with synthetic captions of the form
    'a satellite image of {class_name}'.

    Yields (image_tensor, caption_string) tuples.
    """

    def __init__(self, split: str = "train", img_size: int = 64) -> None:
        from datasets import load_dataset

        self.ds = load_dataset("blanchon/EuroSAT_RGB", split=split)
        self.transform = default_image_transform(img_size)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        label_idx = ex["label"]
        class_name = EUROSAT_CLASSES[label_idx]
        caption = f"a satellite image of {class_name}"
        return self.transform(img), caption


def build_eurosat_loaders(
    img_size: int = 64,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train = EuroSATCLIPDataset("train[:80%]", img_size=img_size)
    val = EuroSATCLIPDataset("train[80%:90%]", img_size=img_size)
    test = EuroSATCLIPDataset("train[90%:]", img_size=img_size)

    def _collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        caps = [b[1] for b in batch]
        return imgs, caps

    train_dl = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True,
    )
    test_dl = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True,
    )
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# RESISC45 — §4 (LoRA / full FT downstream task)
# ---------------------------------------------------------------------------


class RESISC45Dataset(Dataset):
    """Remote-sensing scene classification with 45 categories.

    Yields (image_tensor, label_int) tuples.
    """

    def __init__(self, split: str = "train", img_size: int = 64) -> None:
        from datasets import load_dataset

        self.ds = load_dataset("timm/resisc45", split=split)
        self.transform = default_image_transform(img_size)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        return self.transform(img), int(ex["label"])


def build_resisc45_loaders(
    img_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train = RESISC45Dataset("train", img_size=img_size)
    test = RESISC45Dataset("validation", img_size=img_size)
    train_dl = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    test_dl = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return train_dl, test_dl


# ---------------------------------------------------------------------------
# CLEVR — §5 (VLM training and evaluation)
# ---------------------------------------------------------------------------


class CLEVRMiniDataset(Dataset):
    """Preprocessed 10k-example CLEVR subset.

    Expects on disk:
        data/clevr_mini/{split}.jsonl   (one JSON per line: image_file, question, answer, q_type)
        data/clevr_mini/images/         (PNG files referenced by image_file)
    """

    def __init__(
        self,
        split: str = "train",
        root: str = "data/clevr_mini",
        img_size: int = 64,
    ) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.transform = default_image_transform(img_size)
        with open(self.root / f"{split}.jsonl") as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        img = Image.open(self.root / "images" / ex["image_file"]).convert("RGB")
        return {
            "image": self.transform(img),
            "question": ex["question"],
            "answer": ex["answer"],
            "q_type": ex.get("q_type", "other"),  # "spatial" or "other" for §6
        }


def build_clevr_loaders(
    img_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train = CLEVRMiniDataset("train", img_size=img_size)
    val = CLEVRMiniDataset("val", img_size=img_size)

    def _collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "question": [b["question"] for b in batch],
            "answer": [b["answer"] for b in batch],
            "q_type": [b["q_type"] for b in batch],
        }

    train_dl = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True,
    )
    return train_dl, val_dl
