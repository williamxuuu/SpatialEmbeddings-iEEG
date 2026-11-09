"""Shared data containers.

Every dataset adapter (synthetic, Neuroprobe, ...) returns a `DataBundle`.
Each item of a split is a dict: {"x": FloatTensor (n_electrodes, n_samples),
"y": int, "subject": str}. Different subjects can have different electrode
counts; `Collator` pads them and adds the spatial inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SubjectInfo:
    subject_id: str
    coords: np.ndarray                     # (E, 3) electrode coordinates, MNI mm
    electrode_names: list[str]
    # Optional per-electrode region labels shipped with the dataset, e.g.
    # {"destrieux": [...], "dk": [...]}. Used for exact parity with papers.
    dataset_labels: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_electrodes(self) -> int:
        return len(self.electrode_names)


@dataclass
class DataBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    subjects: dict[str, SubjectInfo]
    train_subjects: set[str]               # spatial vocab is fit on these only
    sfreq: float
    n_classes: int
    description: str = ""


class SubjectTagged(Dataset):
    """Wraps any dataset returning (x, y) or dicts, and tags items with a subject id."""

    def __init__(self, base: Dataset, subject: str, transform=None):
        self.base, self.subject, self.transform = base, subject, transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        if isinstance(item, dict):
            x = item.get("data", item.get("x"))
            y = item.get("label", item.get("y"))
        else:
            x, y = item[0], item[1]
        x = torch.as_tensor(x, dtype=torch.float32)
        if self.transform is not None:
            x = self.transform(x)
        return {"x": x, "y": int(torch.as_tensor(y).item()), "subject": self.subject}


class ArrayDataset(Dataset):
    """In-memory windows (used by the synthetic generator)."""

    def __init__(self, xs: list[np.ndarray], ys: list[int], subjects: list[str], transform=None):
        self.xs, self.ys, self.subjects, self.transform = xs, ys, subjects, transform

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        x = torch.as_tensor(self.xs[i], dtype=torch.float32)
        if self.transform is not None:
            x = self.transform(x)
        return {"x": x, "y": int(self.ys[i]), "subject": self.subjects[i]}


class Collator:
    """Pads a batch to the largest electrode count and attaches spatial inputs.

    spatial_table[subject] = {"idx": LongTensor (E,k), "w": FloatTensor (E,k),
                              "coords": FloatTensor (E,3)}
    """

    def __init__(self, spatial_table: dict[str, dict[str, torch.Tensor]]):
        self.table = spatial_table

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        B = len(items)
        E = max(it["x"].shape[0] for it in items)
        T = items[0]["x"].shape[1]
        k = next(iter(self.table.values()))["idx"].shape[1]
        x = torch.zeros(B, E, T)
        mask = torch.zeros(B, E, dtype=torch.bool)
        idx = torch.zeros(B, E, k, dtype=torch.long)
        w = torch.zeros(B, E, k)
        coords = torch.zeros(B, E, 3)
        for b, it in enumerate(items):
            e = it["x"].shape[0]
            s = self.table[it["subject"]]
            if s["idx"].shape[0] != e:
                raise ValueError(
                    f"Subject {it['subject']}: data has {e} electrodes but spatial table has "
                    f"{s['idx'].shape[0]}. Electrode lists are out of sync.")
            x[b, :e] = it["x"]
            mask[b, :e] = True
            idx[b, :e] = s["idx"]
            w[b, :e] = s["w"]
            coords[b, :e] = s["coords"]
        y = torch.tensor([it["y"] for it in items], dtype=torch.long)
        return {"x": x, "mask": mask, "sp_idx": idx, "sp_w": w, "coords": coords, "y": y,
                "subject": [it["subject"] for it in items]}
