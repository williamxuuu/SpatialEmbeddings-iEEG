"""Spatial encoders: batch -> (B, E, d) vector per electrode, added to its tokens.

RegionEncoder is the same for hard and soft assignment: one learned vector per
region, and an electrode gets the weighted sum  sum_j w_j * table[region_j].
Hard = one weight of 1. So Experiment 2 changes ONLY the weights, not the model.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class NoSpatial(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, batch):
        B, E = batch["mask"].shape
        return batch["mask"].new_zeros(B, E, self.d, dtype=torch.float32)


class RegionEncoder(nn.Module):
    def __init__(self, n_regions: int, d: int):
        super().__init__()
        self.table = nn.Embedding(n_regions, d)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, batch):
        emb = self.table(batch["sp_idx"])                        # (B, E, k, d)
        return (emb * batch["sp_w"].unsqueeze(-1)).sum(-2)       # (B, E, d)


class CoordEncoder(nn.Module):
    """Continuous xyz position (Fourier features -> linear). The 'infinitely fine' end."""

    def __init__(self, d: int, n_freqs: int = 16, min_wavelength_mm: float = 4.0,
                 max_wavelength_mm: float = 256.0):
        super().__init__()
        freqs = 2 * math.pi / torch.logspace(math.log10(min_wavelength_mm),
                                             math.log10(max_wavelength_mm), n_freqs)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(3 * 2 * n_freqs, d)

    def forward(self, batch):
        c = batch["coords"].unsqueeze(-1) * self.freqs           # (B, E, 3, F)
        f = torch.cat([c.sin(), c.cos()], -1).flatten(-2)        # (B, E, 6F)
        return self.proj(f) * batch["mask"].unsqueeze(-1)


def build_spatial_encoder(spec: dict, n_regions: int, d: int) -> nn.Module:
    t = spec.get("type", "none")
    if t == "none":
        return NoSpatial(d)
    if t == "region":
        return RegionEncoder(n_regions, d)
    if t == "coords":
        return CoordEncoder(d, **spec.get("coords_kwargs", {}))
    raise ValueError(f"Unknown spatial type {t!r}")
