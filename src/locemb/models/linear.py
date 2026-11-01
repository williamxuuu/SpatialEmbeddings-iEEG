"""Linear baseline (the MAPPA-style "how far does a simple model get" check).

Features: log power in a few frequency bands for each electrode.
How space enters a linear model: electrodes are pooled INTO REGIONS using the
same (soft or hard) region weights as the transformer, giving a fixed-size
vector (regions x bands) that means the same thing for every subject. Then one
linear layer. So the region resolution matters here too:
  - spatial.type=none   -> average over all electrodes (location-blind)
  - spatial.type=region -> per-region averages (hard or soft weights)
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 70), (70, 150)]


class BandPower(nn.Module):
    def __init__(self, sfreq: float, bands=DEFAULT_BANDS):
        super().__init__()
        self.sfreq = sfreq
        self.bands = [(lo, hi) for lo, hi in bands if lo < sfreq / 2]

    def forward(self, x):                                   # (B,E,T) -> (B,E,F)
        spec = torch.fft.rfft(x, dim=-1).abs() ** 2
        freqs = torch.fft.rfftfreq(x.shape[-1], 1 / self.sfreq).to(x.device)
        feats = [spec[..., (freqs >= lo) & (freqs < min(hi, self.sfreq / 2))].mean(-1)
                 for lo, hi in self.bands]
        return torch.log(torch.stack(feats, -1) + 1e-6)


class RegionPooledLinear(nn.Module):
    def __init__(self, n_regions: int, sfreq: float, n_classes: int, spatial_type: str):
        super().__init__()
        if spatial_type == "coords":
            raise ValueError("Linear baseline supports spatial.type none or region, not coords")
        self.bp = BandPower(sfreq)
        self.F = len(self.bp.bands)
        self.K = n_regions if spatial_type == "region" else 1
        self.region = spatial_type == "region"
        self.norm = nn.LayerNorm(self.F, elementwise_affine=False)
        self.head = nn.Linear(self.K * self.F, n_classes)

    def forward(self, batch):
        f = self.norm(self.bp(batch["x"]))                    # (B,E,F)
        m = batch["mask"].float().unsqueeze(-1)
        if not self.region:
            pooled = (f * m).sum(1) / m.sum(1).clamp_min(1)
            return self.head(pooled)
        B = f.shape[0]
        w = batch["sp_w"] * batch["mask"].unsqueeze(-1)        # (B,E,k)
        idx = batch["sp_idx"]                                  # (B,E,k)
        R = f.new_zeros(B, self.K, self.F)
        Z = f.new_zeros(B, self.K)
        R.scatter_add_(1, idx.reshape(B, -1, 1).expand(-1, -1, self.F),
                       (w.unsqueeze(-1) * f.unsqueeze(2)).reshape(B, -1, self.F))
        Z.scatter_add_(1, idx.reshape(B, -1), w.reshape(B, -1))
        R = R / Z.clamp_min(1e-6).unsqueeze(-1)                # empty regions stay 0
        return self.head(R.reshape(B, -1))
