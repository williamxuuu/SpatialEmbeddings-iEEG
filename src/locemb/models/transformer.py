"""A small BaRISTA/PopT-style population transformer, trained from scratch.

Each electrode's 1-s window is cut into P time patches. A small CNN turns each
patch into a d-dim token. We add (a) the electrode's spatial vector and (b) a
learned time-position vector, then one transformer attends across all
electrodes x patches. A CLS token is read out for classification.

This is a stand-in so the pipeline runs end to end. To test the "does the best
resolution depend on architecture" question, plug the real models (BaRISTA,
PopT, ...) in behind the same `forward(batch) -> logits` interface.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchEncoder(nn.Module):
    def __init__(self, d: int, n_patches: int, channels: int = 32):
        super().__init__()
        C = channels
        self.net = nn.Sequential(
            nn.Conv1d(1, C, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(C, C, 5, stride=2, padding=2, dilation=1), nn.GELU(),
            nn.Conv1d(C, C, 5, stride=2, padding=4, dilation=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(n_patches),
        )
        self.proj = nn.Linear(C, d)

    def forward(self, x):                      # x: (N, T)
        h = self.net(x.unsqueeze(1))           # (N, C, P)
        return self.proj(h.transpose(1, 2))    # (N, P, d)


class PopulationTransformer(nn.Module):
    def __init__(self, spatial: nn.Module, n_classes: int, d: int = 64, n_patches: int = 4,
                 layers: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.spatial = spatial
        self.patch = PatchEncoder(d, n_patches)
        self.time_pos = nn.Parameter(torch.zeros(n_patches, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.time_pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        self.P, self.d = n_patches, d

    def forward(self, batch):
        x, mask = batch["x"], batch["mask"]                     # (B,E,T), (B,E)
        B, E, T = x.shape
        tok = self.patch(x.reshape(B * E, T)).reshape(B, E, self.P, self.d)
        tok = tok + self.spatial(batch).unsqueeze(2) + self.time_pos
        tok = tok.reshape(B, E * self.P, self.d)
        pad = ~mask.repeat_interleave(self.P, dim=1)            # True = ignore
        tok = torch.cat([self.cls.expand(B, -1, -1), tok], 1)
        pad = torch.cat([pad.new_zeros(B, 1), pad], 1)
        h = self.encoder(tok, src_key_padding_mask=pad)
        return self.head(self.norm(h[:, 0]))
