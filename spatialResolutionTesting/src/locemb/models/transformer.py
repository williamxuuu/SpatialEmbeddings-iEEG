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


class SpectralPatchEncoder(nn.Module):
    """Log-power spectrum of each time patch -> d-dim token.

    The patch (T / P samples, Hann window) gives frequency bins every sfreq*P/T Hz
    (4 Hz for 1 s at 512 Hz with 4 patches); bins up to `max_hz` are kept. No
    per-token normalisation, so how much power an electrode has in this window
    (relative to its whole-recording level, if zscore=session) reaches the model.
    This is the STFT input the strong Neuroprobe baselines use.
    """

    def __init__(self, d: int, n_patches: int, sfreq: float, n_samples: int, max_hz: float = 200.0):
        super().__init__()
        self.P = n_patches
        self.L = n_samples // n_patches
        freqs = torch.fft.rfftfreq(self.L, 1.0 / sfreq)
        keep = (freqs > 0) & (freqs <= max_hz)
        self.register_buffer("keep", keep, persistent=False)
        self.register_buffer("win", torch.hann_window(self.L), persistent=False)
        F = int(keep.sum())
        self.net = nn.Sequential(nn.Linear(F, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):                      # x: (N, T)
        x = x[:, : self.P * self.L].reshape(x.shape[0], self.P, self.L).float()
        x = x - x.mean(-1, keepdim=True)
        spec = torch.fft.rfft(x * self.win, dim=-1).abs().pow(2)[..., self.keep]
        return self.net(torch.log10(spec + 1e-4))  # (N, P, d)


class PopulationTransformer(nn.Module):
    def __init__(self, spatial: nn.Module, n_classes: int, d: int = 64, n_patches: int = 4,
                 layers: int = 4, heads: int = 4, dropout: float = 0.1,
                 frontend: str = "cnn", readout: str = "cls",
                 sfreq: float | None = None, n_samples: int | None = None):
        super().__init__()
        self.spatial = spatial
        if frontend == "cnn":
            self.patch = PatchEncoder(d, n_patches)
        elif frontend == "stft":
            if not sfreq or not n_samples:
                raise ValueError("model.frontend=stft needs the sample rate and window length")
            self.patch = SpectralPatchEncoder(d, n_patches, sfreq, n_samples)
        else:
            raise ValueError(f"Unknown model.frontend {frontend!r} (cnn | stft)")
        if readout not in ("cls", "mean"):
            raise ValueError(f"Unknown model.readout {readout!r} (cls | mean)")
        self.readout = readout
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
        if self.readout == "mean":   # average over real (non-padded) electrode tokens
            keep = (~pad[:, 1:]).unsqueeze(-1).to(h.dtype)
            z = (h[:, 1:] * keep).sum(1) / keep.sum(1).clamp_min(1)
        else:
            z = h[:, 0]
        return self.head(self.norm(z))
