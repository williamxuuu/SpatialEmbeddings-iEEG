"""Configurable signal cleaning. This is the "preprocessing axis" knob.

Config block (all keys optional):

    preprocess:
      reref: none | car        # car = subtract the mean across electrodes
      notch_hz: 60             # remove line noise + harmonics; null = off
      bandpass: [1, 200]       # Hz; null = off
      resample_hz: 512         # null = keep original rate
      zscore: true             # per window, per electrode
      clip: 10                 # clip after z-scoring; null = off

Two ways to apply it:

- `Preprocessor`: everything on each short window. Used for synthetic data.
  Filtering 1 s windows distorts low frequencies (in a simulated test the
  1-4 Hz power came out ~2x too high) and window edges, so don't use it
  for real data.
- `SessionFilter` + `WindowOps`: the filters (notch, bandpass, resample)
  run once over each whole recording (see data/session_cache.py), and
  only the per-window steps (reref, zscore, clip) run on each window.
  All three filters and CAR are linear, so doing CAR per window on
  session-filtered data gives the same result as doing CAR first.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch
from scipy import signal


class Preprocessor:
    def __init__(self, cfg: dict | None, sfreq: float):
        cfg = cfg or {}
        self.sfreq_in = float(sfreq)
        self.reref = cfg.get("reref", "none")
        self.zscore = cfg.get("zscore", True)
        self.clip = cfg.get("clip", None)
        self.resample_hz = cfg.get("resample_hz", None)
        self.sfreq_out = float(self.resample_hz or sfreq)

        self.notch = []
        if cfg.get("notch_hz"):
            f0 = float(cfg["notch_hz"])
            for h in np.arange(f0, self.sfreq_in / 2, f0):
                self.notch.append(signal.iirnotch(h, Q=30.0, fs=self.sfreq_in))
        self.sos = None
        if cfg.get("bandpass"):
            lo, hi = cfg["bandpass"]
            hi = min(hi, 0.45 * self.sfreq_in)
            self.sos = signal.butter(4, [lo, hi], btype="bandpass", fs=self.sfreq_in, output="sos")
        if self.resample_hz:
            fr = Fraction(int(self.resample_hz), int(self.sfreq_in)).limit_denominator(1000)
            self.up, self.down = fr.numerator, fr.denominator

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        a = x.numpy().astype(np.float64)
        if self.reref == "car":
            a = a - a.mean(axis=0, keepdims=True)
        elif self.reref not in ("none", None):
            raise ValueError(f"Unknown reref: {self.reref}")
        for b, aa in self.notch:
            a = signal.filtfilt(b, aa, a, axis=-1)
        if self.sos is not None:
            a = signal.sosfiltfilt(self.sos, a, axis=-1)
        if self.resample_hz:
            a = signal.resample_poly(a, self.up, self.down, axis=-1)
        if self.zscore:
            a = (a - a.mean(-1, keepdims=True)) / (a.std(-1, keepdims=True) + 1e-6)
        if self.clip:
            a = np.clip(a, -self.clip, self.clip)
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def filter_spec(cfg: dict | None) -> dict:
    """The parts of a preprocess config that are applied to whole sessions."""
    cfg = cfg or {}
    bp = cfg.get("bandpass")
    return {"notch_hz": cfg.get("notch_hz"), "bandpass": list(bp) if bp else None,
            "resample_hz": cfg.get("resample_hz")}


class SessionFilter:
    """Notch + bandpass + resample for one long 1-D signal (a whole recording).

    Order: bandpass at the raw rate, then resample (its anti-alias low-pass
    removes everything above the new Nyquist), then notch the line-noise
    harmonics still below it. A test on a simulated signal matched the
    textbook order (notch -> bandpass -> resample, all at 2048 Hz) within
    0.1-1% power in every band from 1 to 200 Hz. Doing the bandpass after
    resampling did NOT match near its 200 Hz edge (1.2-1.6x more power at
    125-200 Hz), so it stays at the raw rate. Notching at the low rate is
    ~4x cheaper and the notches are narrow, so their order doesn't matter.
    """

    def __init__(self, spec: dict, sfreq: float):
        self.sfreq_in = float(sfreq)
        self.resample_hz = spec.get("resample_hz")
        self.sfreq_out = float(self.resample_hz or sfreq)
        self.up = self.down = 1
        if self.resample_hz:
            fr = Fraction(int(self.resample_hz), int(self.sfreq_in)).limit_denominator(1000)
            self.up, self.down = fr.numerator, fr.denominator
        self.notch = []
        if spec.get("notch_hz"):
            f0 = float(spec["notch_hz"])
            for h in np.arange(f0, 0.95 * self.sfreq_out / 2, f0):
                self.notch.append(signal.iirnotch(h, Q=30.0, fs=self.sfreq_out))
        self.sos = None
        if spec.get("bandpass"):
            lo, hi = spec["bandpass"]
            hi = min(hi, 0.45 * self.sfreq_out)
            self.sos = signal.butter(4, [lo, hi], btype="bandpass", fs=self.sfreq_in, output="sos")

    def out_len(self, n_in: int) -> int:
        return int(np.ceil(n_in * self.up / self.down))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64)
        if self.sos is not None:
            a = signal.sosfiltfilt(self.sos, a)
        if self.up != 1 or self.down != 1:
            a = signal.resample_poly(a, self.up, self.down)
        for b, aa in self.notch:
            a = signal.filtfilt(b, aa, a)
        return a


class WindowOps:
    """The per-window steps: re-reference, z-score, clip. Input (E, T) array."""

    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.reref = cfg.get("reref", "none")
        if self.reref not in ("none", None, "car"):
            raise ValueError(f"Unknown reref: {self.reref}")
        self.zscore = cfg.get("zscore", True)
        self.clip = cfg.get("clip", None)

    def __call__(self, a: np.ndarray) -> torch.Tensor:
        a = np.asarray(a, dtype=np.float32)
        if self.reref == "car":
            a = a - a.mean(axis=0, keepdims=True)
        if self.zscore:
            a = (a - a.mean(-1, keepdims=True)) / (a.std(-1, keepdims=True) + 1e-6)
        if self.clip:
            a = np.clip(a, -self.clip, self.clip)
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
