"""Diagnostic: per-window, per-electrode log band power on every Lite session,
computed exactly the way the v2 model sees the signal (CAR, session z-score,
clip, 4 Hann patches of 0.25 s, log10 power).

Used to tell data-level problems (does the speech vs non-speech contrast keep
its sign across people / movies?) from model-level ones (identity shortcuts,
overfitting). Writes one .npz per session to --out:
    feat   float16 (N, E, B)   mean over the 4 patches of log10 band power
    y      int8    (N,)
    start  int64   (N,)        window start sample (2048 Hz), i.e. time in the movie
    part   str                 train | val | test   (cross_session roles)
    electrodes, coords (E, 3)

    python scripts/diag_bandpower.py --out <dir>
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from locemb.config import load_config
from locemb.data import build_data

BANDS = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 200)]


def band_feats(x: torch.Tensor, sfreq: float, P: int = 4) -> torch.Tensor:
    """x: (B, E, T) -> (B, E, len(BANDS)) mean over patches of log10 band power."""
    B, E, T = x.shape
    L = T // P
    x = x[..., : P * L].reshape(B, E, P, L)
    x = x - x.mean(-1, keepdim=True)
    spec = torch.fft.rfft(x * torch.hann_window(L), dim=-1).abs().pow(2)
    f = torch.fft.rfftfreq(L, 1.0 / sfreq)
    out = []
    for lo, hi in BANDS:
        k = (f > lo) & (f <= hi)
        out.append(torch.log10(spec[..., k].mean(-1) + 1e-4).mean(-1))
    return torch.stack(out, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/neuroprobe_v2.yaml")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config, ["data.split=cross_session", "data.electrodes=all",
                                 "preprocess.reref=car", "model.frontend=stft"])
    bundle = build_data(cfg, 0)
    print(bundle.description, flush=True)
    for part, cat in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
        for ds in cat.datasets:
            sid = ds.subject
            info = bundle.subjects[sid]
            dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=a.workers,
                            collate_fn=lambda it: (torch.stack([i["x"] for i in it]),
                                                   torch.tensor([i["y"] for i in it])))
            F, Y = [], []
            for x, y in dl:
                F.append(band_feats(x, bundle.sfreq).numpy().astype(np.float16)); Y.append(y.numpy())
            np.savez_compressed(os.path.join(a.out, f"{sid}_{part}.npz"), feat=np.concatenate(F),
                                y=np.concatenate(Y).astype(np.int8), start=np.asarray(ds.starts),
                                part=part, electrodes=np.array(info.electrode_names),
                                coords=np.asarray(info.coords))
            print(f"[diag] {sid} {part}: {len(ds)} windows x {info.n_electrodes} electrodes", flush=True)


if __name__ == "__main__":
    main()
