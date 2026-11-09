"""Fake sEEG with a known spatial rule. For smoke tests and sanity checks ONLY.

The rule: a few "hot spots" in the brain respond to the label. Some hot spots
make high-frequency power go UP, others make it go DOWN, and each subject has a
random overall power level. So a model can only use an electrode well if it
knows WHERE the electrode is. A spatial encoding should therefore help here,
which makes this a quick check that the spatial code path does something.
Nothing learned on this data says anything about real brains.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayDataset, DataBundle, SubjectInfo
from .preprocess import Preprocessor


def _sample_coords(rng, n):
    # Points inside a brain-sized ellipsoid in MNI-like mm coordinates.
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1, 1, size=3)
        if (p ** 2).sum() <= 1:
            pts.append(p * np.array([65.0, 90.0, 60.0]) + np.array([0.0, -15.0, 15.0]))
    return np.array(pts)


def make_synthetic(cfg: dict, seed: int) -> DataBundle:
    c = cfg.get("synthetic", {})
    n_subj = c.get("n_subjects", 8)
    n_test_subj = c.get("n_test_subjects", 2)
    n_win = c.get("windows_per_subject", 200)
    sfreq = float(c.get("sfreq", 256))
    T = int(c.get("seconds", 1.0) * sfreq)
    e_lo, e_hi = c.get("electrodes", [30, 60])
    amp = c.get("effect_size", 0.6)
    blob_mm = c.get("blob_mm", 15.0)
    split = cfg["data"].get("split", "cross_subject")

    rng = np.random.default_rng(1234)  # fixed "brain": same hot spots for every seed
    centers = _sample_coords(rng, 6)
    signs = np.array([1, -1, 1, -1, 1, -1], dtype=float)

    rng = np.random.default_rng(seed)
    subjects, xs, ys, ss = {}, [], [], []
    t = np.arange(T) / sfreq
    for si in range(n_subj):
        sid = f"syn{si:02d}"
        E = int(rng.integers(e_lo, e_hi + 1))
        coords = _sample_coords(rng, E)
        d2 = ((coords[:, None, :] - centers[None]) ** 2).sum(-1)
        resp = (np.exp(-d2 / (2 * blob_mm ** 2)) * signs).sum(1)        # (E,)
        gain = rng.uniform(0.5, 2.0)                                     # subject power level
        subjects[sid] = SubjectInfo(sid, coords, [f"{sid}_e{i}" for i in range(E)])
        for _ in range(n_win):
            y = int(rng.integers(0, 2))
            noise = np.cumsum(rng.normal(size=(E, T)), axis=1)           # slow drift
            noise = noise - noise.mean(1, keepdims=True)
            noise = 0.05 * noise + rng.normal(size=(E, T))
            f = rng.uniform(60, 90)
            burst = np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi, size=(E, 1)))
            mod = 1.0 + amp * (2 * y - 1) * resp[:, None]
            x = gain * (noise + np.clip(mod, 0, None) * burst)
            xs.append(x.astype(np.float32)); ys.append(y); ss.append(sid)

    pre = Preprocessor(cfg.get("preprocess"), sfreq)
    ids = sorted(subjects)
    if split == "cross_subject":
        test_ids = set(ids[-n_test_subj:])
        tr = [i for i, s in enumerate(ss) if s not in test_ids]
        te = [i for i, s in enumerate(ss) if s in test_ids]
        rng.shuffle(tr)
        n_val = len(tr) // 8
        va, tr = tr[:n_val], tr[n_val:]
        train_subjects = set(ids) - test_ids
    elif split == "within_subject":
        order = rng.permutation(len(ys))
        n = len(order)
        tr, va, te = order[: int(.7 * n)], order[int(.7 * n): int(.8 * n)], order[int(.8 * n):]
        train_subjects = set(ids)
    else:
        raise ValueError(f"Unknown split {split}")

    def sub(ix):
        return ArrayDataset([xs[i] for i in ix], [ys[i] for i in ix], [ss[i] for i in ix], pre)

    return DataBundle(sub(tr), sub(va), sub(te), subjects, train_subjects,
                      sfreq=pre.sfreq_out, n_classes=2,
                      description=f"synthetic ({split}, {n_subj} subjects)")
