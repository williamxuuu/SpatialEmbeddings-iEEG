"""Turn electrode coordinates into region weights.

Every spatial setting, hard or soft, becomes the same thing: for each
electrode, up to k (region index, weight) pairs whose weights sum to 1.

    hard       one region, weight 1 (what BaRISTA / most papers do)
    sphere     share of atlas points inside a ball of radius r around the
               electrode that belong to each region
    gaussian   same, but nearer points count more (Gaussian, width sigma)
    trilinear  grid only: blend of the 8 nearest cube centres
    prob       probabilistic atlas: how often this spot is region X across people

Experiment 2 compares these. Experiment 1 varies the atlas/grid (resolution)
with mode=hard.

Config block:

    spatial:
      type: none | coords | region
      source: atlas | prob_atlas | dataset_labels
      atlas: lobes | ho | destrieux | schaefer200 | grid20 | ho_prob | ...
      include_subcortical: true
      mode: hard | sphere | gaussian | trilinear | prob
      radius_mm: 5            # sphere
      sigma_mm: 3             # gaussian (and optional smoothing for prob)
      fallback_mm: 5          # hard mode: if electrode lands on no label
                              # (e.g. white matter), take nearest label within this
      topk: 8
      min_weight: 0.05
      control: none | shuffle_electrodes | random_neighbors
      unseen: unknown | nearest_seen   # test people's regions never seen in training:
                              # map to <unknown> (default), or to the nearest region
                              # that training did see (atlas source only)
      unseen_max_mm: 20
      dataset_label_key: destrieux   # for source=dataset_labels
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch

from .atlases import UNKNOWN, GridLabeler, get_labeler, get_prob_labeler


@dataclass
class Assignment:
    names: list[str]                                   # index 0 = <unknown>
    table: dict[str, dict[str, torch.Tensor]]          # subject -> idx, w, coords
    stats: dict = field(default_factory=dict)

    @property
    def n_regions(self) -> int:
        return len(self.names)


def ball_offsets(radius: float, step: float = 1.0):
    r = np.arange(-radius, radius + 1e-9, step)
    g = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    d = np.linalg.norm(g, axis=1)
    keep = d <= radius + 1e-9
    g, d = g[keep], d[keep]
    order = np.argsort(d, kind="stable")
    return g[order], d[order]


# ----------------------------------------------------------------------------
# Per-electrode raw weights: list[dict[name -> weight]]
# ----------------------------------------------------------------------------
def _votes_categorical(labeler, coords, mode, radius, sigma, fallback):
    E = len(coords)
    out: list[dict[str, float]] = [dict() for _ in range(E)]

    if mode == "hard":
        codes = labeler(coords)
        if fallback and (codes < 0).any():
            off, _ = ball_offsets(fallback)
            for e in np.where(codes < 0)[0]:
                c = labeler(coords[e] + off)
                hit = np.where(c >= 0)[0]
                if len(hit):
                    codes[e] = c[hit[0]]  # offsets are sorted by distance -> nearest
        for e, c in enumerate(codes):
            if c >= 0:
                out[e][labeler.name_of(int(c))] = 1.0
        return out

    if mode == "trilinear":
        if not isinstance(labeler, GridLabeler):
            raise ValueError("mode=trilinear only works with grid atlases")
        s = labeler.size
        u = coords / s - 0.5                     # position in units of cube centres
        base = np.floor(u).astype(np.int64)
        frac = u - base
        for corner in np.ndindex(2, 2, 2):
            cnr = np.array(corner)
            wt = np.prod(np.where(cnr == 1, frac, 1 - frac), axis=1)
            codes = labeler.encode(base + cnr)
            for e in range(E):
                if wt[e] > 0:
                    nm = labeler.name_of(int(codes[e]))
                    out[e][nm] = out[e].get(nm, 0.0) + float(wt[e])
        return out

    if mode == "sphere":
        off, dist = ball_offsets(radius)
        pw = np.ones(len(off))
    elif mode == "gaussian":
        off, dist = ball_offsets(3 * sigma)
        pw = np.exp(-dist ** 2 / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown mode {mode!r} for categorical atlas")

    M = len(off)
    codes = labeler((coords[:, None, :] + off[None]).reshape(-1, 3)).reshape(E, M)
    for e in range(E):
        ok = codes[e] >= 0
        if not ok.any():
            continue
        u, inv = np.unique(codes[e][ok], return_inverse=True)
        sums = np.bincount(inv, weights=pw[ok])
        for c, s_ in zip(u, sums):
            out[e][labeler.name_of(int(c))] = float(s_)
    # electrodes with nothing labeled nearby: try hard fallback
    missing = [e for e in range(E) if not out[e]]
    if missing and fallback:
        sub = _votes_categorical(labeler, coords[missing], "hard", 0, 0, max(fallback, radius))
        for e, v in zip(missing, sub):
            out[e] = v
    return out


def _votes_prob(labeler, coords, mode, sigma, fallback):
    E = len(coords)
    if sigma and sigma > 0:
        off, dist = ball_offsets(3 * sigma)
        pw = np.exp(-dist ** 2 / (2 * sigma ** 2))
        P = labeler((coords[:, None, :] + off[None]).reshape(-1, 3)).reshape(E, len(off), -1)
        P = (P * pw[None, :, None]).sum(1) / pw.sum()
    else:
        P = labeler(coords)
    if fallback:
        off, _ = ball_offsets(fallback)
        for e in np.where(P.sum(1) <= 1e-6)[0]:
            Q = labeler(coords[e] + off)
            hit = np.where(Q.sum(1) > 1e-6)[0]
            if len(hit):
                P[e] = Q[hit[0]]
    out = []
    for e in range(E):
        if mode == "hard":
            out.append({labeler.names[int(P[e].argmax())]: 1.0} if P[e].sum() > 1e-6 else {})
        else:
            out.append({labeler.names[j]: float(P[e, j]) for j in np.nonzero(P[e] > 1e-6)[0]})
    return out


def raw_votes(spec: dict, subj) -> list[dict[str, float]]:
    source = spec.get("source", "atlas")
    mode = spec.get("mode", "hard")
    if source == "dataset_labels":
        key = spec["dataset_label_key"]
        labels = subj.dataset_labels[key]
        return [{lab: 1.0} if lab and lab.lower() not in ("unknown", "nan") else {} for lab in labels]
    coords = np.asarray(subj.coords, dtype=np.float64)
    inc_sub = spec.get("include_subcortical", True)
    if source == "prob_atlas":
        return _votes_prob(get_prob_labeler(spec["atlas"], inc_sub), coords, mode,
                           spec.get("sigma_mm", 0), spec.get("fallback_mm", 5))
    lab = get_labeler(spec["atlas"], inc_sub)
    return _votes_categorical(lab, coords, mode, spec.get("radius_mm", 5),
                              spec.get("sigma_mm", 3), spec.get("fallback_mm", 5))


# ----------------------------------------------------------------------------
# Build the full table
# ----------------------------------------------------------------------------
def _entropy(w):
    w = w[w > 0]
    return float(-(w * np.log(w)).sum()) if len(w) else 0.0


def _nearest_seen(spec: dict, xyz: np.ndarray, index: dict, max_mm: float):
    """Name of the region nearest to xyz (within max_mm) that training saw, or None."""
    lab = get_labeler(spec["atlas"], spec.get("include_subcortical", True))
    off, _ = ball_offsets(max_mm)                      # sorted by distance
    for c in lab(xyz[None] + off):
        if c >= 0 and (n := lab.name_of(int(c))) in index and n != UNKNOWN:
            return n
    return None


def build_assignment(spec: dict, subjects: dict, fit_subjects: set[str], seed: int = 0) -> Assignment:
    sp_type = spec.get("type", "none")
    k = int(spec.get("topk", 8)) if sp_type == "region" else 1

    if sp_type in ("none", "coords"):
        table = {sid: {"idx": torch.zeros(s.n_electrodes, 1, dtype=torch.long),
                       "w": torch.zeros(s.n_electrodes, 1),
                       "coords": torch.as_tensor(np.asarray(s.coords), dtype=torch.float32)}
                 for sid, s in subjects.items()}
        return Assignment([UNKNOWN], table, {"n_regions": 0, "type": sp_type})
    if sp_type != "region":
        raise ValueError(f"Unknown spatial.type {sp_type!r}")

    min_w = float(spec.get("min_weight", 0.05))
    votes = {}
    for sid, s in subjects.items():
        v = raw_votes(spec, s)
        if len(v) != s.n_electrodes:
            raise ValueError(f"{sid}: {len(v)} labels for {s.n_electrodes} electrodes")
        norm = []
        for d in v:
            tot = sum(d.values())
            d = {n: w / tot for n, w in d.items()} if tot > 0 else {}
            top = sorted(d.items(), key=lambda t: -t[1])
            top = [t for i, t in enumerate(top) if i == 0 or t[1] >= min_w][:k]
            tot = sum(w for _, w in top)
            norm.append([(n, w / tot) for n, w in top])
        votes[sid] = norm

    # Vocabulary comes from TRAINING subjects only (no peeking at test people).
    vocab = sorted({n for sid in fit_subjects for el in votes[sid] for n, _ in el})
    names = [UNKNOWN] + vocab
    index = {n: i for i, n in enumerate(names)}

    rng = np.random.default_rng(seed + 991)
    control = spec.get("control", "none")
    unseen = spec.get("unseen", "unknown")
    if unseen not in ("unknown", "nearest_seen"):
        raise ValueError(f"Unknown spatial.unseen {unseen!r}")
    if unseen == "nearest_seen" and spec.get("source", "atlas") != "atlas":
        raise ValueError("spatial.unseen=nearest_seen needs spatial.source=atlas")
    n_remapped = 0
    table = {}
    for sid, s in subjects.items():
        E = s.n_electrodes
        idx = np.zeros((E, k), dtype=np.int64)
        w = np.zeros((E, k), dtype=np.float32)
        for e, el in enumerate(votes[sid]):
            if not el:
                idx[e, 0], w[e, 0] = 0, 1.0           # no label anywhere near: <unknown>
                continue
            near = None
            if unseen == "nearest_seen" and any(n not in index for n, _ in el):
                near = _nearest_seen(spec, np.asarray(s.coords[e], dtype=np.float64), index,
                                     float(spec.get("unseen_max_mm", 20)))
                n_remapped += near is not None
            merged: dict[int, float] = {}
            for n, wt in el:
                j = index.get(n, index.get(near, 0))   # never seen in training -> nearest seen, else <unknown>
                merged[j] = merged.get(j, 0.0) + wt
            for col, (j, wt) in enumerate(sorted(merged.items(), key=lambda t: -t[1])):
                idx[e, col], w[e, col] = j, wt

        if control == "shuffle_electrodes":
            # Break the link between an electrode and its true location,
            # keeping the same set of region codes within the subject.
            perm = rng.permutation(E)
            idx, w = idx[perm], w[perm]
        elif control == "random_neighbors":
            # Keep each electrode's main region, but swap its secondary regions
            # for random ones (same weights). Tests whether the RIGHT neighbours
            # matter, or just spreading weight around.
            V = len(names)
            for e in range(E):
                for col in range(1, k):
                    if w[e, col] > 0 and V > 2:
                        r = idx[e, 0]
                        while r == idx[e, 0]:
                            r = int(rng.integers(1, V))
                        idx[e, col] = r
        elif control not in ("none", None):
            raise ValueError(f"Unknown control {control!r}")

        table[sid] = {"idx": torch.from_numpy(idx), "w": torch.from_numpy(w),
                      "coords": torch.as_tensor(np.asarray(s.coords), dtype=torch.float32)}

    # ---------------- stats worth logging with every run ----------------
    top1 = Counter(int(table[sid]["idx"][e, 0]) for sid in fit_subjects
                   for e in range(subjects[sid].n_electrodes))
    region_counts = np.array([top1.get(i, 0) for i in range(1, len(names))])
    all_w = [table[sid]["w"][e].numpy() for sid in subjects for e in range(subjects[sid].n_electrodes)]
    test_ids = [sid for sid in subjects if sid not in fit_subjects]

    def unk_frac(ids):
        tot = sum(subjects[s].n_electrodes for s in ids)
        if tot == 0:
            return None
        return float(sum((table[s]["w"] * (table[s]["idx"] == 0)).sum() for s in ids) / tot)

    stats = {
        "type": "region",
        "n_regions": len(vocab),
        "electrodes_per_region_median": float(np.median(region_counts)) if len(region_counts) else 0.0,
        "frac_regions_lt3_electrodes": float((region_counts < 3).mean()) if len(region_counts) else 0.0,
        "frac_weight_unknown_train": unk_frac(sorted(fit_subjects)),
        "frac_weight_unknown_test_subjects": unk_frac(test_ids),
        "mean_weight_entropy_nats": float(np.mean([_entropy(x) for x in all_w])),
        "frac_electrodes_multi_region": float(np.mean([(x > 0).sum() > 1 for x in all_w])),
        "frac_test_electrodes_remapped": (n_remapped / sum(subjects[s].n_electrodes for s in test_ids)
                                          if test_ids else None),
    }
    return Assignment(names, table, stats)
