"""Adapter for Neuroprobe / Brain Treebank (https://github.com/azaho/neuroprobe).

Written against the neuroprobe package source (v0.1.6/0.1.8, commit b282acd),
and tested against a fake data folder laid out like the real one
(tests/test_neuroprobe_adapter.py). Not yet run on the real recordings: run
`python scripts/inspect_neuroprobe.py` on the cluster first.

How it works:
1. Neuroprobe's own dataset classes pick the windows and labels
   (`output_indices=True`, so they return sample positions, not signals).
   We always use Lite windows (<=3500 per task, Lite sessions only), so the
   examples are exactly the benchmark's.
2. The signal comes from our whole-recording filter cache (session_cache.py).
3. `electrodes` picks which electrodes the model sees:
     lite: Neuroprobe Lite's <=120 per subject (the benchmark's choice;
           picked by how well a linear model decoded from them)
     all:  every usable electrode (no corrupted / trigger / unlocalized ones)
   Windows and labels are identical in both, so the two are directly comparable.

Config:

    data:
      name: neuroprobe
      root: /path/to/braintreebank   # or $ROOT_DIR_BRAINTREEBANK
      cache_dir: null                # default: $LOCEMB_CACHE_DIR or <root>/../locemb_cache
      eval_name: speech              # one of npb.config.NEUROPROBE_TASKS
      split: cross_session           # within_session | cross_session | cross_subject
      fold: 0                        # within_session only (Lite has 2 folds)
      electrodes: lite               # lite | all
      # (subject, test trial). within/cross_session: one entry per subject;
      # cross_session trains on that subject's other Lite trial.
      subject_trials: [[1, 2], [2, 4], [3, 1], [4, 1], [7, 1], [10, 1]]
      # cross_subject only: these subjects are tested (on their trial above);
      # every Lite trial of the other subjects is used for training.
      test_subjects: [2, 7]
      exclude_test_movie: true       # cross_subject: drop training trials of the same movie

All listed subjects are trained into ONE model, so location codes are shared
across people. Validation = first half of each test recording's windows,
test = second half (Neuroprobe's own convention).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from .base import DataBundle, SubjectInfo
from .preprocess import WindowOps, filter_spec
from .session_cache import SessionData, build_session, default_cache_dir


class CachedWindows(Dataset):
    def __init__(self, session: SessionData, rows: np.ndarray, starts: np.ndarray,
                 labels: np.ndarray, n_out: int, subject: str, ops: WindowOps):
        self.session, self.rows, self.starts, self.labels = session, rows, starts, labels
        self.n_out, self.subject, self.ops = n_out, subject, ops

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.ops(self.session.window(int(self.starts[i]), self.n_out, self.rows))
        return {"x": x, "y": int(self.labels[i]), "subject": self.subject}


def _indices(ds) -> tuple[np.ndarray, np.ndarray, int]:
    """(window starts at 2048 Hz, labels, window length at 2048 Hz) for a
    Neuroprobe dataset built with output_indices=True (or a Subset of one)."""
    starts, labels, lens = [], [], set()
    for i in range(len(ds)):
        item = ds[i]
        (a, b), y = (item["data"], item["label"]) if isinstance(item, dict) else item
        starts.append(int(a)); labels.append(int(y)); lens.add(int(b) - int(a))
    if len(lens) != 1:
        raise RuntimeError(f"windows have different lengths: {sorted(lens)[:5]}")
    return np.array(starts), np.array(labels), lens.pop()


def _halves(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(n // 2), np.arange(n // 2, n)


def make_neuroprobe(cfg: dict, seed: int) -> DataBundle:
    d = cfg["data"]
    root = d.get("root") or os.environ.get("ROOT_DIR_BRAINTREEBANK")
    if not root:
        raise ValueError("Set data.root or $ROOT_DIR_BRAINTREEBANK")
    os.environ["ROOT_DIR_BRAINTREEBANK"] = root  # neuroprobe reads it at import time
    import neuroprobe as npb  # noqa: imported late so other code works without it
    from neuroprobe import config as npc

    split = d.get("split", "cross_session")
    eval_name = d["eval_name"]
    which = d.get("electrodes", "lite")
    if which not in ("lite", "all"):
        raise ValueError(f"data.electrodes must be lite or all, not {which!r}")
    pairs = [(int(s), int(t)) for s, t in d["subject_trials"]]
    lite_pairs = set(npc.NEUROPROBE_LITE_SUBJECT_TRIALS)
    for p in pairs:
        if p not in lite_pairs:
            raise ValueError(f"(subject, trial) {p} is not a Neuroprobe Lite session: "
                             f"{sorted(lite_pairs)}")

    spec = filter_spec(cfg.get("preprocess"))
    cache_dir = d.get("cache_dir") or default_cache_dir(root)
    ops = WindowOps(cfg.get("preprocess"))
    subjects_npb = {s: npb.BrainTreebankSubject(s, cache=False, dtype=torch.float32,
                                                coordinates_type="mni") for s, _ in pairs}

    sessions: dict[tuple[int, int], SessionData] = {}

    def session(s: int, t: int) -> SessionData:
        if (s, t) not in sessions:
            sessions[(s, t)] = SessionData(build_session(subjects_npb[s], t, spec, cache_dir, root))
        return sessions[(s, t)]

    infos: dict[str, SubjectInfo] = {}
    elec: dict[int, list[str]] = {}

    def subject_info(s: int) -> SubjectInfo:
        sid = f"btb{s:02d}"
        if sid not in infos:
            subj = subjects_npb[s]
            if which == "lite":
                labels = [e for e in npc.NEUROPROBE_LITE_ELECTRODES[subj.subject_identifier]
                          if e in subj.electrode_labels]
            else:
                labels = list(subj.electrode_labels)
            all_xyz = np.asarray(subj.get_electrode_coordinates(), dtype=np.float64)
            pos = {e: i for i, e in enumerate(subj.electrode_labels)}
            xyz = all_xyz[[pos[e] for e in labels]]
            if np.isnan(xyz).any():
                raise RuntimeError(f"subject {s}: {int(np.isnan(xyz).any(1).sum())} electrodes "
                                   "without MNI coordinates")
            extra = {}
            try:  # per-electrode atlas labels shipped with Brain Treebank (e.g. DK / Destrieux)
                meta = subj.get_all_electrode_metadata().set_index("Electrode")
                for col in meta.columns:
                    # text columns: "object" in pandas < 3, a string dtype in pandas >= 3
                    if pd.api.types.is_object_dtype(meta[col]) or pd.api.types.is_string_dtype(meta[col]):
                        extra[col] = [str(meta.at[e, col]) for e in labels]
            except Exception as ex:  # not needed for atlas-based encodings
                print(f"[neuroprobe] subject {s}: no dataset labels ({ex})")
            infos[sid] = SubjectInfo(sid, xyz, labels, dataset_labels=extra)
            elec[s] = labels
        return infos[sid]

    def windows(s: int, t: int, ds, idx: np.ndarray | None = None) -> CachedWindows:
        starts, labels, n_raw = _indices(ds)
        if idx is not None:
            starts, labels = starts[idx], labels[idx]
        sd = session(s, t)
        info = subject_info(s)
        n_out = int(round(n_raw * sd.ratio))
        return CachedWindows(sd, sd.rows(elec[s]), starts, labels, n_out, info.subject_id, ops)

    trains, vals, tests, train_sids = [], [], [], set()
    kw = dict(eval_name=eval_name, lite=True, output_indices=True)
    if split in ("within_session", "cross_session"):
        for s, t in pairs:
            subj = subjects_npb[s]
            if split == "within_session":
                folds = npb.generate_splits_within_session(subj, t, **kw)
                fold = folds[int(d.get("fold", 0))]
                train_t = t
            else:
                fold = npb.generate_splits_cross_session(subj, t, **kw)[0]
                train_t = [tt for ss, tt in npc.NEUROPROBE_LITE_SUBJECT_TRIALS if ss == s and tt != t][0]
            trains.append(windows(s, train_t, fold["train_dataset"]))
            vals.append(windows(s, t, fold["val_dataset"]))
            tests.append(windows(s, t, fold["test_dataset"]))
            train_sids.add(f"btb{s:02d}")
    elif split == "cross_subject":
        test_subjects = {int(x) for x in d["test_subjects"]}
        listed = {s for s, _ in pairs}
        if not test_subjects <= listed:
            raise ValueError(f"test_subjects {sorted(test_subjects)} must be in subject_trials")
        test_movies = {npc.BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{s}_{t}"]
                       for s, t in pairs if s in test_subjects}
        for s, t in pairs:
            if s not in test_subjects:
                continue
            ds = npb.BrainTreebankSubjectTrialBenchmarkDataset(subjects_npb[s], t, torch.float32, **kw)
            v, te = _halves(len(ds))
            vals.append(windows(s, t, ds, v))
            tests.append(windows(s, t, ds, te))
        for s, t in npc.NEUROPROBE_LITE_SUBJECT_TRIALS:
            if s in test_subjects or s not in listed:
                continue
            movie = npc.BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{s}_{t}"]
            if d.get("exclude_test_movie", True) and movie in test_movies:
                print(f"[neuroprobe] skip train sub {s} trial {t}: same movie as a test session ({movie})")
                continue
            ds = npb.BrainTreebankSubjectTrialBenchmarkDataset(subjects_npb[s], t, torch.float32, **kw)
            trains.append(windows(s, t, ds))
            train_sids.add(f"btb{s:02d}")
    else:
        raise ValueError(f"Unknown split {split!r}")

    sfreq = next(iter(sessions.values())).meta["sfreq_out"]
    n_classes = len({int(y) for w in trains for y in np.unique(w.labels)})
    return DataBundle(ConcatDataset(trains), ConcatDataset(vals), ConcatDataset(tests),
                      infos, train_subjects=train_sids, sfreq=float(sfreq),
                      n_classes=max(2, n_classes),
                      description=f"neuroprobe {eval_name} {split} electrodes={which}")
