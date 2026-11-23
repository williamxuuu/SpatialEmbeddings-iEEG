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
import re
import zlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from .base import DataBundle, SubjectInfo
from .preprocess import WindowOps, filter_spec, zscore_mode
from .session_cache import SessionData, build_session, default_cache_dir


class CachedWindows(Dataset):
    """Windows of one cached recording.

    `read_rows` are the cache rows read for each window; `mix` (E_out x E_read,
    or None = identity) turns them into the channels the model sees. That is how
    Laplacian re-referencing reaches shaft neighbours the model itself may not
    see (e.g. a Lite electrode whose neighbour is outside the Lite set).
    `stats` = per-channel (mean, sd) over the whole recording, for zscore=session.
    """

    def __init__(self, session: SessionData, read_rows: np.ndarray, starts: np.ndarray,
                 labels: np.ndarray, n_out: int, subject: str, ops: WindowOps,
                 mix: np.ndarray | None = None, stats=None):
        self.session, self.rows, self.starts, self.labels = session, read_rows, starts, labels
        self.n_out, self.subject, self.ops = n_out, subject, ops
        self.mix, self.stats = mix, stats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        a = self.session.window(int(self.starts[i]), self.n_out, self.rows)
        if self.mix is not None:
            a = self.mix @ a
        return {"x": self.ops(a, self.stats), "y": int(self.labels[i]), "subject": self.subject}


class SparseMix:
    """A re-referencing matrix with few non-zeros per row (Laplacian: <= 3),
    applied by indexing instead of a BLAS matrix product. Inside DataLoader
    workers a dense `mix @ a` starts one BLAS thread pool per worker (7 workers x
    8 threads on 8 cores in the smoke test), which made each window ~20x slower."""

    def __init__(self, mix: np.ndarray):
        k = max(1, int((mix != 0).sum(1).max()))
        self.cols = np.zeros((mix.shape[0], k), dtype=np.int64)
        self.w = np.zeros((mix.shape[0], k), dtype=np.float32)
        for i, row in enumerate(mix):
            nz = np.nonzero(row)[0]
            self.cols[i, :len(nz)], self.w[i, :len(nz)] = nz, row[nz]
        self.shape = mix.shape

    def __matmul__(self, a: np.ndarray) -> np.ndarray:
        return (a[self.cols] * self.w[..., None]).sum(1)


_CONTACT = re.compile(r"^(.*?)(\d+)$")


def laplacian_mix(labels: list[str], available: list[str]) -> tuple[list[str], np.ndarray, int]:
    """Shaft Laplacian: each contact minus the mean of its neighbours (contact
    number +-1 on the same shaft, e.g. F3aOF5 and F3aOF7 for F3aOF6) among the
    `available` (usable) contacts. A contact with only one usable neighbour gets
    a bipolar difference with it; with none it is left as is (counted, logged).
    Neuroprobe's own Laplacian drops such contacts instead; we keep them so the
    electrode set (and so the spatial codes) is the same in every preprocessing arm.

    Returns (labels to read, mix matrix (len(labels) x len(read)), n_unreferenced).
    """
    avail = set(available)
    read = list(labels)
    pos = {e: i for i, e in enumerate(read)}
    rows = []
    n_none = 0
    for e in labels:
        m = _CONTACT.match(e)
        nbrs = []
        if m:
            shaft, num = m.group(1), int(m.group(2))
            nbrs = [f"{shaft}{num + d}" for d in (-1, 1) if f"{shaft}{num + d}" in avail]
        for n in nbrs:
            if n not in pos:
                pos[n] = len(read); read.append(n)
        rows.append((pos[e], [pos[n] for n in nbrs]))
        n_none += not nbrs
    mix = np.zeros((len(labels), len(read)), dtype=np.float32)
    for i, (me, nb) in enumerate(rows):
        mix[i, me] = 1.0
        for j in nb:
            mix[i, j] -= 1.0 / len(nb)
    return read, mix, n_none


def session_stats(sd: SessionData, rows: np.ndarray, mix: np.ndarray | None, ops: WindowOps,
                  n_out: int, n_windows: int = 256, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean / SD of the re-referenced signal over one whole recording,
    estimated from `n_windows` windows at fixed random positions (no labels used).
    For the test recording this is unlabeled test-session data, as in
    Neuroprobe's "global z-scoring" baselines."""
    rng = np.random.default_rng(seed)
    T = sd.n_samples
    starts = rng.integers(0, max(1, T - n_out), size=n_windows)
    s1 = s2 = None
    n = 0
    for j0 in np.sort(starts):
        a = sd.window_out(int(j0), n_out, rows).astype(np.float64)
        if mix is not None:
            a = mix @ a
        if ops.reref == "car":
            a = a - a.mean(0, keepdims=True)
        s1 = a.sum(1) if s1 is None else s1 + a.sum(1)
        s2 = (a ** 2).sum(1) if s2 is None else s2 + (a ** 2).sum(1)
        n += a.shape[1]
    mu = s1 / n
    var = np.maximum(s2 / n - mu ** 2, 0)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)


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


def _val_people(d: dict, split: str, npc) -> list[int]:
    """data.val_subjects (cross_subject only): people held out of training whose
    recording (their trial in subject_trials) is the validation set that picks the
    epoch. Validation is then cross-person, like the test, and uses no test labels.

      null / []  old behaviour: validation = first half of the test recording
      auto       the next person after the (single) test person in subject_trials
                 order, cyclic, skipping anyone whose recording is the same movie as
                 the test person's (e.g. 4 and 7 both have megamind):
                 1->2, 2->3, 3->4, 4->10, 7->10, 10->1 for the Lite list
      [s, ...]   these people
    """
    v = d.get("val_subjects")
    if v in (None, [], "none"):
        return []
    if split != "cross_subject":
        raise ValueError("data.val_subjects only applies to split=cross_subject")
    if v != "auto":
        return [int(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
    order = [int(s) for s, _ in d["subject_trials"]]
    trial = {int(s): int(t) for s, t in d["subject_trials"]}
    test = [int(x) for x in d["test_subjects"]]
    if len(test) != 1:
        raise ValueError("data.val_subjects=auto needs exactly one test subject")
    movie = lambda s: npc.BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{s}_{trial[s]}"]
    i = order.index(test[0])
    for j in range(1, len(order)):
        s = order[(i + j) % len(order)]
        if movie(s) != movie(test[0]):
            return [s]
    raise ValueError("no validation person with a different movie")


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
    # Data-density knob: keep a fixed random fraction of each person's electrodes.
    # The subset depends only on (subject, electrode_subset_seed), never on the
    # training seed or the spatial setting, so every encoding sees the same electrodes.
    efrac = float(d.get("electrode_frac", 1.0))
    if not 0.0 < efrac <= 1.0:
        raise ValueError(f"data.electrode_frac must be in (0, 1], not {efrac}")
    esub_seed = int(d.get("electrode_subset_seed", 0))
    # Training-only density knob (cross-subject): keep a fixed random fraction of each
    # TRAINING person's electrodes; the held-out person keeps all of theirs. Thins
    # electrodes per region while the number of people stays fixed. The subset depends
    # only on (person, train_electrode_seed), never on the spatial setting.
    tfrac = float(d.get("train_electrode_frac", 1.0))
    if not 0.0 < tfrac <= 1.0:
        raise ValueError(f"data.train_electrode_frac must be in (0, 1], not {tfrac}")
    if tfrac < 1.0 and split != "cross_subject":
        raise ValueError("data.train_electrode_frac only applies to split=cross_subject")
    held_out = {int(x) for x in d.get("test_subjects", [])} if split == "cross_subject" else set()
    val_people = _val_people(d, split, npc)
    held_out |= set(val_people)   # validation people keep all their electrodes, like test people
    tsub_seed = int(d.get("train_electrode_seed", 0))
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
            sd = SessionData(build_session(subjects_npb[s], t, spec, cache_dir, root))
            if d.get("prefetch", True):
                print(f"[neuroprobe] prefetched sub {s} trial {t} in {sd.prefetch():.1f}s", flush=True)
            sessions[(s, t)] = sd
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
            if efrac < 1.0:
                rng = np.random.default_rng(zlib.crc32(f"elec_{s}_{esub_seed}".encode()))
                n_keep = max(1, int(round(efrac * len(labels))))
                keep = set(rng.choice(len(labels), size=n_keep, replace=False).tolist())
                labels = [e for i, e in enumerate(labels) if i in keep]   # original order
            if tfrac < 1.0 and s not in held_out:
                rng = np.random.default_rng(zlib.crc32(f"telec_{s}_{tsub_seed}".encode()))
                n_keep = max(1, int(round(tfrac * len(labels))))
                keep = set(rng.choice(len(labels), size=n_keep, replace=False).tolist())
                labels = [e for i, e in enumerate(labels) if i in keep]
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

    prep = cfg.get("preprocess") or {}
    reref = prep.get("reref", "none")
    zmode = zscore_mode(prep)
    read_plan: dict[tuple[int, int], tuple] = {}
    stats_cache: dict[tuple[int, int], tuple] = {}
    n_unref: dict[str, int] = {}

    def plan(s: int, t: int):
        """(cache rows to read, mix matrix or None) for subject s, recording t."""
        if (s, t) not in read_plan:
            sd = session(s, t)
            labels = elec[s]
            if reref == "laplacian":
                read, mix, n0 = laplacian_mix(labels, sd.labels)
                n_unref[f"btb{s:02d}"] = n0
                read_plan[(s, t)] = (sd.rows(read), SparseMix(mix))
            else:
                read_plan[(s, t)] = (sd.rows(labels), None)
        return read_plan[(s, t)]

    def windows(s: int, t: int, ds, idx: np.ndarray | None = None) -> CachedWindows:
        starts, labels, n_raw = _indices(ds)
        if idx is not None:
            starts, labels = starts[idx], labels[idx]
        sd = session(s, t)
        info = subject_info(s)
        n_out = int(round(n_raw * sd.ratio))
        rows, mix = plan(s, t)
        stats = None
        if zmode == "session":
            if (s, t) not in stats_cache:
                stats_cache[(s, t)] = session_stats(sd, rows, mix, ops, n_out,
                                                    seed=zlib.crc32(f"{s}_{t}".encode()))
            stats = stats_cache[(s, t)]
        return CachedWindows(sd, rows, starts, labels, n_out, info.subject_id, ops, mix, stats)

    trains, vals, tests, monitors, train_sids = [], [], [], [], set()
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
        if test_subjects & set(val_people):
            raise ValueError(f"data.val_subjects {val_people} overlaps test_subjects {sorted(test_subjects)}")
        # Movies of the test AND validation recordings: training sessions of these movies
        # are dropped, so neither held-out person shares a stimulus with training.
        test_movies = {npc.BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{s}_{t}"]
                       for s, t in pairs if s in test_subjects or s in val_people}
        for s, t in pairs:
            if s not in test_subjects:
                continue
            ds = npb.BrainTreebankSubjectTrialBenchmarkDataset(subjects_npb[s], t, torch.float32, **kw)
            v, te = _halves(len(ds))
            if val_people:
                # Model selection uses the validation person; the first half of the test
                # recording is still evaluated every epoch (monitor), never used to select.
                monitors.append(windows(s, t, ds, v))
            else:
                vals.append(windows(s, t, ds, v))
            tests.append(windows(s, t, ds, te))
        for s, t in pairs:
            if s in val_people:
                ds = npb.BrainTreebankSubjectTrialBenchmarkDataset(subjects_npb[s], t, torch.float32, **kw)
                vals.append(windows(s, t, ds))
                print(f"[neuroprobe] validation person {s} trial {t} ({len(ds)} windows); not trained on")
        def _excluded(s, t):
            movie = npc.BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{s}_{t}"]
            return d.get("exclude_test_movie", True) and movie in test_movies

        # Data-density knob: train on only k of the eligible people. Subsets are nested
        # (one fixed random order per held-out set and train_subset_seed; take the first
        # k), so k=1 is inside k=2, which is inside k=5, and the same subset is used for
        # every spatial setting.
        eligible = sorted({s for s, t in npc.NEUROPROBE_LITE_SUBJECT_TRIALS
                           if s in listed and s not in test_subjects and s not in val_people
                           and not _excluded(s, t)})
        chosen = set(eligible)
        k = d.get("n_train_subjects")
        if k is not None:
            k = int(k)
            if not 1 <= k <= len(eligible):
                raise ValueError(f"data.n_train_subjects={k} but only {len(eligible)} eligible "
                                 f"training people: {eligible}")
            key = f"train_{sorted(test_subjects)}_{int(d.get('train_subset_seed', 0))}"
            order = np.random.default_rng(zlib.crc32(key.encode())).permutation(eligible)
            chosen = {int(x) for x in order[:k]}
            print(f"[neuroprobe] training people (k={k}): {sorted(chosen)} of {eligible}")
        for s, t in npc.NEUROPROBE_LITE_SUBJECT_TRIALS:
            if s in test_subjects or s in val_people or s not in listed or s not in chosen:
                continue
            if _excluded(s, t):
                movie = npc.BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{s}_{t}"]
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
                      monitor=ConcatDataset(monitors) if monitors else None,
                      val_subjects=[f"btb{s:02d}" for s in val_people],
                      description=f"neuroprobe {eval_name} {split} electrodes={which} frac={efrac:g} train_frac={tfrac:g} "
                                  f"reref={reref} zscore={zmode}"
                                  + (f" laplacian_unreferenced={n_unref}" if n_unref else ""))
