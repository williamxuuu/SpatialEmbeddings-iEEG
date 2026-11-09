"""Filter each whole Brain Treebank recording once and save it to disk.

Why: filtering 1-second windows one at a time distorts low frequencies and
window edges (see data/preprocess.py). Filtering the whole recording avoids
that, and it is also faster: each recording is filtered once, not once per
window per epoch per run.

What is saved, per (subject, trial, filter settings):

    <cache_dir>/<key>/sub_<s>_trial<ttt>/
        data.npy     float16 (n_electrodes, n_samples), memory-mapped on read
        scales.npy   float32 (n_electrodes,)  -- data.npy * scale = filtered signal
        meta.json    electrode labels (row order), sample rates, filter settings

Every usable electrode is saved (Neuroprobe's own filter: no corrupted, trigger
or unlocalized contacts), so one cache serves both the "lite" (<=120
electrodes) and "all electrodes" settings.

Each electrode is divided by its whole-recording SD before the float16 cast,
so float16 keeps ~3 significant digits whatever the units are. The scale is
multiplied back on read, so CAR across electrodes still sees the original
relative amplitudes.

Build ahead of time (recommended; a CPU job):
    python scripts/build_session_cache.py --config configs/neuroprobe.yaml
Otherwise the first run that needs a recording builds it. A directory lock
stops two jobs building the same recording at once.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

import numpy as np

from .preprocess import SessionFilter

CACHE_VERSION = 1
RAW_SFREQ = 2048.0


def cache_key(spec: dict) -> str:
    blob = json.dumps({"v": CACHE_VERSION, **spec}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def session_dir(cache_dir: str, spec: dict, subject_id: int, trial_id: int) -> str:
    return os.path.join(cache_dir, cache_key(spec), f"sub_{subject_id}_trial{trial_id:03d}")


def default_cache_dir(root: str | None) -> str:
    if os.environ.get("LOCEMB_CACHE_DIR"):
        return os.environ["LOCEMB_CACHE_DIR"]
    if not root:
        raise ValueError("Set data.cache_dir, $LOCEMB_CACHE_DIR or data.root")
    return os.path.join(os.path.dirname(os.path.abspath(root)), "locemb_cache")


def build_session(subject, trial_id: int, spec: dict, cache_dir: str, root: str,
                  log=print) -> str:
    """Filter every usable electrode of one recording and write it to the cache.

    `subject` is a neuroprobe.BrainTreebankSubject (only its electrode list and
    h5 key mapping are used). Returns the session directory.
    """
    import h5py

    out = session_dir(cache_dir, spec, subject.subject_id, trial_id)
    if os.path.exists(os.path.join(out, "meta.json")):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lock = out + ".lock"
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            if os.path.exists(os.path.join(out, "meta.json")):
                return out
            if time.time() - os.path.getmtime(lock) > 6 * 3600:
                log(f"[cache] removing stale lock {lock}")
                shutil.rmtree(lock, ignore_errors=True)
                continue
            log(f"[cache] waiting for another job to build {out}")
            time.sleep(30)
    try:
        if os.path.exists(os.path.join(out, "meta.json")):
            return out
        tmp = f"{out}.tmp{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        labels = list(subject.electrode_labels)
        filt = SessionFilter(spec, RAW_SFREQ)
        h5 = os.path.join(root, f"sub_{subject.subject_id}_trial{trial_id:03d}.h5")
        t0 = time.time()
        with h5py.File(h5, "r") as f:
            keys = [subject.h5_neural_data_keys[e] for e in labels]
            n_in = f["data"][keys[0]].shape[0]
            n_out = filt.out_len(n_in)
            arr = np.lib.format.open_memmap(os.path.join(tmp, "data.npy"), mode="w+",
                                            dtype=np.float16, shape=(len(labels), n_out))
            scales = np.ones(len(labels), dtype=np.float32)
            for i, k in enumerate(keys):
                y = filt(f["data"][k][:])
                if len(y) != n_out:
                    raise RuntimeError(f"{k}: expected {n_out} samples after filtering, got {len(y)}")
                s = float(np.std(y)) or 1.0
                scales[i] = s
                arr[i] = (y / s).astype(np.float16)
                if i % 20 == 0:
                    log(f"[cache] sub {subject.subject_id} trial {trial_id}: "
                        f"{i + 1}/{len(keys)} electrodes ({time.time() - t0:.0f}s)")
            arr.flush()
            del arr
        np.save(os.path.join(tmp, "scales.npy"), scales)
        meta = {"subject_id": int(subject.subject_id), "trial_id": int(trial_id),
                "electrode_labels": labels, "sfreq_in": RAW_SFREQ, "sfreq_out": filt.sfreq_out,
                "n_in": int(n_in), "n_out": int(n_out), "spec": spec, "version": CACHE_VERSION,
                "build_sec": round(time.time() - t0, 1)}
        with open(os.path.join(tmp, "meta.json"), "w") as fh:
            json.dump(meta, fh)
        os.replace(tmp, out)
        log(f"[cache] wrote {out} ({len(labels)} electrodes, {n_out} samples, "
            f"{time.time() - t0:.0f}s)")
        return out
    finally:
        shutil.rmtree(lock, ignore_errors=True)


class SessionData:
    """Read windows from one cached recording. Opens the file lazily, so it is
    safe to send to DataLoader worker processes."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.labels = self.meta["electrode_labels"]
        self.row = {e: i for i, e in enumerate(self.labels)}
        self.ratio = self.meta["sfreq_out"] / self.meta["sfreq_in"]
        self._arr = None
        self._scales = None

    def __getstate__(self):
        st = dict(self.__dict__)
        st["_arr"] = None
        st["_scales"] = None
        return st

    def _open(self):
        if self._arr is None:
            self._arr = np.load(os.path.join(self.path, "data.npy"), mmap_mode="r")
            self._scales = np.load(os.path.join(self.path, "scales.npy"))

    def rows(self, labels: list[str]) -> np.ndarray:
        missing = [e for e in labels if e not in self.row]
        if missing:
            raise KeyError(f"{len(missing)} electrodes not in cache {self.path}: {missing[:5]}")
        return np.array([self.row[e] for e in labels])

    def window(self, start_raw: int, n_out: int, rows: np.ndarray) -> np.ndarray:
        """`start_raw` is a sample index at the raw 2048 Hz rate (what Neuroprobe
        returns); `n_out` is the window length at the cached rate."""
        self._open()
        j0 = int(round(start_raw * self.ratio))
        j0 = max(0, min(j0, self._arr.shape[1] - n_out))
        a = np.asarray(self._arr[np.sort(rows), j0:j0 + n_out], dtype=np.float32)
        order = np.argsort(np.argsort(rows))
        return a[order] * self._scales[rows][:, None]
