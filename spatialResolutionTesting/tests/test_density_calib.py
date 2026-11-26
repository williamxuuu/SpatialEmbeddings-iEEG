"""train_window_subset (within-person density knob) and calibrate (frozen model,
refit linear readout on a new person)."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from locemb.data.neuroprobe_adapter import N_TIME_CHUNKS, train_window_subset
from locemb.train import calibrate


def test_window_subset_contiguous_and_deterministic():
    starts = np.random.default_rng(0).permutation(np.arange(1000) * 512)   # shuffled order
    a = train_window_subset(starts, 0.1, "twin_1_2_0")
    assert len(a) == 100 and np.array_equal(a, train_window_subset(starts, 0.1, "twin_1_2_0"))
    # whole chunks: kept start times form round(0.1 * N_TIME_CHUNKS) contiguous runs
    t = np.sort(starts[a]) // 512
    assert (np.diff(t) > 1).sum() + 1 <= round(0.1 * N_TIME_CHUNKS)
    assert not np.array_equal(a, train_window_subset(starts, 0.1, "twin_1_2_1"))
    assert len(train_window_subset(starts, 1.0, "x")) == 1000
    assert len(train_window_subset(starts, 0.001, "x")) == 1000 // N_TIME_CHUNKS   # at least one chunk


class _Feat(torch.nn.Module):
    def features(self, b):
        return b["x"]


def _loader(X, y, sub):
    items = [{"x": torch.tensor(X[i]), "y": torch.tensor(int(y[i])), "subject": sub} for i in range(len(y))]
    coll = lambda B: {"x": torch.stack([b["x"] for b in B]), "y": torch.stack([b["y"] for b in B]),
                      "subject": [b["subject"] for b in B]}
    return DataLoader(items, batch_size=32, collate_fn=coll)


def test_calibrate_learns_separable_and_chance_on_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    X = rng.normal(size=(400, 8)).astype(np.float32); X[:, 0] += 3 * y
    out = calibrate(_Feat(), _loader(X[:200], y[:200], "s1"), _loader(X[200:], y[200:], "s1"),
                    torch.device("cpu"), [0.25, 1.0])
    assert out["1"]["auc"] > 0.95 and out["0.25"]["n_calib"]["s1"] == 50
    Xn = rng.normal(size=(400, 8)).astype(np.float32)
    out = calibrate(_Feat(), _loader(Xn[:200], y[:200], "s1"), _loader(Xn[200:], y[200:], "s1"),
                    torch.device("cpu"), [1.0])
    assert 0.35 < out["1"]["auc"] < 0.65
