"""Tests for the post-pilot fixes: normalization modes, Laplacian re-referencing,
spectral front end, mean readout, electrode-ID control, coordinate jitter,
min_epochs / select=last, the location probe and saved predictions."""

import json

import numpy as np
import pytest
import torch

from locemb.config import load_config
from locemb.data.base import SubjectInfo
from locemb.data.neuroprobe_adapter import laplacian_mix
from locemb.data.preprocess import WindowOps, zscore_mode
from locemb.models.transformer import PopulationTransformer
from locemb.spatial import build_assignment
from locemb.spatial.encodings import NoSpatial
from locemb.train import run


def test_zscore_modes_and_backcompat():
    assert zscore_mode({"zscore": True}) == "window"
    assert zscore_mode({}) == "window"
    assert zscore_mode({"zscore": "session"}) == "session"
    assert zscore_mode({"zscore": False}) == "none"
    with pytest.raises(ValueError):
        zscore_mode({"zscore": "bogus"})


def test_session_zscore_keeps_power_changes():
    # Window-level z-scoring erases a 3x amplitude rise; session-level keeps it.
    rng = np.random.default_rng(0)
    quiet, loud = rng.normal(size=(2, 512)), 3 * rng.normal(size=(2, 512))
    w = WindowOps({"zscore": "window", "clip": None})
    assert abs(w(loud).std().item() - w(quiet).std().item()) < 0.05
    s = WindowOps({"zscore": "session", "clip": None})
    stats = (np.zeros(2, np.float32), np.ones(2, np.float32))
    assert s(loud, stats).std().item() > 2.5 * s(quiet, stats).std().item()
    with pytest.raises(ValueError):
        s(loud)  # session mode without stats is an error, not a silent no-op


def test_laplacian_mix_uses_shaft_neighbours():
    avail = ["A1", "A2", "A3", "A5", "B1", "B2", "C7"]
    read, mix, n_none = laplacian_mix(["A2", "A5", "B1", "C7"], avail)
    assert read[:4] == ["A2", "A5", "B1", "C7"]          # model channels first, in order
    x = {e: np.full(4, float(i + 1)) for i, e in enumerate(read)}
    out = mix @ np.stack([x[e] for e in read])
    v = {e: x[e][0] for e in read}
    assert np.isclose(out[0, 0], v["A2"] - (v["A1"] + v["A3"]) / 2)   # two neighbours
    assert np.isclose(out[1, 0], v["A5"])                             # A4, A6 missing -> left as is
    assert np.isclose(out[2, 0], v["B1"] - v["B2"])                   # one neighbour -> bipolar
    assert np.isclose(out[3, 0], v["C7"])
    assert n_none == 2
    assert np.allclose(mix.sum(1), [0, 1, 0, 1])          # referenced rows sum to 0


def _batch(B=3, E=5, T=512):
    return {"x": torch.randn(B, E, T), "mask": torch.tensor([[1] * E, [1] * 3 + [0] * 2, [1] * E], dtype=torch.bool)}


@pytest.mark.parametrize("frontend,readout", [("cnn", "cls"), ("stft", "cls"), ("stft", "mean"), ("cnn", "mean")])
def test_transformer_frontends(frontend, readout):
    m = PopulationTransformer(NoSpatial(32), 2, d=32, layers=1, heads=2, frontend=frontend,
                              readout=readout, sfreq=512.0, n_samples=512)
    b = _batch()
    out = m(b)
    assert out.shape == (3, 2) and torch.isfinite(out).all()
    # padded electrodes must not change the output
    b2 = dict(b); b2["x"] = b["x"].clone(); b2["x"][1, 3:] = 1e3
    m.eval()
    assert torch.allclose(m(b)[1], m(b2)[1], atol=1e-4)


def test_stft_front_end_sees_power_level():
    from locemb.models.transformer import SpectralPatchEncoder
    enc = SpectralPatchEncoder(16, 4, 512.0, 512)
    x = torch.randn(2, 512)
    spec = lambda z: torch.fft.rfft(z[:, :128] * enc.win, dim=-1).abs().pow(2)[..., enc.keep]
    assert torch.allclose(spec(3 * x), 9 * spec(x), rtol=1e-4)   # no per-token normalisation
    assert enc.keep.sum() == 50                                   # 4 Hz bins, 4..200 Hz


def _subjects():
    rng = np.random.default_rng(1)
    return {f"s{i}": SubjectInfo(f"s{i}", rng.uniform(-60, 60, (6, 3)), [f"E{j}" for j in range(6)])
            for i in range(3)}


def test_electrode_id_control():
    subs = _subjects()
    a = build_assignment({"type": "electrode_id"}, subs, {"s0", "s1"})
    assert a.n_regions == 1 + 12                           # one row per training electrode + unknown
    assert (a.table["s2"]["idx"][:, 0] == 0).all()          # unseen person: all <unknown>
    assert len(set(a.table["s0"]["idx"][:, 0].tolist())) == 6
    assert a.stats["type"] == "electrode_id"


def test_jitter_moves_coordinates_deterministically():
    subs = _subjects()
    a0 = build_assignment({"type": "coords"}, subs, {"s0"}, seed=0)
    a1 = build_assignment({"type": "coords", "jitter_mm": 2.0}, subs, {"s0"}, seed=0)
    a2 = build_assignment({"type": "coords", "jitter_mm": 2.0}, subs, {"s0"}, seed=0)
    d = (a1.table["s0"]["coords"] - a0.table["s0"]["coords"]).numpy()
    assert 0.5 < d.std() < 4 and torch.equal(a1.table["s0"]["coords"], a2.table["s0"]["coords"])
    assert np.allclose(subs["s0"].coords, a0.table["s0"]["coords"].numpy(), atol=1e-5)  # originals untouched


def test_min_epochs_select_last_probe_and_preds(tmp_path):
    cfg = load_config("configs/smoke.yaml", {
        "results_dir": str(tmp_path), "run_name": "fx", "train.epochs": 3, "train.patience": 1,
        "train.min_epochs": 3, "train.min_delta": 1.0, "train.select": "last",
        "spatial.type": "region", "spatial.atlas": "grid40", "synthetic.n_subjects": 4})
    m = run(cfg)
    assert m["epochs_run"] == 3                            # min_epochs beats patience=1
    assert m["select"] == "last"
    assert m["test_auc"] == m["test_last"]["auc"]
    assert m["location_probe"] is not None and "drop_mean_subject_auc" in m["location_probe"]
    p = np.load(tmp_path / "smoke" / "fx" / "preds.npz")
    assert len(p["y"]) == m["test_n"] == len(p["p_last"]) == len(p["subject"])
    json.dumps(m)  # serialisable
