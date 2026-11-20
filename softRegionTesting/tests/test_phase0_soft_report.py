"""Experiment 2 Phase 0 scorer on synthetic runs with a known effect."""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase0_soft_report as rep  # noqa: E402

SUBJ = ["btb01", "btb02", "btb03", "btb04", "btb07", "btb10"]


def _xsess(gain):
    rows, rng = [], np.random.default_rng(0)
    sp = ["none"] + [f"{a}_{k}" for a in rep.ATLASES for k in ("hard", "gauss3", "gauss3_randnbr")]
    for s, dens, seed in itertools.product(sp, ("full", "half"), range(3)):
        v = 0.70 + (gain[(s.split("_")[0], dens)] if s.endswith("gauss3") else 0) + rng.normal(0, 1e-4)
        for subj in SUBJ:
            rows.append(dict(run=f"{s}_{dens}_{seed}", spatial=s, density=dens, seed=str(seed),
                             subject=subj, auc=v, val_last=v, escape_epoch=3))
    return pd.DataFrame(rows)


def _xsubj(gain):
    rows = []
    sp = [f"{a}_{k}" for a in rep.ATLASES for k in ("hard", "gauss3", "hard_nearest", "gauss3_nearest")]
    folds = ["ho01", "ho02", "ho03", "ho04", "ho07", "ho10"]
    for s, fold, dens, seed in itertools.product(sp, folds, ("k2", "k5"), range(2)):
        v = 0.60 + (gain[(s.split("_")[0], dens)] if "gauss3" in s else 0)
        rows.append(dict(run=f"{s}_{fold}_{dens}_{seed}", spatial=s, fold=fold, density=dens, seed=str(seed),
                         subject="btb" + fold[2:], auc=v, val_last=v,
                         escape_epoch=2 if seed == 0 else 12, frac_unknown_test=0.05))
    return pd.DataFrame(rows)


def test_rules(monkeypatch, tmp_path):
    gx = {("schaefer400", "full"): 0.01, ("schaefer400", "half"): 0.02,
          ("schaefer1000", "full"): 0.03, ("schaefer1000", "half"): 0.05}
    gs = {("schaefer400", "k5"): 0.01, ("schaefer400", "k2"): -0.01,
          ("schaefer1000", "k5"): 0.02, ("schaefer1000", "k2"): 0.04}
    data = {"x": _xsess(gx), "s": _xsubj(gs)}
    monkeypatch.setattr(rep, "load_runs", lambda d: data[d[0]].copy())
    rep.main(["x", "s", "--out", str(tmp_path / "r.md")])
    t = (tmp_path / "r.md").read_text()
    line = lambda k: next(l for l in t.splitlines() if k in l)  # noqa: E731
    assert line("F0 schaefer400/full").startswith("- [PASS]")
    d1 = line("D1 soft - hard schaefer1000/full")
    assert "[PASS]" in d1 and ("+0.0299" in d1 or "+0.0300" in d1 or "+0.0301" in d1)
    assert "[PASS]" in line("D2 soft - randnbr schaefer400/half")
    assert "[PASS]" in line("D3 density schaefer400") and "[PASS]" in line("D3 resolution")
    # cross-subject: seed 1 escapes at epoch 12 -> 50% escaped -> F0 fails (needs 75%)
    f0 = line("F0 schaefer1000/k2")
    assert "[FAIL]" in f0 and "24/48" in f0
    d1s = line("D1 soft - hard schaefer400/k2")
    assert "[FAIL]" in d1s and "-0.0100" in d1s
    assert "[PASS]" in line("D1 soft - hard schaefer1000/k5")
    assert "[n/a]" in [l for l in t.splitlines() if "D3 density schaefer400" in l][1]
    assert "[PASS]" in line("D4 equal coverage schaefer1000/k2")


def test_xsubj_k5_only(monkeypatch, tmp_path):
    gs = {("schaefer400", "k5"): 0.02, ("schaefer1000", "k5"): -0.02,
          ("schaefer400", "k2"): 0.0, ("schaefer1000", "k2"): 0.0}
    xs = _xsubj(gs)
    xs = xs[xs["density"] == "k5"].drop(columns="density")
    data = {"x": pd.DataFrame(), "s": xs}
    monkeypatch.setattr(rep, "load_runs", lambda d: data[d[0]].copy())
    rep.main(["x", "s", "--out", str(tmp_path / "r.md")])
    t = (tmp_path / "r.md").read_text()
    d1 = next(l for l in t.splitlines() if "D1 soft - hard schaefer400/k5" in l)
    assert "[PASS]" in d1 and "sign-flip p = " in d1 and ", 0.031 over people" in d1
    assert "[FAIL]" in next(l for l in t.splitlines() if "D1 soft - hard schaefer1000/k5" in l)
    assert "D3 density" not in t and "/k2" not in t


def test_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(rep, "load_runs", lambda d: pd.DataFrame())
    rep.main(["x", "s", "--out", str(tmp_path / "r.md")])
    assert (tmp_path / "r.md").read_text().count("No runs yet") == 2
