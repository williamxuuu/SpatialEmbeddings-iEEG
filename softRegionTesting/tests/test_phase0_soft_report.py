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
    for s, fold, seed in itertools.product(sp, ("heldout2_full", "heldout2_small", "heldout3_full", "heldout3_small"), range(2)):
        size = fold.split("_")[1]
        v = 0.60 + (gain[(s.split("_")[0], size)] if "gauss3" in s else 0)
        rows.append(dict(run=f"{s}_{fold}_{seed}", spatial=s, fold=fold, seed=str(seed), subject=fold[7],
                         auc=v, val_last=v, escape_epoch=2 if seed == 0 else 12, frac_unknown_test=0.05))
    return pd.DataFrame(rows)


def test_rules(monkeypatch, tmp_path):
    gx = {("destrieux", "full"): 0.01, ("destrieux", "half"): 0.02,
          ("schaefer400", "full"): 0.03, ("schaefer400", "half"): 0.05}
    gs = {("destrieux", "full"): 0.01, ("destrieux", "small"): -0.01,
          ("schaefer400", "full"): 0.02, ("schaefer400", "small"): 0.04}
    data = {"x": _xsess(gx), "s": _xsubj(gs)}
    monkeypatch.setattr(rep, "load_runs", lambda d: data[d[0]].copy())
    rep.main(["x", "s", "--out", str(tmp_path / "r.md")])
    t = (tmp_path / "r.md").read_text()
    line = lambda k: next(l for l in t.splitlines() if k in l)  # noqa: E731
    assert line("F0 destrieux/full").startswith("- [PASS]")
    assert ("+0.0299" in line("D1 soft - hard schaefer400/full") or "+0.0300" in line("D1 soft - hard schaefer400/full")) and "[PASS]" in line("D1 soft - hard schaefer400/full")
    assert "[PASS]" in line("D2 soft - randnbr destrieux/half")
    assert "[PASS]" in line("D3 density destrieux") and "[PASS]" in line("D3 resolution")
    # cross-subject: seed 1 escapes at epoch 12 -> 8/16 escaped -> F0 fails
    assert "[FAIL]" in line("F0 schaefer400/small") and "8/16" in line("F0 schaefer400/small")
    assert "[FAIL]" in line("D1 soft - hard destrieux/small") and "-0.0100" in line("D1 soft - hard destrieux/small")
    sub_d3 = [l for l in t.splitlines() if "D3 density" in l]
    assert "[FAIL]" in sub_d3[2] and "[PASS]" in sub_d3[3]   # destrieux small<full, schaefer small>full
    assert "[PASS]" in line("D4 equal coverage schaefer400/small")


def test_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(rep, "load_runs", lambda d: pd.DataFrame())
    rep.main(["x", "s", "--out", str(tmp_path / "r.md")])
    assert (tmp_path / "r.md").read_text().count("No runs yet") == 2
