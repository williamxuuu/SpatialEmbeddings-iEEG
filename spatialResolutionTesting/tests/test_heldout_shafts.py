"""Held-out-shaft split helpers and report (2026-11-27)."""
import json
import os
import sys

import numpy as np

from locemb.data.neuroprobe_adapter import laplacian_mix, pick_shafts, shaft_of

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _labels():
    return [f"{sh}{i}" for sh in ("F3aOF", "T1aIb", "LT2aA", "LF3cIc", "T2bH", "T1cIf", "P1x", "O2y")
            for i in range(1, 9)]


def test_shaft_of():
    assert shaft_of("F3aOF6") == "F3aOF"
    assert shaft_of("LT2aA12") == "LT2aA"
    assert shaft_of("REF") == "REF"


def test_pick_shafts_whole_deterministic_never_all():
    L = _labels()
    a = pick_shafts(L, 0.25, "hshaft_1_0")
    assert a == pick_shafts(L, 0.25, "hshaft_1_0") and len(a) == 2
    assert pick_shafts(L, 0.25, "hshaft_1_1") != a or pick_shafts(L, 0.25, "hshaft_1_2") != a
    assert len(pick_shafts(L, 0.99, "x")) == 7 and len(pick_shafts(L, 0.01, "x")) == 1
    hidden = [e for e in L if shaft_of(e) in a]
    train = [e for e in L if shaft_of(e) not in a]
    assert not ({shaft_of(e) for e in hidden} & {shaft_of(e) for e in train})


def test_laplacian_never_reads_hidden_shaft():
    L = _labels()
    hid = pick_shafts(L, 0.25, "hshaft_3_0")
    train = [e for e in L if shaft_of(e) not in hid]
    read, mix, _ = laplacian_mix(train, L)          # all session contacts available
    used = [read[j] for j in np.nonzero(np.abs(mix).sum(0))[0]]
    assert not any(shaft_of(e) in hid for e in used)


def test_xheld_report_on_fake_runs(tmp_path):
    import main_report as mr
    rng = np.random.default_rng(0)
    subs = ["btb01", "btb02", "btb03", "btb04", "btb07", "btb10"]
    for dens, k in (("s25", 0.25), ("s100", 1.0)):
        for sp, h in (("none", None), ("grid80", 80), ("grid40", 40), ("grid20", 20), ("grid10", 10), ("grid5", 5),
                      ("schaefer400_hard", None), ("schaefer400_gauss3", None)):
            for split in range(2):
                for seed in range(2):
                    gain = 0 if h is None else 0.05 - 0.01 * (np.log2(h) - np.log2(20 / k)) ** 2
                    per = {f"{s}h": float(0.6 + gain + rng.normal(0, 0.01)) for s in subs}
                    tests = {"train_shafts": {"per_subject_auc": {s: 0.8 for s in subs}},
                             "all_shafts": {"per_subject_auc": {f"{s}a": 0.78 for s in subs}}}
                    if sp != "none":
                        tests["heldout_unknown"] = {"per_subject_auc": {k2: v - 0.02 for k2, v in per.items()}}
                        tests["heldout_shuffled_cells"] = {"per_subject_auc": {k2: v - 0.03 for k2, v in per.items()}}
                    name = f"density-{dens}__spatial-{sp}__split-{split}__seed-{seed}"
                    d = tmp_path / name
                    d.mkdir()
                    json.dump({"exp_name": "heldout_shafts", "run_name": name, "seed": seed,
                               "test_per_subject_auc": per, "test_best": {"per_subject_auc": per},
                               "test_last": {"per_subject_auc": per}, "history": [], "spatial": {},
                               "heldout": {"tests": tests,
                                           "coverage": {f"{s}h": (1.0 if h is None else min(1, h / 40)) for s in subs},
                                           "info": {f"{s}h": {"n_shared_shafts": 0,
                                                              "dist_nearest_own_train_mm_median": 8.0 + rng.normal()}
                                                    for s in subs}}},
                              open(d / "metrics.json", "w"))
    text = mr.xheld([str(tmp_path)], str(tmp_path / "rep.md"))
    assert "Leak check" in text and "OK" in text
    assert "true - shuffled_cells = +0.0300" in text
    assert "R1 analysis failed" not in text, text
