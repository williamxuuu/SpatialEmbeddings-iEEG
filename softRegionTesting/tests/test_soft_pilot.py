"""Experiment 2 pilot: sweep files, the report's rules, and (slow) real sweep
lines on the fake Brain Treebank folder."""
import json
import os
import subprocess
import sys

import pytest
import yaml

from conftest import SRT
from locemb.sweep import expand

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT = os.path.join(HERE, "soft_report.py")


def _sweep(name):
    return expand(yaml.safe_load(open(os.path.join(HERE, "configs", "sweeps", f"{name}.yaml"))))


@pytest.mark.parametrize("name,n", [("pilot_soft", 40), ("pilot_soft_xsubj", 50), ("exp2_soft_regions", 117)])
def test_sweeps_expand(name, n):
    jobs = _sweep(name)
    assert len(jobs) == n and len({j["overrides"]["run_name"] for j in jobs}) == n
    for j in jobs:
        assert os.path.exists(os.path.join(SRT, j["config"]))       # jobs run from spatialResolutionTesting/
        assert j["overrides"]["results_dir"] == "../softRegionTesting/results"


def _fake_results(root, exp, gains, seeds=5):
    """Write metrics.json files with AUC = 0.70 + gains[setting] + a per-seed wobble."""
    for sp, g in gains.items():
        for s in range(seeds):
            d = root / exp / f"spatial-{sp}__seed-{s}"
            d.mkdir(parents=True)
            auc = 0.70 + g + 0.002 * (s - 2)
            json.dump({"run_name": d.name, "seed": s, "test_auc": auc, "runtime_sec": 600,
                       "spatial": {"mean_weight_entropy_nats": 0.4 if "sphere" in sp else 0.0},
                       "test_per_subject_auc": {"btb02": auc + 0.01, "btb07": auc - 0.01}},
                      open(d / "metrics.json", "w"))


def _report(*dirs):
    return subprocess.run([sys.executable, REPORT, *map(str, dirs)], capture_output=True,
                          text=True, check=True).stdout


BASE = {"none": -0.05, "destrieux_hard": 0.0, "destrieux_sphere10": 0.01, "coords": 0.0,
        "ho_prob_argmax": 0.0, "ho_prob_soft": 0.01}


def test_report_real_neighbours(tmp_path):
    _fake_results(tmp_path, "pilot_soft_xsubj", {**BASE, "destrieux_gauss3": 0.02,
                  "destrieux_gauss3_randnbr": 0.0, "destrieux_hard_nearest": 0.005,
                  "destrieux_gauss3_nearest": 0.025})
    out = _report(tmp_path / "pilot_soft_xsubj")
    assert "Check 1 (soft beats hard):** PASS" in out and "Check 3" in out
    assert "the real neighbours matter" in out
    assert (tmp_path / "pilot_soft_xsubj" / "soft_report.md").exists()


def test_report_smoothing_and_coverage(tmp_path):
    _fake_results(tmp_path, "a_xsubj", {**BASE, "destrieux_gauss3": 0.02, "destrieux_gauss3_randnbr": 0.03,
                                        "destrieux_hard_nearest": 0.0, "destrieux_gauss3_nearest": 0.0})
    assert "spreading weight" in _report(tmp_path / "a_xsubj")
    _fake_results(tmp_path, "b_xsubj", {**BASE, "destrieux_gauss3": 0.02, "destrieux_gauss3_randnbr": 0.0,
                                        "destrieux_hard_nearest": 0.02, "destrieux_gauss3_nearest": 0.02})
    assert "fewer 'unknown' regions" in _report(tmp_path / "b_xsubj")


def test_report_new_movies_only(tmp_path):
    _fake_results(tmp_path, "pilot_soft", {**BASE, "destrieux_gauss3": 0.02, "destrieux_gauss3_randnbr": 0.0})
    _fake_results(tmp_path, "pilot_soft_xsubj", {**BASE, "destrieux_gauss3": -0.01,
                  "destrieux_gauss3_randnbr": 0.0, "destrieux_hard_nearest": 0.0,
                  "destrieux_gauss3_nearest": -0.01})
    out = _report(tmp_path / "pilot_soft", tmp_path / "pilot_soft_xsubj")
    assert "name tag" in out and (tmp_path / "soft_report.md").exists()


@pytest.mark.skipif(not os.environ.get("LOCEMB_SLOW"), reason="several minutes; set LOCEMB_SLOW=1 to run")
def test_pilot_lines_run_on_fake_data(fake_root, tmp_path, monkeypatch):
    from locemb.config import load_config
    from locemb.train import run
    monkeypatch.chdir(SRT)
    root, _ = fake_root
    want = {"none", "destrieux_hard", "destrieux_hard_nearest", "destrieux_gauss3",
            "destrieux_gauss3_nearest", "destrieux_gauss3_randnbr", "ho_prob_soft"}
    for name in ("pilot_soft", "pilot_soft_xsubj"):
        for j in _sweep(name):
            if j["labels"]["seed"] not in ("0", "1") or j["labels"]["spatial"] not in want:
                continue
            ov = dict(j["overrides"])
            ov.update({"data.root": str(root), "data.cache_dir": str(tmp_path / "cache"),
                       "results_dir": str(tmp_path / "res"), "train.epochs": 1, "train.num_workers": 0,
                       "data.subject_trials": [[4, 1], [7, 1], [10, 1]]})
            if name.endswith("xsubj"):
                ov["data.test_subjects"] = [10]
            m = run(load_config(j["config"], ov))
            assert 0 <= m["test_auc"] <= 1
            if j["labels"]["spatial"].endswith("_nearest"):
                assert m["spatial"]["frac_test_electrodes_remapped"] is not None
    out = _report(tmp_path / "res" / "pilot_soft", tmp_path / "res" / "pilot_soft_xsubj")
    assert "Check 1" in out and "New movies vs new people" in out


@pytest.mark.parametrize("name", ["pilot_soft", "pilot_soft_xsubj", "exp2_soft_regions"])
def test_settings_after_defaults(name):
    """What each run really gets once configs/base.yaml defaults are merged in."""
    from locemb.config import load_config
    seen = {}
    for j in _sweep(name):
        sp = load_config(os.path.join(SRT, j["config"]), j["overrides"])["spatial"]
        seen.setdefault(j["labels"]["spatial"], sp)
    for k in ("ho_prob_argmax", "ho_prob_soft"):
        assert seen[k]["sigma_mm"] == 0          # no hidden 3 mm blur from base.yaml
    if "ho_prob_soft_s3" in seen:
        assert seen["ho_prob_soft_s3"]["sigma_mm"] == 3
    rows = [tuple(sorted((k, str(v)) for k, v in sp.items())) for sp in seen.values()]
    assert len(set(rows)) == len(rows)           # no two settings are secretly the same
