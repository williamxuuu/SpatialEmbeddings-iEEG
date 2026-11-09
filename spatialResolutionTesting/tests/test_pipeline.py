"""End-to-end: tiny synthetic runs through train.run for each model / spatial type."""

import json

import pytest

from locemb.config import load_config
from locemb.sweep import expand
from locemb.train import run


@pytest.mark.parametrize("model,spatial", [
    ("transformer", {"spatial.type": "none"}),
    ("transformer", {"spatial.type": "coords"}),
    ("transformer", {"spatial.type": "region", "spatial.atlas": "grid40", "spatial.mode": "trilinear"}),
    ("linear", {"spatial.type": "none"}),
    ("linear", {"spatial.type": "region", "spatial.atlas": "grid40"}),
])
def test_tiny_run(tmp_path, model, spatial):
    cfg = load_config("configs/smoke.yaml", {
        "results_dir": str(tmp_path), "run_name": "t", "model.name": model,
        "train.epochs": 1, "synthetic.windows_per_subject": 30, "synthetic.n_subjects": 4,
        **spatial})
    m = run(cfg)
    assert 0.0 <= m["test_acc"] <= 1.0
    saved = json.loads((tmp_path / "smoke" / "t" / "metrics.json").read_text())
    assert saved["run_name"] == "t"
    # second call is skipped (resumable sweeps)
    assert run(cfg)["runtime_sec"] == saved["runtime_sec"]


def test_sweep_expansion_counts():
    import yaml
    for name, n in [("smoke", 8), ("exp1_resolution", 93), ("pilot_lite", 42), ("pilot_lite_xsubj", 42)]:
        jobs = expand(yaml.safe_load(open(f"configs/sweeps/{name}.yaml")))
        assert len(jobs) == n
        assert len({j["overrides"]["run_name"] for j in jobs}) == n
