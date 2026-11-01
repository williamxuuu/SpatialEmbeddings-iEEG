"""Rough GPU-hour budget for a sweep.

    # with a guess of minutes per run
    python scripts/estimate_compute.py configs/sweeps/exp1_resolution.yaml --minutes 30 --parallel 8
    # with measured times from a pilot (uses mean runtime per model type)
    python scripts/estimate_compute.py configs/sweeps/exp1_resolution.yaml --pilot results/pilot --parallel 8
"""
import argparse
import glob
import json
import os
from collections import Counter

import yaml

from locemb.sweep import expand

ap = argparse.ArgumentParser()
ap.add_argument("sweeps", nargs="+")
ap.add_argument("--minutes", type=float, default=None, help="guessed minutes per run")
ap.add_argument("--pilot", help="results dir with finished pilot runs (measured times)")
ap.add_argument("--parallel", type=int, default=8, help="GPUs you expect to get at once")
a = ap.parse_args()

per_model = {}
if a.pilot:
    times = {}
    for p in glob.glob(os.path.join(a.pilot, "*", "metrics.json")):
        m = json.load(open(p))
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(p), "config.yaml")))
        times.setdefault(cfg["model"]["name"], []).append(m["runtime_sec"] / 60)
    per_model = {k: sum(v) / len(v) for k, v in times.items()}
    print("pilot minutes/run:", {k: round(v, 1) for k, v in per_model.items()})

total_min, total_runs = 0.0, 0
for s in a.sweeps:
    sw = yaml.safe_load(open(s))
    jobs = expand(sw)
    models = Counter(j["overrides"].get("model.name", "transformer") for j in jobs)
    mins = 0.0
    for mdl, n in models.items():
        per = per_model.get(mdl, a.minutes)
        if per is None:
            raise SystemExit(f"no time for model {mdl}: pass --minutes or a pilot that includes it")
        mins += n * per
    total_min += mins; total_runs += len(jobs)
    print(f"{sw['exp_name']:24s} runs={len(jobs):4d} {dict(models)}  ~{mins / 60:.1f} GPU-h")
print(f"{'TOTAL':24s} runs={total_runs:4d}  ~{total_min / 60:.1f} GPU-h  "
      f"~{total_min / 60 / a.parallel:.1f} h wall-clock at {a.parallel} GPUs (ignoring queue wait)")
