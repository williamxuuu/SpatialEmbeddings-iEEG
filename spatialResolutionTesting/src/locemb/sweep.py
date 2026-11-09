"""Expand a sweep YAML into one JSON line per run (one line = one Slurm array task).

    python -m locemb.sweep configs/sweeps/exp1_resolution.yaml
    -> writes runs/exp1_resolution.jsonl and prints the number of runs

Sweep YAML format:

    config: configs/neuroprobe.yaml      # base config every run starts from
    exp_name: exp1_resolution
    overrides: {model.name: transformer} # applied to every run
    axes:                                # Cartesian product of all axes
      spatial:                           # named variants: name -> overrides
        none:  {spatial.type: none}
        grid20: {spatial.type: region, spatial.atlas: grid20}
      seed: [0, 1, 2]                    # plain list: values for key "seed"
    exclude:                             # optional: drop combos (match on axis values)
      - {spatial: none, model: linear}
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import yaml


def expand(sweep: dict) -> list[dict]:
    axes = sweep.get("axes", {})
    names, choices = [], []
    for axis, spec in axes.items():
        names.append(axis)
        if isinstance(spec, dict):
            choices.append([(str(k), v or {}) for k, v in spec.items()])
        else:
            choices.append([(f"{v}", {axis: v}) for v in spec])
    jobs = []
    for combo in itertools.product(*choices) if choices else [()]:
        labels = {n: c[0] for n, c in zip(names, combo)}
        if any(all(labels.get(k) == str(v) for k, v in ex.items()) for ex in sweep.get("exclude", [])):
            continue
        ov = dict(sweep.get("overrides", {}))
        for _, o in combo:
            ov.update(o)
        run_name = "__".join(f"{n}-{labels[n]}" for n in names) or "run"
        ov["exp_name"] = sweep["exp_name"]
        ov["run_name"] = run_name
        jobs.append({"config": sweep["config"], "overrides": ov, "labels": labels})
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--out-dir", default="runs")
    a = ap.parse_args(argv)
    with open(a.sweep) as f:
        sweep = yaml.safe_load(f)
    jobs = expand(sweep)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{sweep['exp_name']}.jsonl")
    with open(out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{len(jobs)} {out}")


if __name__ == "__main__":
    main()
