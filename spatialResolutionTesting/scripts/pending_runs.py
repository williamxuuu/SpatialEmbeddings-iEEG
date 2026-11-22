"""Write the sweep lines that have no metrics.json yet to a new .jsonl file.

    python scripts/pending_runs.py runs/foo.jsonl   ->  prints "<n_pending> runs/foo.pending.jsonl"

Same output-path rule as locemb.train.run: <results_dir>/<exp_name>/<run_name>/metrics.json,
where results_dir comes from the line's overrides, else its base config, else "results".
Used by slurm/submit_sweep.sh so a re-submission queues only missing runs.
"""
import json
import os
import sys

import yaml


def main(path: str) -> None:
    base_cache: dict[str, dict] = {}
    pending = []
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    for ln in lines:
        rec = json.loads(ln)
        ov = rec.get("overrides", {})
        cfg_path = rec.get("config")
        if cfg_path not in base_cache:
            base_cache[cfg_path] = {}
            if cfg_path and os.path.exists(cfg_path):
                with open(cfg_path) as g:
                    base_cache[cfg_path] = yaml.safe_load(g) or {}
        base = base_cache[cfg_path]
        rd = ov.get("results_dir", base.get("results_dir", "results"))
        exp = ov.get("exp_name", base.get("exp_name"))
        run = ov.get("run_name", base.get("run_name"))
        if exp is None or run is None or not os.path.exists(os.path.join(rd, exp, run, "metrics.json")):
            pending.append(ln if ln.endswith("\n") else ln + "\n")
    out = path[:-len(".jsonl")] + ".pending.jsonl" if path.endswith(".jsonl") else path + ".pending"
    with open(out, "w") as f:
        f.writelines(pending)
    print(len(pending), out)
    print(f"[pending] {len(pending)}/{len(lines)} runs still to do", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
