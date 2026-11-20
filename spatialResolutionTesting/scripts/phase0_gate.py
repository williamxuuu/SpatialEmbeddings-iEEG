"""Go / no-go for Phase 1, run automatically after Phase 0 (slurm/phase1_after_phase0.sbatch).

    python scripts/phase0_gate.py [--apply]

Phase 1 launches only if ALL of these hold (written 2026-11-20, before any Phase 0 result):
  1. Every Phase 0 run finished and passed check_run (45 cross-session + 24 cross-subject).
  2. phase0_report picks a recipe: it leaves the loss plateau by epoch 10 in >= 8/9
     cross-session runs and >= 6/8 cross-subject runs (validation data only).
  3. Noise is small enough to see the effects we are looking for: the projected smallest
     detectable difference between two encodings (MDE, below) is <= MAX_MDE AUC.
  4. Projected Phase 1 cost <= MAX_A100_HOURS.

MDE: the picked recipe's seed-to-seed SD of each test subject's AUC (cross-session Phase 0
runs, pooled over encodings and subjects) -> a subject's encoding difference, averaged over
Phase 1's 3 seeds, has SD sqrt(2) * sd / sqrt(3). The sign-flip test over 6 subjects reaches
p = 0.031 only when all 6 differences agree; P(all 6 > 0) >= 0.8 needs
Phase(delta / SD) >= 0.8 ** (1/6), i.e. delta >= 1.79 SD. This counts seed noise only
(differences between people in the size of the effect make the real MDE larger), so it is a
floor: if even this is too big, Phase 1 cannot answer the question.

With --apply (only after a GO), writes the picked recipe into configs/neuroprobe_v2.yaml.
Prints `GO <minutes_per_task>` on the last line, or `NO-GO`. Report: reports/phase0_gate.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from runs_table import load_runs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_MDE = 0.03
MAX_A100_HOURS = 200.0
N_PHASE1 = {"cross_session": 120, "cross_subject": 432}
EXPECTED = {"phase0_recipe": 45, "phase0_xsubj": 24}
PER_TASK = 3
Z80 = 1.79          # Phi^-1(0.8 ** (1/6))
SETUP_MIN = 2.0     # per-run overhead not in runtime_sec (imports, prefetch, atlas)
RECIPE_KEYS = {"model.frontend": "frontend", "model.readout": "readout",
               "preprocess.zscore": "zscore", "preprocess.reref": "reref"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    os.makedirs("reports", exist_ok=True)
    lines, ok = ["# Phase 0 gate\n"], True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        lines.append(f"- [{'PASS' if cond else 'FAIL'}] {msg}")
        return cond

    # 1. all runs present and valid
    for exp, n in EXPECTED.items():
        dirs = sorted(d for d in os.listdir(f"results/{exp}")) if os.path.isdir(f"results/{exp}") else []
        r = subprocess.run([sys.executable, "scripts/check_run.py", *[f"results/{exp}/{d}" for d in dirs],
                            "--expect-cuda", "--epochs", "40"], capture_output=True, text=True)
        n_ok = sum(1 for l in r.stdout.splitlines() if l.startswith("OK "))
        check(n_ok == n, f"{exp}: {n_ok}/{n} runs finished and valid")

    # 2. recipe rule
    if os.path.exists("reports/phase0_pick.json"):
        os.remove("reports/phase0_pick.json")   # never act on a stale decision
    pr = subprocess.run([sys.executable, "scripts/phase0_report.py", "results/phase0_recipe",
                         "--xsubj", "results/phase0_xsubj", "--json", "reports/phase0_pick.json"],
                        capture_output=True, text=True)
    with open("reports/phase0_report.txt", "w") as f:
        f.write(pr.stdout + pr.stderr)
    pick = None
    if os.path.exists("reports/phase0_pick.json"):
        d = json.load(open("reports/phase0_pick.json"))
        pick = d["pick"]
        check(pick is not None, f"recipe rule: {pick or 'none'} ({d['why']})")
    else:
        check(False, "recipe rule: phase0_report failed (see reports/phase0_report.txt)")

    mins = {}
    if pick:
        xs = load_runs(["results/phase0_recipe"])
        xs = xs[xs["recipe"] == pick]
        sub = load_runs(["results/phase0_xsubj"])
        sub = sub[sub["recipe"] == pick]
        # 3. noise -> MDE
        sd = (xs.groupby(["spatial", "subject"])["auc"].var(ddof=1).mean()) ** 0.5
        mde = Z80 * math.sqrt(2) * sd / math.sqrt(3)
        check(mde <= MAX_MDE, f"noise: seed SD of a subject's AUC = {sd:.4f} -> smallest detectable "
                              f"difference ~{mde:.3f} AUC (limit {MAX_MDE})")
        sd_x = (sub.groupby(["spatial", "subject"])["auc"].var(ddof=1).mean()) ** 0.5
        lines.append(f"- [info] cross-subject seed SD {sd_x:.4f} -> MDE ~{Z80 * math.sqrt(2) * sd_x / math.sqrt(3):.3f} "
                     f"(from 2 seeds x 4 cells; too few to gate on)")
        # 4. cost
        for split, df in (("cross_session", xs), ("cross_subject", sub)):
            rt = df.drop_duplicates("run")["runtime_sec"]
            mins[split] = (rt.mean() / 60 + SETUP_MIN, rt.max() / 60 + SETUP_MIN)
        hours = sum(N_PHASE1[s] * mins[s][0] / 60 for s in N_PHASE1)
        check(hours <= MAX_A100_HOURS,
              f"cost: {mins['cross_session'][0]:.1f} min/run cross-session, {mins['cross_subject'][0]:.1f} "
              f"cross-subject -> Phase 1 ~{hours:.0f} A100-hours (limit {MAX_A100_HOURS:.0f})")

    task_min = 0
    if ok and pick:
        task_min = int(math.ceil(min(8 * 60, PER_TASK * max(m[1] for m in mins.values()) * 1.5 + 15)))
        if a.apply:
            sw = yaml.safe_load(open("configs/sweeps/phase0_recipe.yaml"))
            rec = sw["axes"]["recipe"][pick]
            p = "configs/neuroprobe_v2.yaml"
            txt = open(p).read()
            for k, short in RECIPE_KEYS.items():
                txt, n = re.subn(rf"^(  {short}:\s*)\S+.*$", rf"\g<1>{rec[k]}           # set by Phase 0 gate: {pick}",
                                 txt, count=1, flags=re.M)
                if n != 1:
                    sys.exit(f"could not set {short} in {p}")
            open(p, "w").write(txt)
            lines.append(f"\nApplied recipe `{pick}` to configs/neuroprobe_v2.yaml: "
                         + ", ".join(f"{k}={rec[k]}" for k in RECIPE_KEYS))
        lines.append(f"\n**GO.** Phase 1 array tasks: {PER_TASK} runs each, --time {task_min} min.")
    else:
        lines.append("\n**NO-GO.** Phase 1 was not submitted. See reports/phase0_report.txt.")
    with open("reports/phase0_gate.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"GO {task_min}" if ok and pick else "NO-GO")
    return 0 if ok and pick else 1


if __name__ == "__main__":
    sys.exit(main())
