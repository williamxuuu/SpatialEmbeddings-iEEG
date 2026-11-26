"""Gate for main_xsess / main_xsubj: check the 10-run concurrent smoke test.

    python scripts/check_smoke_main.py results/smoke_main      # exit 0 = PASS

The sweeps are submitted with --dependency=afterok on the job running this, so a
FAIL here keeps them from starting.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

MAX_SEC_PER_EPOCH = 60.0   # full-data cross-session epoch; normal ~18 s, the 2026-11-22 stall was 20 min+
EXPECTED = 10


def main(d: str) -> int:
    runs = {}
    for mp in glob.glob(os.path.join(d, "*", "metrics.json")):
        m = json.load(open(mp))
        runs[m["run_name"].split("__")[0].split("-", 1)[1]] = m
    fails = []
    if len(runs) != EXPECTED:
        fails.append(f"{len(runs)}/{EXPECTED} runs finished: {sorted(runs)}")
    for name, m in sorted(runs.items()):
        secs = [h["sec"] for h in m["history"]]
        auc = m["test_auc"]
        print(f"{name:20s} epochs={len(secs)} sec/epoch med={np.median(secs):6.1f} max={max(secs):6.1f} "
              f"train_windows={m.get('n_train_windows')} test_auc={auc:.3f} regions={m['spatial'].get('n_regions')}")
        if not all(math.isfinite(h["train_loss"]) for h in m["history"]):
            fails.append(f"{name}: non-finite training loss")
        if not math.isfinite(auc):
            fails.append(f"{name}: test AUC not finite")
        if name.startswith("xsess_w100") and max(secs) > MAX_SEC_PER_EPOCH:
            fails.append(f"{name}: an epoch took {max(secs):.0f} s > {MAX_SEC_PER_EPOCH:.0f} s (I/O stall?)")
        if name.startswith("xsubj"):
            cal = m.get("calibration") or {}
            if set(cal) != {"0.1", "0.25", "1"}:
                fails.append(f"{name}: calibration keys {sorted(cal)}")
            for k, v in cal.items():
                print(f"    calibrated {k}: auc={v['auc']:.3f} n={v['n_calib']}")
                if not math.isfinite(v["auc"]):
                    fails.append(f"{name}: calibrated {k} AUC not finite")
    full = runs.get("xsess_w100_none", {}).get("n_train_windows")
    for name, frac in (("xsess_w10_none", 0.1), ("xsess_w10_grid20", 0.1), ("xsess_w30_grid5", 0.3)):
        n = runs.get(name, {}).get("n_train_windows")
        if full and n:
            share = n / full
            print(f"{name}: {share:.3f} of the full training windows (target {frac})")
            if abs(share - frac) > 0.03:
                fails.append(f"{name}: kept {share:.3f} of training windows, expected ~{frac}")
    eid = runs.get("xsess_w100_eid", {}).get("spatial", {}).get("n_regions") or 0
    if eid < 500:
        fails.append(f"electrode_id has only {eid} codes (expected one per electrode, ~1000)")
    print("PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
