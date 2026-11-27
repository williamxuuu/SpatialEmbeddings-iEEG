"""Gate for heldout_shafts / main_xsubj_ext: check the 10-run concurrent smoke test.

    python scripts/check_smoke_ext.py results/smoke_ext      # exit 0 = PASS
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

MAX_SEC_PER_EPOCH = 90.0   # full-data epoch; normal ~18-30 s, the 2026-11-22 stall was 20 min+
EXPECTED = 10
HELD_TESTS = {"train_shafts", "all_shafts", "heldout_unknown", "heldout_shuffled_cells"}


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
              f"test_auc={auc:.3f} regions={m['spatial'].get('n_regions')} subjects={sorted(m['test_per_subject_auc'])}")
        if not all(math.isfinite(h["train_loss"]) for h in m["history"]):
            fails.append(f"{name}: non-finite training loss")
        if not math.isfinite(auc):
            fails.append(f"{name}: test AUC not finite")
        if max(secs) > MAX_SEC_PER_EPOCH:
            fails.append(f"{name}: an epoch took {max(secs):.0f} s > {MAX_SEC_PER_EPOCH:.0f} s (I/O stall?)")
        if name.startswith("held"):
            ho = m.get("heldout") or {}
            subs = set(m["test_per_subject_auc"])
            if len(subs) != 6 or not all(s.endswith("h") for s in subs):
                fails.append(f"{name}: test subjects {sorted(subs)} are not the 6 held-out pseudo-subjects")
            want = HELD_TESTS if "none" not in name else {"train_shafts", "all_shafts"}
            got = set(ho.get("tests") or {})
            if got != want:
                fails.append(f"{name}: extra tests {sorted(got)}, expected {sorted(want)}")
            for t, v in (ho.get("tests") or {}).items():
                print(f"    {t}: {v['mean_subject_auc']:.3f}")
                if not math.isfinite(v["mean_subject_auc"]):
                    fails.append(f"{name}: {t} AUC not finite")
            info = ho.get("info") or {}
            for sid, i in sorted(info.items()):
                print(f"    {sid}: shafts held/train {i['n_shafts_heldout']}/{i['n_shafts_train']} contacts "
                      f"{i['n_contacts_heldout']}/{i['n_contacts_train']} dist {i['dist_nearest_own_train_mm_median']:.1f} mm "
                      f"coverage {ho['coverage'][sid]:.2f}")
                if i["n_shared_shafts"] != 0:
                    fails.append(f"{name} {sid}: {i['n_shared_shafts']} shafts both trained and held out (LEAK)")
                if not 0.0 <= ho["coverage"][sid] <= 1.0 + 1e-6:
                    fails.append(f"{name} {sid}: coverage {ho['coverage'][sid]}")
        else:
            cal = m.get("calibration") or {}
            if set(cal) != {"0.1", "0.25", "1"} or not all(math.isfinite(v["auc"]) for v in cal.values()):
                fails.append(f"{name}: calibration {cal and {k: v['auc'] for k, v in cal.items()}}")
    full = runs.get("held_s100_grid20", {}).get("heldout", {}).get("info", {})
    thin = runs.get("held_s25_grid5", {}).get("heldout", {}).get("info", {})
    if full and thin:
        share = np.mean([thin[s]["n_shafts_train"] / full[s]["n_shafts_train"] for s in full])
        print(f"s25 keeps {share:.2f} of the training shafts (target 0.25)")
        if not 0.15 <= share <= 0.4:
            fails.append(f"s25 kept {share:.2f} of training shafts")
    print("PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
