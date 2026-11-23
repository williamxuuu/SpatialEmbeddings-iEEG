"""Score the fix pilot (configs/sweeps/fix_pilot.yaml). Written before any pilot run finished.

    python scripts/fix_pilot_report.py results/fix_pilot > reports/fix_pilot.md

Per run:
  sel    test AUC at the epoch picked on the VALIDATION PERSON (test_best; no test labels)
  last   test AUC at the last epoch (the old protocol's number, with 4 training people)
  oracle max over epochs of the monitor curve (first half of the test recording) --
         a ceiling only; picking the epoch this way would leak test-person labels
Questions (from the sweep header):
  Q1 selection   sel - last, per encoding (positive = selection helps)
  Q2 dropout     d50 - d0 on sel and on last, paired on (person, encoding, seed)
  Q3 noise       seed SD of sel per (person, encoding, drop); before: 0.07-0.13
  Q4 below 0.5   share of runs with sel < 0.5 and last < 0.5; before: 25%
Also: how well the validation person tracks the held-out person (corr over epochs of
val_auc vs monitor_auc), and the epoch picked.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def load(d):
    R = []
    for f in glob.glob(os.path.join(d, "*", "metrics.json")):
        m = json.load(open(f))
        p = dict(x.split("-", 1) for x in m["run_name"].split("__"))
        h = m["history"]
        val = np.array([e["val_auc"] for e in h]); mon = np.array([e.get("monitor_auc", np.nan) for e in h])
        R.append(dict(fold=p["fold"], spatial=p["spatial"], drop=p["drop"], seed=int(p["seed"]),
                      sel=m["test_best"]["auc"], last=m["test_last"]["auc"], oracle=np.nanmax(mon),
                      sel_epoch=m["best_epoch"], n_epochs=len(h),
                      track=np.corrcoef(val, mon)[0, 1] if len(h) > 2 else np.nan,
                      val_people=",".join(m.get("val_subjects") or [])))
    return pd.DataFrame(R)


def main(d):
    df = load(d)
    pd.set_option("display.width", 200)
    print(f"# Fix pilot report\n\n{len(df)} runs from `{d}` (planned 126).\n")
    if df.empty:
        return
    df["arm"] = df.spatial + "|" + df["drop"]
    print("## Means per arm (over people and seeds)\n")
    g = df.groupby("arm")
    t = g[["sel", "last", "oracle", "sel_epoch", "track"]].mean()
    t["sel<0.5"] = g.sel.apply(lambda x: (x < 0.5).mean()); t["last<0.5"] = g["last"].apply(lambda x: (x < 0.5).mean())
    t["n"] = g.size()
    print(t.round(3).to_string(), "\n")
    print("## Per person, `sel` (mean over seeds)\n")
    print(df.pivot_table(index="arm", columns="fold", values="sel", aggfunc="mean").round(3).to_string(), "\n")
    print("## Q1 selection: sel - last (mean, share > 0)\n")
    df["q1"] = df.sel - df["last"]
    print(df.groupby("arm").q1.agg(["mean", lambda x: (x > 0).mean()]).round(3).to_string(), "\n")
    print("## Q2 code dropout: d50 - d0, paired on person x encoding x seed\n")
    k = ["fold", "spatial", "seed"]
    a = df[df["drop"] == "d50"].set_index(k); b = df[df["drop"] == "d0"].set_index(k)
    for col in ("sel", "last"):
        dd = (a[col] - b[col]).dropna()
        if len(dd):
            print(f"- {col}: overall {dd.mean():+.3f} (n={len(dd)}, {np.mean(dd > 0):.0%} > 0); by encoding "
                  + ", ".join(f"{s} {v:+.3f}" for s, v in dd.groupby(level='spatial').mean().items()))
    print("\n## Q3 noise: seed SD of `sel` per (person, arm), median over people\n")
    sd = df.groupby(["arm", "fold"]).sel.std().groupby("arm").median()
    sdl = df.groupby(["arm", "fold"])["last"].std().groupby("arm").median()
    print(pd.DataFrame({"sd_sel": sd, "sd_last": sdl}).round(3).to_string(), "\n")
    print(f"Before the fix: 0.07-0.13 (last epoch, 5 training people, 40 epochs).\n")
    print("## Validation person per fold\n")
    print(df.groupby("fold").val_people.first().to_string())


if __name__ == "__main__":
    main(sys.argv[1])
