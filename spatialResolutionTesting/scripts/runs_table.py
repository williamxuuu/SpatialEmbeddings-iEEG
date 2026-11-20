"""Load finished runs into one tidy table (one row per run x test subject).

    from runs_table import load_runs
    df = load_runs(["results/phase0_recipe", "results/full_xsess_speech"])

Columns: exp, run, <every sweep axis label, e.g. spatial / recipe / task / fold>,
seed, task, split, subject, auc (selected epoch), auc_best, auc_last,
probe_drop (location-reliance drop in mean-subject AUC, run level),
escaped, escape_epoch, val_last (validation mean-subject AUC at the last epoch),
epochs_run, runtime_sec.

"Escaped" = the training loss left the chance plateau: some epoch's train loss
is below ln 2 - ESCAPE_MARGIN. escape_epoch = first such epoch (None if never).
"""

from __future__ import annotations

import glob
import json
import math
import os

import pandas as pd
import yaml

LN2 = math.log(2)
ESCAPE_MARGIN = 0.02


def _labels(run_name: str) -> dict:
    out = {}
    for part in run_name.split("__"):
        if "-" in part:
            k, v = part.split("-", 1)
            out[k] = v
    return out


def escape_epoch(history: list[dict], margin: float = ESCAPE_MARGIN):
    for h in history:
        if h.get("train_loss") is not None and h["train_loss"] < LN2 - margin:
            return int(h["epoch"])
    return None


def load_runs(dirs: list[str]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for mp in sorted(glob.glob(os.path.join(d, "*", "metrics.json"))):
            with open(mp) as f:
                m = json.load(f)
            cp = os.path.join(os.path.dirname(mp), "config.yaml")
            cfg = yaml.safe_load(open(cp)) if os.path.exists(cp) else {}
            data = cfg.get("data", {})
            hist = m.get("history", [])
            ee = escape_epoch(hist)
            best = m.get("test_best") or {}
            last = m.get("test_last") or {}
            probe = m.get("location_probe") or {}
            base = {"exp": m["exp_name"], "run": m["run_name"], **_labels(m["run_name"]),
                    "seed": int(m["seed"]), "task": data.get("eval_name"), "split": data.get("split"),
                    "electrodes": data.get("electrodes"),
                    "escaped": ee is not None, "escape_epoch": ee,
                    "val_last": hist[-1].get("val_mean_subject_auc") if hist else None,
                    "probe_drop": probe.get("drop_mean_subject_auc"),
                    "n_regions": (m.get("spatial") or {}).get("n_regions"),
                    "frac_unknown_test": (m.get("spatial") or {}).get("frac_weight_unknown_test_subjects"),
                    "epochs_run": m.get("epochs_run"), "runtime_sec": m.get("runtime_sec"),
                    "select": m.get("select", "best")}
            for sub, auc in m["test_per_subject_auc"].items():
                rows.append({**base, "subject": sub, "auc": auc,
                             "auc_best": (best.get("per_subject_auc") or {}).get(sub),
                             "auc_last": (last.get("per_subject_auc") or {}).get(sub)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["seed"] = df["seed"].astype(int)
    return df
