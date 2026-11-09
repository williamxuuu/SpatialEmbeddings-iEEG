"""Gather every metrics.json of an experiment into tables.

    python scripts/collect_results.py results/exp1_resolution
    python scripts/collect_results.py results/exp2_soft_regions --sweep runs/exp2_soft_regions.jsonl

Writes <exp_dir>/all_runs.csv and <exp_dir>/summary.csv, prints the summary,
lists runs that haven't finished (with --sweep), and for settings named
<family>_hard prints soft-minus-hard differences paired by seed.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def parse_labels(run_name):
    out = {}
    for part in run_name.split("__"):
        if "-" in part:
            k, v = part.split("-", 1)
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--sweep", help="runs/<exp>.jsonl, to list missing runs")
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(a.exp_dir, "*", "metrics.json"))):
        m = json.load(open(p))
        r = {"run_name": m["run_name"], **parse_labels(m["run_name"]),
             "test_auc": m["test_auc"], "test_acc": m["test_acc"], "best_val": m["best_val"],
             "runtime_min": m["runtime_sec"] / 60, "epochs_run": m["epochs_run"],
             "n_params": m["n_params"], "gpu": m.get("gpu")}
        r.update({f"sp_{k}": v for k, v in m.get("spatial", {}).items()})
        rows.append(r)
    if not rows:
        print("no finished runs yet"); return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.exp_dir, "all_runs.csv"), index=False)

    keys = [k for k in ["spatial", "model"] if k in df.columns]
    agg = df.groupby(keys).agg(
        auc_mean=("test_auc", "mean"), auc_sd=("test_auc", "std"), n_seeds=("test_auc", "size"),
        n_regions=("sp_n_regions", "mean"),
        unknown_test=("sp_frac_weight_unknown_test_subjects", "mean") if "sp_frac_weight_unknown_test_subjects" in df else ("test_auc", "size"),
        multi_region=("sp_frac_electrodes_multi_region", "mean") if "sp_frac_electrodes_multi_region" in df else ("test_auc", "size"),
        minutes=("runtime_min", "mean"),
    ).reset_index().sort_values(keys[::-1] + ["auc_mean"], ascending=[True] * len(keys) + [False])
    agg.to_csv(os.path.join(a.exp_dir, "summary.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_rows", 500)
    print(agg.round(4).to_string(index=False))
    print(f"\n{len(df)} runs, {df.runtime_min.sum() / 60:.1f} GPU-hours used in total")

    # Paired soft-vs-hard differences (Experiment 2)
    if "spatial" in df.columns and "seed" in df.columns and df.spatial.str.endswith("_hard").any():
        print("\nSoft minus hard (same family, model and seed). >0 means soft helped:")
        out = []
        for (model,), g in df.groupby(["model"] if "model" in df else [lambda _: "all"]):
            for fam_hard in sorted(g.spatial[g.spatial.str.endswith("_hard")].unique()):
                fam = fam_hard[: -len("_hard")]
                base = g[g.spatial == fam_hard].set_index("seed").test_auc
                for sp in sorted(g.spatial[g.spatial.str.startswith(fam + "_")].unique()):
                    if sp == fam_hard:
                        continue
                    other = g[g.spatial == sp].set_index("seed").test_auc
                    d = (other - base).dropna()
                    if len(d):
                        out.append({"model": model, "setting": sp, "vs": fam_hard, "delta_mean": d.mean(),
                                    "delta_sd": d.std(), "n_pairs": len(d), "n_positive": int((d > 0).sum())})
        if out:
            print(pd.DataFrame(out).round(4).to_string(index=False))
        print("(With 3 seeds this is a rough signal, not a significance test.)")

    if a.sweep:
        want = [json.loads(l)["overrides"]["run_name"] for l in open(a.sweep) if l.strip()]
        missing = [i for i, n in enumerate(want) if n not in set(df.run_name)]
        print(f"\n{len(missing)} of {len(want)} runs not finished"
              + (f"; array indices: {','.join(map(str, missing))}" if missing else ""))


if __name__ == "__main__":
    main()
