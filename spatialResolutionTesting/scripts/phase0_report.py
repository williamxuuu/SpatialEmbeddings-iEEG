"""Apply the Phase 0 recipe rule (written in configs/sweeps/phase0_recipe.yaml
before any run) and print which recipe to copy into configs/neuroprobe_v2.yaml.

    python scripts/phase0_report.py results/phase0_recipe

Uses VALIDATION data only for the decision (val mean-subject AUC at the last
epoch). Test AUCs are printed for information, after the decision.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from runs_table import load_runs  # noqa: E402

ORDER = ["r0_pilot_input", "r1_cnn_session", "r2_stft_car", "r3_stft_lap", "r4_stft_lap_mean"]
TIE = 0.005


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--min-escaped", type=int, default=8, help="of 9 runs per recipe")
    ap.add_argument("--escape-by", type=int, default=10, help="epoch")
    a = ap.parse_args(argv)
    df = load_runs(a.results)
    if df.empty:
        sys.exit("no finished runs")
    runs = df.drop_duplicates("run")
    runs = runs.assign(esc_early=runs["escape_epoch"].notna() & (runs["escape_epoch"] <= a.escape_by))
    g = runs.groupby("recipe").agg(n=("run", "size"), val=("val_last", "mean"), val_sd=("val_last", "std"),
                                   escaped=("esc_early", "sum"), probe=("probe_drop", "mean"),
                                   min_per_run=("runtime_sec", lambda s: s.mean() / 60))
    g = g.reindex([r for r in ORDER if r in g.index])
    print("Decision table (validation only):\n")
    print(g.round(4).to_string())
    ok = g[g["escaped"] >= a.min_escaped]
    if ok.empty:
        print("\nNO recipe leaves the plateau reliably. Do not launch the full run; look at the logs.")
        return 1
    top = ok["val"].max()
    pick = next(r for r in ok.index if ok.loc[r, "val"] >= top - TIE)
    print(f"\nPICK: {pick} (val {ok.loc[pick, 'val']:.4f}; best eligible {top:.4f}; ties within {TIE} -> simpler)")
    by_sp = runs.pivot_table(index="recipe", columns="spatial", values="val_last", aggfunc="mean")
    print("\nValidation AUC by recipe x spatial:\n" + by_sp.reindex(g.index).round(4).to_string())
    t = df.groupby(["recipe", "spatial"])["auc"].mean().unstack()
    print("\n(For information only, after the decision) test mean-subject AUC:\n"
          + t.reindex(g.index).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
