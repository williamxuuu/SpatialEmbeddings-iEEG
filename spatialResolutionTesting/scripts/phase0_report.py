"""Apply the Phase 0 recipe rule (written in configs/sweeps/phase0_recipe.yaml
before any run) and print which recipe to copy into configs/neuroprobe_v2.yaml.

    python scripts/phase0_report.py results/phase0_recipe --xsubj results/phase0_xsubj

Uses VALIDATION data only for the decision (val mean-subject AUC at the last
epoch). Test AUCs are printed for information, after the decision.
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--xsubj", help="results of phase0_xsubj (cross-subject check)")
    ap.add_argument("--min-escaped-xsubj", type=int, default=6, help="of 8 runs per recipe")
    ap.add_argument("--json", help="also write the decision here (used by scripts/phase0_gate.py)")
    a = ap.parse_args(argv)

    def out(pick, why):
        if a.json:
            with open(a.json, "w") as fh:
                json.dump({"pick": pick, "why": why}, fh)
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
        out(None, "no recipe leaves the plateau in enough cross-session runs")
        return 1
    xs_ok = None
    if a.xsubj:
        xr = load_runs([a.xsubj]).drop_duplicates("run")
        xr = xr.assign(esc_early=xr["escape_epoch"].notna() & (xr["escape_epoch"] <= a.escape_by))
        xg = xr.groupby("recipe").agg(n=("run", "size"), escaped=("esc_early", "sum"),
                                      val=("val_last", "mean"))
        print("\nCross-subject check (validation only):\n" + xg.round(4).to_string())
        xs_ok = set(xg.index[xg["escaped"] >= a.min_escaped_xsubj])
    # Rank eligible recipes: best validation first, ties within TIE -> simpler (earlier in ORDER).
    ranked = []
    rest = list(ok.index)
    while rest:
        top = max(ok.loc[r, "val"] for r in rest)
        nxt = next(r for r in rest if ok.loc[r, "val"] >= top - TIE)
        ranked.append(nxt); rest.remove(nxt)
    pick = None
    for r in ranked:
        if xs_ok is None:
            pick = r; break
        if r in xs_ok:
            pick = r; break
        why = "not in the cross-subject check" if r not in xg.index else "stalls on new people"
        print(f"skip {r}: best on cross-session but {why}")
    if pick is None:
        print("\nNO recipe passes both checks. Do not launch the full run; look at the logs.")
        out(None, "no cross-session-eligible recipe also learns cross-subject")
        return 1
    print(f"\nPICK: {pick} (val {ok.loc[pick, 'val']:.4f}; ties within {TIE} -> simpler"
          + ("; passes the cross-subject check)" if xs_ok is not None else "; no cross-subject check given)"))
    out(pick, "passes" + (" both checks" if xs_ok is not None else " the cross-session check"))
    by_sp = runs.pivot_table(index="recipe", columns="spatial", values="val_last", aggfunc="mean")
    print("\nValidation AUC by recipe x spatial:\n" + by_sp.reindex(g.index).round(4).to_string())
    t = df.groupby(["recipe", "spatial"])["auc"].mean().unstack()
    print("\n(For information only, after the decision) test mean-subject AUC:\n"
          + t.reindex(g.index).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
