"""Reports for the main experiment (written 2026-11-26, before any run).

    python scripts/main_report.py xsess results/main_xsess --out reports/main_xsess.md
    python scripts/main_report.py xsubj results/main_xsubj --out reports/main_xsubj.md

xsess (configs/sweeps/main_xsess.yaml): the density x resolution analysis
(analyze_density.py, outcome = selected epoch), then the controls: per density,
paired on person x seed, seed-averaged, exact sign-flip over the 6 people.
xsubj (configs/sweeps/main_xsubj.yaml): zero-shot and calibrated AUC per encoding,
gain over none per person, sign-flip over people, share below 0.5, seed SD.
Both also report training health: share of runs that left the loss plateau, epochs
run and the selected epoch, so a density level that never learns is visible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_density as ad  # noqa: E402
from runs_table import escape_epoch, load_runs  # noqa: E402

tab = lambda t: "```\n" + t.to_string() + "\n```\n"  # noqa: E731


def contrast(df: pd.DataFrame, a: str, b: str, col: str = "auc") -> list[str]:
    """a - b per density: paired on person x seed, seed mean per person, sign-flip."""
    out = []
    for d in sorted(df["density"].unique(), key=ad.level):
        r = df[df["density"] == d]
        pa = r[r["spatial"] == a].set_index(["subject", "seed"])[col]
        pb = r[r["spatial"] == b].set_index(["subject", "seed"])[col]
        diff = (pa - pb).dropna()
        if diff.empty:
            continue
        per = diff.groupby(level="subject").mean()
        m, p, npos, n = ad.signflip(per.to_numpy())
        out.append(f"- {d}: {a} - {b} = {m:+.4f} ({npos}/{n} people > 0, sign-flip p = {p:.3f}; "
                   f"{len(diff)} pairs)")
    return out


def health(dirs: list[str]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for mp in glob.glob(os.path.join(d, "*", "metrics.json")):
            m = json.load(open(mp))
            lab = dict(p.split("-", 1) for p in m["run_name"].split("__") if "-" in p)
            rows.append({**lab, "escaped": escape_epoch(m.get("history", [])) is not None,
                         "epochs_run": m.get("epochs_run"), "best_epoch": m.get("best_epoch"),
                         "n_train_windows": m.get("n_train_windows")})
    return pd.DataFrame(rows)


def xsess(dirs: list[str], out: str) -> str:
    ad.OUTCOME = "selected"
    df = load_runs(dirs)
    png = os.path.splitext(out)[0] + ".png"
    text = ad.report(df, out, png) if not df.empty else "# Density x resolution\n\nNo runs yet.\n"
    L = [text, "## Controls (selected epoch; paired on person x seed)\n"]
    if not df.empty:
        L.append("Mean AUC over people and seeds (rows: encoding; columns: density):\n")
        L.append(tab(df.pivot_table(index="spatial", columns="density", values="auc", aggfunc="mean")
                     [sorted(df["density"].unique(), key=ad.level)].round(4)))
        L.append("Does LOCATION matter, or just shared tags?")
        L += contrast(df, "grid20", "grid20_shuffled")
        L.append("\nPer-electrode tags vs nothing:")
        L += contrast(df, "electrode_id", "none")
        L.append("\nBest cube (by mean over people, per density) vs per-electrode tags:")
        for d in sorted(df["density"].unique(), key=ad.level):
            r = df[(df["density"] == d) & df["spatial"].str.fullmatch(r"grid\d+")]
            if r.empty:
                continue
            best = r.groupby("spatial")["auc"].mean().idxmax()
            L += [x.replace(f"- {d}:", f"- {d} (best = {best}):", 1)
                  for x in contrast(df[df["density"] == d], best, "electrode_id")]
        sd = df.groupby(["spatial", "density", "subject"])["auc"].std(ddof=1)
        L.append(f"\nSeed SD of AUC per (encoding, density, person): median {sd.median():.4f}, "
                 f"90th percentile {sd.quantile(0.9):.4f} (planned ~0.025).\n")
    h = health(dirs)
    if not h.empty:
        L.append("## Training health\n")
        L.append(tab(h.groupby("density").agg(runs=("escaped", "size"), left_plateau=("escaped", "mean"),
                                              epochs_run=("epochs_run", "median"),
                                              best_epoch=("best_epoch", "median"),
                                              train_windows=("n_train_windows", "median")).round(2)))
    text = "\n".join(L)
    open(out, "w").write(text)
    return text


def xsubj(dirs: list[str], out: str) -> str:
    rows = []
    for d in dirs:
        for mp in glob.glob(os.path.join(d, "*", "metrics.json")):
            m = json.load(open(mp))
            lab = dict(p.split("-", 1) for p in m["run_name"].split("__") if "-" in p)
            r = {"fold": lab["fold"], "spatial": lab["spatial"], "seed": int(m["seed"]),
                 "zero_shot": m["test_last"]["auc"]}
            for k, v in (m.get("calibration") or {}).items():
                r[f"cal_{k}"] = v["auc"]
            rows.append(r)
    L = ["# New people: zero-shot and calibrated\n",
         "Outcome: AUC on the second half of the held-out person's recording, last epoch "
         "(fixed 20 epochs). cal_f = model frozen, linear readout fit on the first f of the first half. "
         "Analysis fixed in configs/sweeps/main_xsubj.yaml.\n"]
    df = pd.DataFrame(rows)
    if df.empty:
        L.append("No runs yet.")
    else:
        cols = ["zero_shot"] + sorted([c for c in df if c.startswith("cal_")], key=lambda c: float(c[4:]))
        L.append(f"{len(df)} runs.\n\nMean over people and seeds:\n")
        L.append(tab(df.groupby("spatial")[cols].mean().round(4)))
        L.append("Per person (seed mean):\n")
        for c in cols:
            L.append(f"{c}:\n" + tab(df.pivot_table(index="spatial", columns="fold", values=c,
                                                    aggfunc="mean").round(3)))
        L.append("Gain over none, per person seed mean, sign-flip over people:\n")
        for c in cols:
            base = df[df["spatial"] == "none"].set_index(["fold", "seed"])[c]
            for s in sorted(set(df["spatial"]) - {"none"}):
                diff = (df[df["spatial"] == s].set_index(["fold", "seed"])[c] - base).dropna()
                if diff.empty:
                    continue
                m, p, npos, n = ad.signflip(diff.groupby(level="fold").mean().to_numpy())
                L.append(f"- {c} {s}: {m:+.4f} ({npos}/{n} people > 0, p = {p:.3f})")
        L.append("\nShare of runs below 0.5, and median seed SD per (encoding, person):\n")
        L.append(tab(pd.DataFrame({**{f"below05_{c}": df.groupby("spatial")[c].apply(lambda x: (x < 0.5).mean())
                                      for c in cols},
                                   **{f"sd_{c}": df.groupby(["spatial", "fold"])[c].std(ddof=1)
                                      .groupby("spatial").median() for c in cols}}).round(3)))
    text = "\n".join(L)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["xsess", "xsubj"])
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print((xsess if a.kind == "xsess" else xsubj)(a.results, a.out))


if __name__ == "__main__":
    main()
