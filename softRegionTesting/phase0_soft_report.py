"""Score Experiment 2 Phase 0 (soft vs hard under the v2 protocol).

    cd spatialResolutionTesting
    python ../softRegionTesting/phase0_soft_report.py \
        ../softRegionTesting/results/phase0_soft_xsess ../softRegionTesting/results/phase0_soft_xsubj

Applies the rules written in the two sweep headers
(configs/sweeps/phase0_soft_xsess.yaml, phase0_soft_xsubj.yaml) before any run:
decisions use validation AUC at the last epoch (val_last), averaged over seeds
(and folds); test AUC per person is printed after, for information only.
This is a feasibility / effect-size check: the rules say which way effects point
and how big they are, not whether they are significant.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "spatialResolutionTesting", "scripts"))
from runs_table import load_runs  # noqa: E402

ATLASES = ("destrieux", "schaefer400")
EARLY = 10  # "leaves the plateau" = escape by this epoch


def _runs(d: str) -> pd.DataFrame:
    df = load_runs([d])
    if df.empty:
        return df
    df["esc_early"] = df["escape_epoch"].notna() & (df["escape_epoch"] <= EARLY)
    return df


def _run_level(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per run (val_last and escape are run-level)."""
    return df.drop_duplicates("run")[keys + ["spatial", "seed", "val_last", "esc_early"]]


def _diff(runs: pd.DataFrame, a: str, b: str, by: list[str]) -> pd.DataFrame:
    """a - b on val_last, paired on everything in `by` + seed; returns mean, SD, n, n>0."""
    pa = runs[runs["spatial"] == a].set_index(by + ["seed"])["val_last"]
    pb = runs[runs["spatial"] == b].set_index(by + ["seed"])["val_last"]
    d = (pa - pb).dropna()
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(level=by) if by else d.groupby(lambda _: "all")
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.size(),
                         "n_pos": g.apply(lambda x: int((x > 0).sum()))})


def _fmt(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:+.4f}"


class Out:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, s: str = "", fence: bool = False):
        self.lines.append(f"```\n{s}\n```\n" if fence else s)

    def rule(self, name: str, ok, msg: str):
        tag = "n/a" if ok is None else ("PASS" if ok else "FAIL")
        self.lines.append(f"- [{tag}] {name}: {msg}")


def score_xsess(df: pd.DataFrame, out: Out):
    out("## Cross-session (phase0_soft_xsess)\n")
    if df.empty:
        out("No runs yet.\n")
        return
    r = _run_level(df, ["density"])
    out(f"{r['run'].nunique() if 'run' in r else len(r)} runs. Validation AUC (last epoch), mean over seeds:\n")
    out(r.pivot_table(index="spatial", columns="density", values="val_last", aggfunc="mean")
        .round(4).to_string(), fence=True)
    eff = {}
    for atlas in ATLASES:
        for dens in ("full", "half"):
            rr = r[r["density"] == dens]
            sets = [f"{atlas}_hard", f"{atlas}_gauss3", f"{atlas}_gauss3_randnbr"]
            sub = rr[rr["spatial"].isin(sets)]
            out.rule(f"F0 {atlas}/{dens}", len(sub) == 9 and sub["esc_early"].sum() >= 8,
                     f"{int(sub['esc_early'].sum())}/{len(sub)} runs leave the plateau by epoch {EARLY} (need >= 8/9)")
            d1 = _diff(rr, f"{atlas}_gauss3", f"{atlas}_hard", [])
            d2 = _diff(rr, f"{atlas}_gauss3", f"{atlas}_gauss3_randnbr", [])
            m1 = None if d1.empty else d1["mean"].iloc[0]
            m2 = None if d2.empty else d2["mean"].iloc[0]
            eff[(atlas, dens)] = m1
            for name, d, m in (("D1 soft - hard", d1, m1), ("D2 soft - randnbr", d2, m2)):
                msg = "no paired runs" if d.empty else (
                    f"{_fmt(m)} (seed SD {d['sd'].iloc[0]:.4f}, {d['n_pos'].iloc[0]}/{d['n'].iloc[0]} seeds > 0)")
                out.rule(f"{name} {atlas}/{dens}", None if m is None else m > 0, msg)
    for atlas in ATLASES:
        a, b = eff.get((atlas, "half")), eff.get((atlas, "full"))
        out.rule(f"D3 density {atlas}", None if a is None or b is None else a > b,
                 f"soft - hard at half {_fmt(a)} vs full {_fmt(b)} (predict half > full)")
    a, b = eff.get(("schaefer400", "full")), eff.get(("destrieux", "full"))
    out.rule("D3 resolution", None if a is None or b is None else a > b,
             f"soft - hard schaefer400 {_fmt(a)} vs destrieux {_fmt(b)} at full (predict schaefer400 >)")
    out("\n(For information only, after the rules) test AUC per person, mean over seeds:\n")
    t = df.pivot_table(index=["density", "spatial"], columns="subject", values="auc", aggfunc="mean")
    out(t.round(3).to_string(), fence=True)


def score_xsubj(df: pd.DataFrame, out: Out):
    out("## Cross-subject (phase0_soft_xsubj)\n")
    if df.empty:
        out("No runs yet.\n")
        return
    df = df.copy()
    df["size"] = df["fold"].str.rsplit("_", n=1).str[1]
    r = _run_level(df, ["fold", "size"])
    out(f"{len(r)} runs. Validation AUC (last epoch), mean over folds and seeds:\n")
    out(r.pivot_table(index="spatial", columns="size", values="val_last", aggfunc="mean")
        .round(4).to_string(), fence=True)
    eff = {}
    for atlas in ATLASES:
        for size in ("full", "small"):
            rr = r[r["size"] == size]
            sub = rr[rr["spatial"].str.startswith(atlas + "_")]
            out.rule(f"F0 {atlas}/{size}", len(sub) == 16 and sub["esc_early"].sum() >= 12,
                     f"{int(sub['esc_early'].sum())}/{len(sub)} runs leave the plateau by epoch {EARLY} (need >= 12/16)")
            for name, a, b in (("D1 soft - hard", f"{atlas}_gauss3", f"{atlas}_hard"),
                               ("D4 equal coverage", f"{atlas}_gauss3_nearest", f"{atlas}_hard_nearest")):
                d = _diff(rr, a, b, ["fold"])
                if d.empty:
                    out.rule(f"{name} {atlas}/{size}", None, "no paired runs")
                    continue
                pooled = (d["mean"] * d["n"]).sum() / d["n"].sum()
                if name.startswith("D1"):
                    eff[(atlas, size)] = pooled
                per_fold = ", ".join(f"{f} {_fmt(m)}" for f, m in d["mean"].items())
                out.rule(f"{name} {atlas}/{size}", pooled > 0,
                         f"{_fmt(pooled)} (per fold: {per_fold}; {int(d['n_pos'].sum())}/{int(d['n'].sum())} runs > 0)")
    for atlas in ATLASES:
        a, b = eff.get((atlas, "small")), eff.get((atlas, "full"))
        out.rule(f"D3 density {atlas}", None if a is None or b is None else a > b,
                 f"soft - hard with 2 training people {_fmt(a)} vs 5 {_fmt(b)} (predict small > full)")
    if "frac_unknown_test" in df.columns:
        out("\nHeld-out weight on regions training never saw (mean):\n")
        out(df.drop_duplicates("run").pivot_table(index="spatial", columns="size", values="frac_unknown_test",
                                                  aggfunc="mean").round(3).to_string(), fence=True)
    out("\n(For information only, after the rules) test AUC of the held-out person, mean over seeds:\n")
    out(df.pivot_table(index="spatial", columns="fold", values="auc", aggfunc="mean").round(3).to_string(), fence=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("xsess")
    ap.add_argument("xsubj")
    ap.add_argument("--out", default=os.path.join(HERE, "reports", "phase0_soft.md"))
    a = ap.parse_args(argv)
    out = Out()
    out("# Experiment 2 Phase 0: soft vs hard (v2 protocol)\n")
    out("Feasibility and effect-size check, not a significance test. Rules: sweep headers.\n")
    score_xsess(_runs(a.xsess), out)
    score_xsubj(_runs(a.xsubj), out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    text = "\n".join(out.lines) + "\n"
    with open(a.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
