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
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "spatialResolutionTesting", "scripts"))
from runs_table import load_runs  # noqa: E402

ATLASES = ("schaefer400", "schaefer1000")
F0_XSESS, F0_XSUBJ = 0.89, 0.75   # share of runs that must leave the plateau
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


def _pair_signflip(runs: pd.DataFrame, a: str, b: str, by: list[str], n_flip: int = 20000, seed: int = 0):
    """Two-sided sign-flip p of mean(a - b) over paired (by + seed) units, and the
    population-level exact sign-flip over the means of `by` groups (e.g. people)."""
    pa = runs[runs["spatial"] == a].set_index(by + ["seed"])["val_last"]
    pb = runs[runs["spatial"] == b].set_index(by + ["seed"])["val_last"]
    d = (pa - pb).dropna()
    if len(d) < 2:
        return None, None
    v = d.to_numpy()
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1, 1], size=(n_flip, len(v))) * v).mean(1)
    p_units = float((1 + np.sum(np.abs(null) >= abs(v.mean()) - 1e-12)) / (1 + n_flip))
    g = d.groupby(level=by).mean().to_numpy()
    signs = np.array(list(itertools.product([1, -1], repeat=len(g)))) if len(g) <= 16 else None
    p_pop = None
    if signs is not None and len(g) > 1:
        p_pop = float(np.mean(np.abs(signs @ g / len(g)) >= abs(g.mean()) - 1e-12))
    return p_units, p_pop


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
            ok = len(sub) > 0 and sub["esc_early"].mean() >= F0_XSESS
            out.rule(f"F0 {atlas}/{dens}", ok,
                     f"{int(sub['esc_early'].sum())}/{len(sub)} runs leave the plateau by epoch {EARLY} "
                     f"(need >= {F0_XSESS:.0%})")
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
    a, b = eff.get(("schaefer1000", "full")), eff.get(("schaefer400", "full"))
    out.rule("D3 resolution", None if a is None or b is None else a > b,
             f"soft - hard schaefer1000 {_fmt(a)} vs schaefer400 {_fmt(b)} at full (predict schaefer1000 >)")
    out("\n(For information only, after the rules) test AUC per person, mean over seeds:\n")
    t = df.pivot_table(index=["density", "spatial"], columns="subject", values="auc", aggfunc="mean")
    out(t.round(3).to_string(), fence=True)


def score_xsubj(df: pd.DataFrame, out: Out):
    out("## Cross-subject (phase0_soft_xsubj)\n")
    if df.empty:
        out("No runs yet.\n")
        return
    df = df.copy()
    df["size"] = df["density"] if "density" in df.columns else "k5"   # k5 only since 2026-11-20
    sizes = [x for x in ("k2", "k5") if x in set(df["size"])]
    r = _run_level(df, ["fold", "size"])
    out(f"{len(r)} runs. Validation AUC (last epoch), mean over folds and seeds:\n")
    out(r.pivot_table(index="spatial", columns="size", values="val_last", aggfunc="mean")
        .round(4).to_string(), fence=True)
    eff = {}
    for atlas in ATLASES:
        for size in sizes:
            rr = r[r["size"] == size]
            sub = rr[rr["spatial"].str.startswith(atlas + "_")]
            ok = len(sub) > 0 and sub["esc_early"].mean() >= F0_XSUBJ
            out.rule(f"F0 {atlas}/{size}", ok,
                     f"{int(sub['esc_early'].sum())}/{len(sub)} runs leave the plateau by epoch {EARLY} "
                     f"(need >= {F0_XSUBJ:.0%})")
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
                p_u, p_pop = _pair_signflip(rr, a, b, ["fold"])
                ptxt = "" if p_u is None else (f"; sign-flip p = {p_u:.4f} over person x seed pairs"
                                               + ("" if p_pop is None else f", {p_pop:.3f} over people"))
                out.rule(f"{name} {atlas}/{size}", pooled > 0,
                         f"{_fmt(pooled)} (per fold: {per_fold}; {int(d['n_pos'].sum())}/{int(d['n'].sum())} runs > 0{ptxt})")
    for atlas in ATLASES if len(sizes) > 1 else []:
        a, b = eff.get((atlas, "k2")), eff.get((atlas, "k5"))
        out.rule(f"D3 density {atlas}", None,
                 f"soft - hard with 2 training people {_fmt(a)} vs 5 {_fmt(b)} (reported; no fixed prediction)")
    if "frac_unknown_test" in df.columns:
        out("\nHeld-out weight on regions training never saw (mean):\n")
        out(df.drop_duplicates("run").pivot_table(index="spatial", columns="size", values="frac_unknown_test",
                                                  aggfunc="mean").round(3).to_string(), fence=True)
    out("\n(For information only, after the rules) test AUC of the held-out person, mean over seeds:\n")
    out(df.pivot_table(index=["size", "spatial"], columns="fold", values="auc", aggfunc="mean").round(3).to_string(), fence=True)


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
