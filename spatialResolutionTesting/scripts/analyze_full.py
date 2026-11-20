"""Analysis for the full run (docs/full_run.md, "Statistics").

    python scripts/analyze_full.py results/full_* ../softRegionTesting/results/full_* --out reports/full

Unit of analysis = the TEST SUBJECT. For each (split, task, spatial, subject):
average the per-subject AUC over seeds and folds/directions. A contrast A - B is
then one number per subject per task. Test: exact paired sign-flip permutation
over subjects (all of a subject's tasks flip together, so tasks are not treated
as independent), two-sided; BH-FDR over the pre-registered contrasts within
each split. With 6 subjects the smallest possible two-sided p is 2/64 = 0.031:
small n is a hard limit, reported as such, not hidden.

Also reports, per encoding: mean AUC with a bootstrap CI (subjects, then seeds,
resampled), how many runs left the loss plateau, the epoch they left it, and the
location-reliance probe (AUC lost when each subject's location codes are
shuffled among its electrodes at test time).

Outputs <out>/summary.md and CSVs. Never submits anything.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from runs_table import load_runs  # noqa: E402

# Pre-registered contrasts (A, B, question). Rows whose encodings are missing
# from the results are skipped and listed as such.
CONTRASTS = [
    ("destrieux", "none", "does anatomical location help at all?"),
    ("destrieux", "destrieux_shuffled", "is it the right location, or any consistent tag?"),
    ("destrieux", "electrode_id", "region sharing vs per-electrode identity"),
    ("coords", "none", "do raw coordinates help?"),
    ("destrieux", "coords", "regions vs coordinates"),
    ("destrieux_gauss3", "destrieux", "soft vs hard (same atlas)"),
    ("destrieux_gauss3", "destrieux_gauss3_randnbr", "soft: right neighbours vs random ones"),
    ("destrieux_gauss3_nearest", "destrieux_nearest", "soft vs hard at equal coverage (cross-subject)"),
    ("ho_prob_soft", "ho_prob_argmax", "probabilistic atlas: mix vs most-likely region"),
    ("schaefer400", "destrieux", "functional fine parcels vs anatomical"),
    ("lobes", "none", "coarse location"),
]


# Difference-of-differences: (A - B) - (C - D). E.g. does jitter hurt hard
# regions more than soft ones? Positive = the first difference is larger.
INTERACTIONS = [
    (("destrieux_gauss3_jit4", "destrieux_gauss3"), ("destrieux_jit4", "destrieux"),
     "localization error (4 mm) hurts hard regions more than soft ones?"),
    (("destrieux_gauss3_jit2", "destrieux_gauss3"), ("destrieux_jit2", "destrieux"),
     "same at 2 mm"),
    (("destrieux_jit4", "destrieux"), ("coords_jit4", "coords"),
     "regions vs coordinates: which is more robust to 4 mm error?"),
]


def md(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    cols = list(df.columns)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else str(v))
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def signflip_p(d: pd.DataFrame) -> tuple[float, float, int, int]:
    """d: rows = subject, columns = task, values = A - B. Returns
    (mean diff, exact two-sided p, subjects improved, n subjects)."""
    d = d.dropna(how="all")
    subj = d.mean(1).to_numpy()            # per-subject mean over tasks
    n = len(subj)
    if n == 0:
        return float("nan"), float("nan"), 0, 0
    obs = subj.mean()
    stats = [np.mean(subj * np.array(s)) for s in itertools.product([1, -1], repeat=n)]
    p = float(np.mean(np.abs(stats) >= abs(obs) - 1e-12))
    return float(obs), p, int((subj > 0).sum()), n


def boot_ci(runs: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """runs: rows (subject, seed, auc) for one encoding/split/task, already
    averaged over folds. Resample subjects, then seeds within subject."""
    rng = np.random.default_rng(seed)
    by_s = {s: g["auc"].to_numpy() for s, g in runs.groupby("subject")}
    subs = list(by_s)
    if not subs:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs))
        vals.append(np.mean([rng.choice(by_s[s], len(by_s[s])).mean() for s in pick]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="reports/full")
    ap.add_argument("--metric", default="auc", choices=["auc", "auc_last", "auc_best"])
    a = ap.parse_args(argv)
    df = load_runs(a.results)
    if df.empty:
        sys.exit("no finished runs")
    df = df.rename(columns={a.metric: "y"}) if a.metric != "auc" else df.assign(y=df["auc"])
    # Guard: the same encoding name from two different sweeps (e.g. tier4_lite's
    # "destrieux" and full_xsubj's "destrieux") must not be silently averaged.
    key = ["split", "task", "spatial", "subject", "seed"] + [c for c in ("direction", "fold") if c in df]
    clash = df.groupby(key)["exp"].nunique()
    if (clash > 1).any():
        bad = clash[clash > 1].reset_index()["exp"].size
        exps = sorted(df["exp"].unique())
        sys.exit(f"{bad} (split, task, encoding, subject, seed, fold) cells come from more than one sweep "
                 f"({exps}). Analyse sweeps with different protocols (tier4_lite / tier4_cnn / "
                 f"tier4_linear / tier3) separately from Tier 1.")
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "runs_long.csv"), index=False)
    lines = [f"# Full-run analysis\n\nMetric: `{a.metric}` (per-subject AUC). Unit: test subject.\n"]

    # 1. per-encoding summary
    per_seed = df.groupby(["split", "task", "spatial", "subject", "seed"], as_index=False)["y"].mean()
    per_subj = per_seed.groupby(["split", "task", "spatial", "subject"], as_index=False)["y"].mean()
    runs = df.drop_duplicates("run")
    esc = runs.groupby(["split", "task", "spatial"]).agg(
        runs=("run", "size"), escaped=("escaped", "mean"), escape_epoch=("escape_epoch", "median"),
        probe_drop=("probe_drop", "mean"))
    rows = []
    for (sp, tk, enc), g in per_subj.groupby(["split", "task", "spatial"]):
        lo, hi = boot_ci(per_seed[(per_seed.split == sp) & (per_seed.task == tk) & (per_seed.spatial == enc)]
                         .rename(columns={"y": "auc"}))
        rows.append({"split": sp, "task": tk, "spatial": enc, "auc": g["y"].mean(),
                     "ci_lo": lo, "ci_hi": hi, "n_subjects": len(g)})
    summ = pd.DataFrame(rows).set_index(["split", "task", "spatial"]).join(esc).reset_index()
    summ.to_csv(os.path.join(a.out, "summary_by_encoding.csv"), index=False)
    for (sp, tk), g in summ.groupby(["split", "task"]):
        lines.append(f"\n## {sp} / {tk}\n")
        lines.append(g.drop(columns=["split", "task"]).sort_values("auc", ascending=False)
                     .pipe(md))

    # 2. pre-registered contrasts
    lines.append("\n# Pre-registered contrasts\n\nmean = A - B averaged over subjects (and tasks); "
                 "p = exact sign-flip over subjects, two-sided; q = BH within split.\n")
    res = []
    for sp, gsp in per_subj.groupby("split"):
        wide = gsp.pivot_table(index=["subject", "task"], columns="spatial", values="y")
        for A, B, why in CONTRASTS:
            if A not in wide or B not in wide:
                res.append({"split": sp, "A": A, "B": B, "question": why, "status": "missing"})
                continue
            d = (wide[A] - wide[B]).unstack("task")
            m, p, imp, n = signflip_p(d)
            res.append({"split": sp, "A": A, "B": B, "question": why, "mean_diff": m, "p": p,
                        "subjects_improved": f"{imp}/{n}", "status": "ok"})
        for (A, B), (C, D), why in INTERACTIONS:
            if not all(x in wide for x in (A, B, C, D)):
                continue
            d = ((wide[A] - wide[B]) - (wide[C] - wide[D])).unstack("task")
            m, p, imp, n = signflip_p(d)
            res.append({"split": sp, "A": f"({A}-{B})", "B": f"({C}-{D})", "question": why,
                        "mean_diff": m, "p": p, "subjects_improved": f"{imp}/{n}", "status": "ok"})
    res = pd.DataFrame(res)
    if "p" in res:
        res["q"] = np.nan
        for sp in res.split.unique():
            ok = (res.split == sp) & (res.status == "ok")
            if ok.any():
                res.loc[ok, "q"] = bh(res.loc[ok, "p"].to_numpy())
    res.to_csv(os.path.join(a.out, "contrasts.csv"), index=False)
    lines.append(md(res))

    # 3. resolution: mean AUC against number of regions the training people use
    from scipy.stats import spearmanr
    nr = runs.groupby(["split", "task", "spatial"])["n_regions"].mean()
    res_tab = summ.set_index(["split", "task", "spatial"]).join(nr).reset_index()
    res_tab = res_tab[res_tab["n_regions"] > 0]
    lines.append("\n# Resolution\n\nSpearman rho between log(#regions) and AUC over region encodings "
                 "(descriptive; encodings are not independent samples).\n")
    for (sp, tk), g in res_tab.groupby(["split", "task"]):
        if len(g) >= 4:
            rho, _ = spearmanr(np.log(g["n_regions"]), g["auc"])
            lines.append(f"- {sp} / {tk}: rho = {rho:.2f} over {len(g)} encodings")
    res_tab.to_csv(os.path.join(a.out, "resolution.csv"), index=False)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {a.out}/summary.md")


if __name__ == "__main__":
    main()
