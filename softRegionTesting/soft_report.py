"""Summarize the Experiment 2 pilot: does soft (a weighted mix of nearby
regions) beat hard (one region per electrode), and if so, why?

    python softRegionTesting/soft_report.py softRegionTesting/results/pilot_soft softRegionTesting/results/pilot_soft_xsubj

Prints the report and writes soft_report.md (inside the folder if you give
one, else next to the folders). It only reports; it never submits anything.

The rules, written before seeing results (see README.md). "Paired" means same
seed; "most seeds" means at least 4 of 5 (80%).

  Check 0  Location helps at all: destrieux_hard beats none on average.
           If not, soft vs hard can't be read.
  Check 1  Soft beats hard: destrieux_gauss3 minus destrieux_hard is at least
           +0.01 AUC on average and above 0 in most seeds.
  Check 2  It's the right neighbours: destrieux_gauss3 beats the same weights
           with random neighbours (randnbr) on average and in most seeds.
  Check 3  (cross-subject only) It's not just fewer "unknown" regions:
           gauss3_nearest minus hard_nearest (same coverage) is at least
           +0.01 on average and above 0 in most seeds.
  Also shown, not pass/fail: coordinates vs soft, probabilistic atlas soft vs
  hard, a much wider mix (10 mm ball), per-person gains, and whether soft only helps
  on new movies (where a mix of regions can act as an electrode name tag).
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

GAIN = 0.01
MOST = 0.8
HARD, SOFT, WIDE = "destrieux_hard", "destrieux_gauss3", "destrieux_sphere10"


def load(exp_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(exp_dir, "*", "metrics.json"))):
        m = json.load(open(p))
        lab = dict(part.split("-", 1) for part in m["run_name"].split("__") if "-" in part)
        sp = m.get("spatial", {})
        rows.append({"spatial": lab.get("spatial", "?"), "seed": int(lab.get("seed", m.get("seed", 0))),
                     "auc": m["test_auc"], "minutes": m["runtime_sec"] / 60,
                     "entropy": sp.get("mean_weight_entropy_nats"),
                     "multi": sp.get("frac_electrodes_multi_region"),
                     "unknown_test": sp.get("frac_weight_unknown_test_subjects"),
                     "per_subject": m.get("test_per_subject_auc", {})})
    return pd.DataFrame(rows)


def paired(df, a, b):
    x = df[df.spatial == a].set_index("seed").auc
    y = df[df.spatial == b].set_index("seed").auc
    return (x - y).dropna()


def passes(d, min_mean):
    """min_mean=0 means 'above 0'; otherwise 'at least min_mean' (with float slack)."""
    if not len(d):
        return None
    ok_mean = d.mean() > 0 if min_mean == 0 else d.mean() >= min_mean - 1e-9
    return bool(ok_mean and (d > 0).sum() >= math.ceil(MOST * len(d)))


def describe(d):
    return (f"{d.mean():+.3f} (per seed: {', '.join(f'{v:+.3f}' for v in d)}; "
            f"{int((d > 0).sum())}/{len(d)} above 0)")


def fmt(v, d=3):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def section(df, name, out):
    out.append(f"\n## {name}\n")
    g = df.groupby("spatial").agg(auc=("auc", "mean"), sd=("auc", "std"), n=("auc", "size"),
                                  multi=("multi", "mean"), entropy=("entropy", "mean"),
                                  unknown_test=("unknown_test", "mean"),
                                  minutes=("minutes", "mean")).sort_values("auc", ascending=False)
    out.append("| setting | AUC mean | SD | seeds | electrodes with >1 region | mixing (entropy) "
               "| unknown weight (test people) | min/run |")
    out.append("|---|---|---|---|---|---|---|---|")
    for sp, r in g.iterrows():
        out.append(f"| {sp} | {r.auc:.3f} | {fmt(r.sd)} | {int(r.n)} | {fmt(r.multi, 2)} | "
                   f"{fmt(r.entropy, 2)} | {fmt(r.unknown_test, 3)} | {r.minutes:.1f} |")

    res = {}
    lines = []

    def check(key, title, a, b, min_mean, explain):
        d = paired(df, a, b)
        res[key] = passes(d, min_mean)
        res[key + "_gain"] = float(d.mean()) if len(d) else None
        if res[key] is None:
            lines.append(f"- **{title}:** not run ({a} or {b} missing).")
        else:
            lines.append(f"- **{title}:** {'PASS' if res[key] else 'FAIL'}. {a} minus {b} = "
                         f"{describe(d)}. {explain}")

    check("c0", "Check 0 (location helps at all)", HARD, "none", 0, "Needs > 0 on average.")
    check("c1", "Check 1 (soft beats hard)", SOFT, HARD, GAIN,
          f"Needs >= +{GAIN} on average and > 0 in most seeds.")
    check("c2", "Check 2 (right neighbours, not just spreading weight)", SOFT, SOFT + "_randnbr", 0,
          "Needs > 0 on average and in most seeds.")
    if (df.spatial == HARD + "_nearest").any():
        check("c3", "Check 3 (not just fewer unknown regions)", SOFT + "_nearest", HARD + "_nearest", GAIN,
              f"Same coverage on both sides. Needs >= +{GAIN} on average and > 0 in most seeds.")
    else:
        res["c3"] = None
    out += [""] + lines

    out.append("\n**Also shown (not pass/fail):**\n")
    for a, b, why in [("coords", SOFT, "if this is >= 0, plain coordinates do as well as soft regions, "
                                        "so soft may help only by adding position detail"),
                      ("ho_prob_soft", "ho_prob_argmax", "probabilistic atlas, soft minus hard"),
                      (WIDE, SOFT, "much wider mix (10 mm ball vs 3 mm weighted)"),
                      (WIDE, HARD, "much wider mix vs one region")]:
        d = paired(df, a, b)
        if len(d):
            out.append(f"- {a} minus {b} = {describe(d)}: {why}.")

    ps = pd.DataFrame([{"spatial": r.spatial, "seed": r.seed, "subject": s, "auc": v}
                       for _, r in df.iterrows() for s, v in r.per_subject.items()])
    if not ps.empty and {SOFT, HARD} <= set(ps.spatial):
        t = ps.pivot_table(index=["subject", "seed"], columns="spatial", values="auc")
        if {SOFT, HARD} <= set(t.columns):
            gain = (t[SOFT] - t[HARD]).groupby("subject").agg(["mean", "size", lambda x: int((x > 0).sum())])
            out.append(f"\n**Soft minus hard, per test person** (mean over seeds; seeds above 0):\n")
            for subj, r in gain.iterrows():
                out.append(f"- {subj}: {r['mean']:+.3f} ({int(r.iloc[2])}/{int(r['size'])})")

    out.append(f"\n**Reading:** {verdict(res)}")
    return res


def verdict(r):
    if r.get("c0") is False:
        return ("Location doesn't beat 'none' here, so soft vs hard can't be read. "
                "Look at the Experiment 1 pilot first.")
    if r.get("c1") is None:
        return "Not enough runs yet."
    if r["c1"] and r.get("c2") is False:
        return ("Soft beats hard, but random neighbours do about as well. The gain looks like "
                "spreading weight (a smoothing effect), not the right neighbours.")
    if r["c1"] and r.get("c3") is False:
        return ("Soft beats hard, but not once both have the same coverage. The gain is probably "
                "that soft gives new people fewer 'unknown' regions, not mixing at borders.")
    if r["c1"] and r.get("c2") and r.get("c3") in (True, None):
        return ("Soft helps, and the real neighbours matter. Worth running the full Experiment 2 sweep "
                "(still a rough signal with 5 seeds).")
    if r.get("c3"):
        return ("Plain soft doesn't clearly beat plain hard, but it does when coverage is matched. "
                "Look at the per-person numbers before deciding.")
    return (f"No clear sign that soft beats hard (needs +{GAIN} AUC in most seeds). A smaller effect "
            "would need more seeds to see; the BaRISTA paper reports seed-to-seed SD of about 0.016.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dirs", nargs="+")
    ap.add_argument("--out", help="where to write the markdown (default: see top of file)")
    a = ap.parse_args()
    out = ["# Experiment 2 pilot: soft vs hard regions",
           "\nAUC = test AUC (0.5 = chance). With 5 seeds these are rough signals, not significance tests. "
           "'Mixing (entropy)' is 0 for one region per electrode and grows as weight spreads over more regions."]
    results, total = {}, []
    for d in a.exp_dirs:
        df = load(d)
        name = os.path.basename(os.path.normpath(d))
        if df.empty:
            out.append(f"\n## {name}\n\nno finished runs yet")
            continue
        out.append(f"\n{len(df)} finished runs in {name}.")
        results[name] = section(df, name, out)
        total.append(df)

    xs = [n for n in results if "xsubj" in n]
    cs = [n for n in results if "xsubj" not in n]
    if xs and cs:
        c, x = results[cs[0]], results[xs[0]]
        out.append("\n## New movies vs new people\n")
        out.append(f"- Soft minus hard: new movies {fmt(c.get('c1_gain'))}, new people {fmt(x.get('c1_gain'))}.")
        if c.get("c1") and x.get("c1") is False:
            out.append("- Soft only helps on new movies of the same people. There, a fine mix of regions can "
                       "act as a name tag for each electrode. Trust the new-people result more.")
    if total:
        df = pd.concat(total)
        out.append(f"\n## Cost\n\n- Mean {df.minutes.mean():.1f} min per run; {df.minutes.sum() / 60:.1f} "
                   f"GPU-hours for these {len(df)} runs.")
        out.append(f"- At this speed, the 117-run full Experiment 2 sweep would take "
                   f"~{117 * df.minutes.mean() / 60:.0f} GPU-hours.")
    text = "\n".join(out)
    path = a.out or (os.path.join(a.exp_dirs[0], "soft_report.md") if len(a.exp_dirs) == 1 else
                     os.path.join(os.path.dirname(os.path.normpath(a.exp_dirs[0])), "soft_report.md"))
    with open(path, "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
