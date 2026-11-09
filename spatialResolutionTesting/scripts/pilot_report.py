"""Summarize the pilot: go/no-go checks, and Lite vs all-electrodes agreement.

    python scripts/pilot_report.py results/pilot_lite
    python scripts/pilot_report.py results/pilot_lite_xsubj

Writes <exp_dir>/pilot_report.md and prints it. It only reports and
recommends; it never submits anything. The rules (written before seeing
results; see docs/pilot.md):

  Gate 1  The model learns something: the best setting's mean AUC minus 2 SD
          (across seeds) is above 0.5.
  Gate 2  Location helps: the best location setting beats "none" by >= 0.03
          AUC on average AND in every seed (paired by seed).
  Gate 3  It's real location: destrieux beats destrieux_shuffled in every seed.
  Lite check  Lite "holds up" if, between the lite and all-electrode runs,
          (a) no two settings swap order when either gap is >= 0.02 AUC, and
          (b) no "location minus none" gain flips sign when either is >= 0.02.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

GAP = 0.02
GATE2 = 0.03
NOT_LOCATION = {"none", "destrieux_shuffled"}


def load(exp_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(exp_dir, "*", "metrics.json"))):
        m = json.load(open(p))
        lab = dict(part.split("-", 1) for part in m["run_name"].split("__") if "-" in part)
        sp = m.get("spatial", {})
        rows.append({**lab, "seed": int(lab.get("seed", m.get("seed", 0))), "auc": m["test_auc"],
                     "minutes": m["runtime_sec"] / 60, "regions": sp.get("n_regions"),
                     "unknown_test": sp.get("frac_weight_unknown_test_subjects"),
                     "elec_per_region": sp.get("electrodes_per_region_median"),
                     "per_subject": m.get("test_per_subject_auc", {})})
    return pd.DataFrame(rows)


def paired(df, a, b):
    x = df[df.spatial == a].set_index("seed").auc
    y = df[df.spatial == b].set_index("seed").auc
    return (x - y).dropna()


def section(df, name, out):
    out.append(f"\n### Electrodes: {name}\n")
    g = df.groupby("spatial").agg(auc=("auc", "mean"), sd=("auc", "std"), n=("auc", "size"),
                                  regions=("regions", "mean"), unknown_test=("unknown_test", "mean"),
                                  elec_per_region=("elec_per_region", "mean"),
                                  minutes=("minutes", "mean")).sort_values("auc", ascending=False)
    out.append("| setting | AUC mean | SD | seeds | regions | unknown (test) | electrodes/region | min/run |")
    out.append("|---|---|---|---|---|---|---|---|")
    for sp, r in g.iterrows():
        f = lambda v, d=3: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"
        out.append(f"| {sp} | {r.auc:.3f} | {f(r.sd)} | {int(r.n)} | {f(r.regions, 0)} | "
                   f"{f(r.unknown_test, 2)} | {f(r.elec_per_region, 1)} | {r.minutes:.1f} |")
    res = {"means": g.auc.to_dict()}
    best = g.index[0]
    lo = g.auc.iloc[0] - 2 * (g.sd.iloc[0] if not np.isnan(g.sd.iloc[0]) else 0)
    res["gate1"] = bool(lo > 0.5)
    out.append(f"\n- **Gate 1 (learns something):** {'PASS' if res['gate1'] else 'FAIL'}. "
               f"Best = {best}, mean - 2 SD = {lo:.3f} (needs > 0.5).")
    gains = {}
    for sp in g.index:
        if sp in NOT_LOCATION:
            continue
        d = paired(df, sp, "none")
        if len(d):
            gains[sp] = d
    res["gains"] = {k: float(v.mean()) for k, v in gains.items()}
    if gains:
        bl = max(gains, key=lambda k: gains[k].mean())
        d = gains[bl]
        res["gate2"] = bool(d.mean() >= GATE2 and (d > 0).all())
        out.append(f"- **Gate 2 (location helps):** {'PASS' if res['gate2'] else 'FAIL'}. "
                   f"Best location setting {bl} minus none = {d.mean():+.3f} "
                   f"(per seed: {', '.join(f'{v:+.3f}' for v in d)}; needs >= +{GATE2} and > 0 in every seed).")
        out.append("  - All location settings minus none: " +
                   ", ".join(f"{k} {v.mean():+.3f} ({int((v > 0).sum())}/{len(v)} seeds > 0)"
                             for k, v in sorted(gains.items(), key=lambda t: -t[1].mean())))
    d = paired(df, "destrieux", "destrieux_shuffled")
    if len(d):
        res["gate3"] = bool((d > 0).all())
        out.append(f"- **Gate 3 (real location, not just more parameters):** "
                   f"{'PASS' if res['gate3'] else 'FAIL'}. destrieux minus shuffled = {d.mean():+.3f} "
                   f"(per seed: {', '.join(f'{v:+.3f}' for v in d)}).")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    a = ap.parse_args()
    df = load(a.exp_dir)
    if df.empty:
        print("no finished runs yet"); return
    out = [f"# Pilot report: {os.path.basename(os.path.normpath(a.exp_dir))}",
           f"\n{len(df)} finished runs. AUC = test AUC (0.5 = chance). "
           f"With 3 seeds these are rough signals, not significance tests."]
    res = {}
    for name in sorted(df.electrodes.unique()) if "electrodes" in df else ["all"]:
        res[name] = section(df[df.electrodes == name] if "electrodes" in df else df, name, out)

    if {"lite", "all"} <= set(res):
        out.append("\n### Does Lite hold up? (lite vs all electrodes)\n")
        L, A = res["lite"]["means"], res["all"]["means"]
        common = [s for s in L if s in A]
        conflicts = []
        for s1, s2 in itertools.combinations(common, 2):
            dl, da = L[s1] - L[s2], A[s1] - A[s2]
            if np.sign(dl) != np.sign(da) and max(abs(dl), abs(da)) >= GAP:
                conflicts.append(f"{s1} vs {s2}: lite {dl:+.3f}, all {da:+.3f}")
        gl, ga = res["lite"].get("gains", {}), res["all"].get("gains", {})
        for s in sorted(set(gl) & set(ga)):
            if np.sign(gl[s]) != np.sign(ga[s]) and max(abs(gl[s]), abs(ga[s])) >= GAP:
                conflicts.append(f"{s} minus none flips: lite {gl[s]:+.3f}, all {ga[s]:+.3f}")
        rho = pd.Series([L[s] for s in common]).corr(pd.Series([A[s] for s in common]), method="spearman")
        out.append(f"- Rank correlation of the {len(common)} settings, lite vs all: {rho:.2f}")
        out.append(f"- Order conflicts larger than {GAP} AUC: {len(conflicts)}")
        out += [f"  - {c}" for c in conflicts]
        verdict = "Lite holds up: run the next sweep on Lite." if not conflicts else \
            "Lite does NOT hold up on these rules: look at the conflicts above before choosing."
        out.append(f"- **Recommendation:** {verdict}")
        out.append("  - (A human should still look: with 3 seeds, a conflict or its absence can be noise.)")

    mins = df.minutes.mean()
    out.append(f"\n### Cost\n\n- Mean {mins:.1f} min per run; {df.minutes.sum() / 60:.1f} GPU-hours for "
               f"these {len(df)} runs.")
    out.append(f"- At this speed, the 111-run Experiment 1 sweep would take ~{111 * mins / 60:.0f} GPU-hours.")
    out.append("\n### Per-subject test AUC (mean over seeds)\n")
    ps = pd.DataFrame([{"electrodes": r.get("electrodes", "-"), "spatial": r.spatial, "subject": s, "auc": v}
                       for _, r in df.iterrows() for s, v in r.per_subject.items()])
    if not ps.empty:
        t = ps.groupby(["electrodes", "spatial", "subject"]).auc.mean().unstack("subject").round(3)
        try:
            out.append(t.to_markdown())
        except ImportError:  # to_markdown needs the optional `tabulate` package
            out.append("```\n" + t.to_string() + "\n```")
    text = "\n".join(out)
    with open(os.path.join(a.exp_dir, "pilot_report.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
