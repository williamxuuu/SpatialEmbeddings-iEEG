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


def xsoft(dirs: list[str], out: str) -> str:
    """configs/sweeps/main_soft_xsess.yaml (+ results/main_xsess for the shared
    none / electrode_id / grid baselines). Selected-epoch test AUC."""
    df = load_runs(dirs)
    L = ["# Soft vs hard regions, within person (main protocol)\n",
         "Outcome: test AUC at the selected (early-stopped) epoch. Contrasts paired on person x seed, "
         "seed mean per person, exact sign-flip over people. Analysis fixed in "
         "configs/sweeps/main_soft_xsess.yaml.\n"]
    if df.empty:
        L.append("No runs yet.")
    else:
        keep = df["spatial"].str.startswith("schaefer") | df["spatial"].isin(
            ["none", "electrode_id", "grid10", "grid20"])
        d = df[keep]
        L.append("Mean AUC over people and seeds:\n")
        L.append(tab(d.pivot_table(index="spatial", columns="density", values="auc", aggfunc="mean")
                     [sorted(d["density"].unique(), key=ad.level)].round(4)))
        for atlas in ("schaefer400", "schaefer1000"):
            L.append(f"## {atlas}\n\nS1 soft - hard:")
            L += contrast(d, f"{atlas}_gauss3", f"{atlas}_hard")
            L.append("\nS2 soft - random neighbours:")
            L += contrast(d, f"{atlas}_gauss3", f"{atlas}_gauss3_randnbr")
            L.append("\nS3 hard - per-electrode tags:")
            L += contrast(d, f"{atlas}_hard", "electrode_id")
            L.append("\nsoft - none:")
            L += contrast(d, f"{atlas}_gauss3", "none")
            L.append("")
        sd = d.groupby(["spatial", "density", "subject"])["auc"].std(ddof=1)
        L.append(f"Seed SD of AUC per (encoding, density, person): median {sd.median():.4f} (planned ~0.021).\n")
    text = "\n".join(L)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(text)
    return text


def _labels(run_name: str) -> dict:
    return dict(p.split("-", 1) for p in run_name.split("__") if "-" in p)


def xsubj2(dirs: list[str], out: str) -> str:
    """configs/sweeps/main_xsubj_ext.yaml + results/main_xsubj (counted as e100)."""
    rows = []
    for d in dirs:
        for mp in glob.glob(os.path.join(d, "*", "metrics.json")):
            m = json.load(open(mp))
            lab = _labels(m["run_name"])
            r = {"fold": lab["fold"], "density": lab.get("density", "e100"), "spatial": lab["spatial"],
                 "seed": int(m["seed"]), "zero_shot": m["test_last"]["auc"],
                 "unknown_w": (m.get("spatial") or {}).get("frac_weight_unknown_test_subjects")}
            for k, v in (m.get("calibration") or {}).items():
                r[f"cal_{k}"] = v["auc"]
            rows.append(r)
    L = ["# New people, extended: resolution x density, soft vs hard\n",
         "Outcome: AUC on the second half of the held-out person's recording, last epoch (fixed 20 "
         "epochs); cal_f = frozen model, linear readout fit on the first f of the first half. "
         "e100 includes results/main_xsubj. Analysis fixed in configs/sweeps/main_xsubj_ext.yaml.\n"]
    df = pd.DataFrame(rows)
    if df.empty:
        L.append("No runs yet.")
    else:
        cols = ["zero_shot"] + sorted([c for c in df if c.startswith("cal_")], key=lambda c: float(c[4:]))
        L.append(f"{len(df)} runs.\n")
        for dens in sorted(df["density"].unique(), key=ad.level):
            r = df[df["density"] == dens]
            L.append(f"## {dens}\n\nMean over people and seeds (unknown_w = share of the new person's "
                     "code weight on <unknown>):\n")
            L.append(tab(r.groupby("spatial")[cols + ["unknown_w"]].mean().round(4)))
            L.append("C1 gain over none, sign-flip over people:\n")
            for c in cols:
                base = r[r["spatial"] == "none"].set_index(["fold", "seed"])[c]
                for sp in sorted(set(r["spatial"]) - {"none"}, key=lambda x: (ad.cube_mm(x) is None, -(ad.cube_mm(x) or 0), x)):
                    diff = (r[r["spatial"] == sp].set_index(["fold", "seed"])[c] - base).dropna()
                    if diff.empty:
                        continue
                    m, p, npos, n = ad.signflip(diff.groupby(level="fold").mean().to_numpy())
                    L.append(f"- {c} {sp}: {m:+.4f} ({npos}/{n} people > 0, p = {p:.3f})")
                g = r[r["spatial"].str.fullmatch(r"grid\d+")]
                if not g.empty:
                    L.append(f"  best cube for {c}: {g.groupby('spatial')[c].mean().idxmax()}")
            L.append("")
        e = df[df["density"] == "e100"]
        L.append("## C2 soft - hard (e100), paired on person x seed, sign-flip over people\n")
        for atlas in ("schaefer400", "schaefer1000"):
            for c in cols:
                a = e[e["spatial"] == f"{atlas}_gauss3"].set_index(["fold", "seed"])[c]
                b = e[e["spatial"] == f"{atlas}_hard"].set_index(["fold", "seed"])[c]
                diff = (a - b).dropna()
                if diff.empty:
                    continue
                m, p, npos, n = ad.signflip(diff.groupby(level="fold").mean().to_numpy())
                L.append(f"- {atlas} {c}: {m:+.4f} ({npos}/{n} people > 0, p = {p:.3f})")
        L.append("\nMedian seed SD per (encoding, density, person):\n")
        L.append(tab(df.groupby(["density", "spatial", "fold"])[cols].std(ddof=1)
                     .groupby(["density", "spatial"]).median().round(3)))
    text = "\n".join(L)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(text)
    return text


def xheld(dirs: list[str], out: str) -> str:
    """configs/sweeps/heldout_shafts.yaml. Outcome = held-out-shaft test AUC at the
    selected epoch, per person (btbXXh). Replicates = split x seed."""
    rows, extra = [], []
    for d in dirs:
        for mp in glob.glob(os.path.join(d, "*", "metrics.json")):
            m = json.load(open(mp))
            lab = _labels(m["run_name"])
            ho = m.get("heldout") or {}
            info, cov = ho.get("info") or {}, ho.get("coverage") or {}
            for sid, auc in m["test_per_subject_auc"].items():
                r = {"subject": sid[:-1] if sid.endswith("h") else sid, "density": lab["density"],
                     "spatial": lab["spatial"], "split": int(lab["split"]), "seed0": int(m["seed"]),
                     "auc": auc, "coverage": cov.get(sid), "run": m["run_name"],
                     "dist_mm": (info.get(sid) or {}).get("dist_nearest_own_train_mm_median"),
                     "shared_shafts": (info.get(sid) or {}).get("n_shared_shafts")}
                for name, t in (ho.get("tests") or {}).items():
                    ps = t.get("per_subject_auc") or {}
                    key = sid if name.startswith("heldout_") else sid[:-1] if name == "train_shafts" else sid[:-1] + "a"
                    r[name] = ps.get(key)
                rows.append(r)
    L = ["# Held-out shafts: location-only transfer to unseen electrodes\n",
         "Outcome: AUC on the held-out shafts only (second half of the test movie), selected epoch "
         "(picked on training shafts). Shaft Laplacian reference. Replicates = shaft split x seed; "
         "seed means per person, exact sign-flip over the 6 people. Analysis fixed in "
         "configs/sweeps/heldout_shafts.yaml.\n"]
    df = pd.DataFrame(rows)
    if df.empty:
        L.append("No runs yet.")
    else:
        bad = df["shared_shafts"].fillna(0).sum()
        L.append(f"{len(df)} person-runs. Leak check (shafts both trained and held out): {int(bad)} "
                 f"({'OK' if bad == 0 else 'LEAK'}).\n")
        df["seed"] = df["split"] * 100 + df["seed0"]
        dens = sorted(df["density"].unique(), key=ad.level)
        L.append("Mean held-out AUC (rows: encoding; columns: density):\n")
        L.append(tab(df.pivot_table(index="spatial", columns="density", values="auc", aggfunc="mean")[dens].round(4)))
        L.append("Mean coverage (share of held-out code weight on cells seen in training):\n")
        L.append(tab(df.pivot_table(index="spatial", columns="density", values="coverage", aggfunc="mean")[dens].round(3)))
        ad.OUTCOME = "selected"
        g = df[df["spatial"].str.fullmatch(r"grid\d+") | (df["spatial"] == "none")].copy()
        g["auc_best"] = g["auc"]
        g["frac_unknown_test"] = 1.0 - g["coverage"].astype(float)
        png = os.path.splitext(out)[0] + ".png"
        try:
            L.append("## R1 resolution x density (cubes)\n")
            L.append(ad.report(g, os.path.splitext(out)[0] + "_r1.md", png))
        except Exception as ex:  # report the rest even if the curve fit fails
            L.append(f"(R1 analysis failed: {ex!r})\n")
        L.append("## R2 does the location carry to unseen shafts? (test-time checks, same model)\n")
        for sp in sorted(set(df["spatial"]) - {"none"}):
            for ctrl in ("heldout_shuffled_cells", "heldout_unknown"):
                if ctrl not in df:
                    continue
                for dn in dens:
                    r = df[(df["spatial"] == sp) & (df["density"] == dn)].dropna(subset=[ctrl])
                    if r.empty:
                        continue
                    per = (r["auc"] - r[ctrl]).groupby(r["subject"]).mean().to_numpy()
                    m, p, npos, n = ad.signflip(per)
                    L.append(f"- {sp} {dn}: true - {ctrl[8:]} = {m:+.4f} ({npos}/{n} people > 0, p = {p:.3f})")
        L.append("\nTrue cubes vs no location (training-time), per density:")
        for sp in sorted(set(df["spatial"]) - {"none"}):
            L += [x.replace("- ", f"- {sp} ", 1) for x in contrast(df, sp, "none")]
        L.append("\nSchaefer 400 soft - hard:")
        L += contrast(df, "schaefer400_gauss3", "schaefer400_hard")
        L.append("\n## R3 coverage\n")
        cub = df[df["spatial"].str.fullmatch(r"grid\d+")].copy()
        base = df[df["spatial"] == "none"].groupby(["subject", "density", "seed"])["auc"].mean()
        cub = cub.join(base.rename("auc_none"), on=["subject", "density", "seed"]).dropna(subset=["auc_none", "coverage"])
        if not cub.empty:
            cub["gain"] = cub["auc"] - cub["auc_none"]
            cub["x"] = np.log2(cub["spatial"].map(ad.cube_mm).astype(float))
            for dn in dens:
                r = cub[cub["density"] == dn]
                if len(r) < 10:
                    continue
                X = np.column_stack([np.ones(len(r)), r["x"], r["x"] ** 2, r["coverage"]])
                b = np.linalg.lstsq(X, r["gain"].to_numpy(), rcond=None)[0]
                vtx = 2 ** (-b[1] / (2 * b[2])) if b[2] < 0 else float("nan")
                X0 = X[:, :3]
                b0 = np.linalg.lstsq(X0, r["gain"].to_numpy(), rcond=None)[0]
                vtx0 = 2 ** (-b0[1] / (2 * b0[2])) if b0[2] < 0 else float("nan")
                L.append(f"- {dn}: gain ~ size + size^2 (+ coverage): best size {vtx0:.1f} mm without, "
                         f"{vtx:.1f} mm with coverage (coverage slope {b[3]:+.3f}; nan = no interior optimum)")
            L.append("\nGain vs distance from held-out contacts to the nearest training contact "
                     "(Spearman over person x split x seed, per cube size):")
            for sp, r in cub.groupby("spatial"):
                rr = r.dropna(subset=["dist_mm"])
                if len(rr) > 5:
                    L.append(f"- {sp}: rho = {rr['gain'].corr(rr['dist_mm'], method='spearman'):+.3f}")
        L.append("\n## Other test inputs (same model)\n")
        oth = [c for c in ("train_shafts", "all_shafts") if c in df]
        if oth:
            L.append(tab(df.groupby(["density", "spatial"])[["auc"] + oth].mean().unstack(0).round(4)))
        sm = df.groupby(["subject", "density", "spatial", "split"])["auc"].mean()
        L.append(f"Noise: SD across shaft splits (seed-mean) median "
                 f"{sm.groupby(['subject', 'density', 'spatial']).std(ddof=1).median():.4f}; SD across seeds "
                 f"within a split median {df.groupby(['subject', 'density', 'spatial', 'split'])['auc'].std(ddof=1).median():.4f}.\n")
    text = "\n".join(L)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["xsess", "xsubj", "xsoft", "xsubj2", "xheld"])
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print({"xsess": xsess, "xsubj": xsubj, "xsoft": xsoft, "xsubj2": xsubj2, "xheld": xheld}[a.kind](a.results, a.out))


if __name__ == "__main__":
    main()
