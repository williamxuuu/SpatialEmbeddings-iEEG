"""Density x resolution analysis (configs/sweeps/density_resolution.yaml).

    python scripts/analyze_density.py results/density_resolution

Writes reports/density_resolution.md and reports/density_resolution.png.
The analysis is the one written in the sweep header before any run:

  gain(h)  = AUC(cube h) - AUC(none): same held-out person, density and seed,
             then averaged over seeds. Outcome = test AUC at the last epoch.
  H1       per person and density, slope of gain on x = log2(80 / h);
           interaction = slope(densest) - slope(sparsest); permutation test of density
           labels within person x seed, sign-flip over people as the population version.
  density  any axis label ending in a number: k5 = 5 training people, e25 = 25% of
           each training person's electrodes; "level" = that number.
  H2       best cube size per density: argmax of the mean curve, and the vertex of
           a quadratic in x with a bootstrap over people.
  H3       does density matter once coverage is known? Leave-one-person-out R^2 of
           gain ~ x * coverage (x, coverage, x:coverage), with and without density.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runs_table import load_runs  # noqa: E402

H_MAX = 80.0  # x = log2(H_MAX / h): 0 for 80 mm, 4 for 5 mm


def cube_mm(name: str):
    m = re.fullmatch(r"grid(\d+(?:\.\d+)?)", str(name))
    return float(m.group(1)) if m else None


def signflip(v: np.ndarray) -> tuple[float, float, int, int]:
    """Exact two-sided sign-flip test of mean(v) = 0. Returns mean, p, n>0, n."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0, 0
    obs = v.mean()
    stats = [np.mean(v * np.array(s)) for s in itertools.product([1, -1], repeat=n)]
    return float(obs), float(np.mean(np.abs(stats) >= abs(obs) - 1e-12)), int((v > 0).sum()), n


def seed_gains(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, density, seed, h): paired gain over 'none' and coverage."""
    auc = "auc_last" if "auc_last" in df and df["auc_last"].notna().all() else "auc"
    df = df[df["subject"] == "btb" + df["fold"].str[2:]].copy() if "fold" in df else df.copy()
    base = (df[df["spatial"] == "none"].set_index(["subject", "density", "seed"])[auc]
            .rename("auc_none"))
    cub = df[df["spatial"].map(cube_mm).notna()].copy()
    cub["h"] = cub["spatial"].map(cube_mm)
    cub = cub.join(base, on=["subject", "density", "seed"])
    cub["gain"] = cub[auc] - cub["auc_none"]
    cub["coverage"] = 1.0 - cub["frac_unknown_test"].astype(float)
    cub["x"] = np.log2(H_MAX / cub["h"])
    return cub.dropna(subset=["gain"])


def gains(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, density, h): seed-mean paired gain over 'none', and coverage."""
    auc = "auc_last" if "auc_last" in df and df["auc_last"].notna().all() else "auc"
    cub = seed_gains(df)
    g = (cub
         .groupby(["subject", "density", "h"])
         .agg(gain=("gain", "mean"), coverage=("coverage", "mean"), n_seeds=("gain", "size"),
              auc=(auc, "mean"), auc_none=("auc_none", "mean"))
         .reset_index())
    g["x"] = np.log2(H_MAX / g["h"])
    g["k"] = g["density"].map(level)
    return g


def level(label: str) -> float:
    """Numeric density level from an axis label: k5 -> 5, e25 -> 25."""
    m = re.search(r"(\d+(?:\.\d+)?)$", str(label))
    if not m:
        raise ValueError(f"density label {label!r} must end in a number")
    return float(m.group(1))


def ends(labels) -> tuple[str, str]:
    """(sparsest, densest) density labels."""
    s = sorted(set(labels), key=level)
    return s[0], s[-1]


def perm_test(df: pd.DataFrame, lo: str | None = None, hi: str | None = None, n_perm: int = 5000, seed: int = 0):
    """Primary H1 test. Statistic: mean over people and seeds of slope(hi) - slope(lo),
    slope = OLS slope of gain on x within one person x seed x density. Null: density
    does not change the gain curve, so density labels are exchangeable within each
    person x seed; permute them (all densities) and recompute. Returns stat, p (two-sided),
    n blocks. Inference is about these people (the sign-flip over people is the
    population-level version)."""
    sg = seed_gains(df)
    if lo is None or hi is None:
        lo, hi = ends(sg["density"])
    sl = (sg.groupby(["subject", "seed", "density"])
          .apply(lambda r: np.polyfit(r["x"], r["gain"], 1)[0] if r["x"].nunique() >= 3 else np.nan)
          .unstack("density").dropna())
    if lo not in sl or hi not in sl or sl.empty:
        return float("nan"), float("nan"), 0
    cols = list(sl.columns)
    S = sl.to_numpy()                      # blocks x densities
    i_lo, i_hi = cols.index(lo), cols.index(hi)
    obs = float(np.mean(S[:, i_hi] - S[:, i_lo]))
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n_perm, S.shape[0], S.shape[1])), axis=2)
    Sp = np.take_along_axis(np.broadcast_to(S, perm.shape), perm, axis=2)
    null = np.mean(Sp[:, :, i_hi] - Sp[:, :, i_lo], axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + n_perm)
    return obs, float(p), S.shape[0]


def slopes(g: pd.DataFrame) -> pd.DataFrame:
    """Per (subject, density): OLS slope of gain on x (needs >= 3 cube sizes)."""
    out = []
    for (s, d), r in g.groupby(["subject", "density"]):
        if r["x"].nunique() >= 3:
            out.append({"subject": s, "density": d, "slope": float(np.polyfit(r["x"], r["gain"], 1)[0])})
    return pd.DataFrame(out).pivot(index="subject", columns="density", values="slope") if out else pd.DataFrame()


def vertex(x: np.ndarray, y: np.ndarray) -> float:
    """Location of the best x under a quadratic fit, clipped to the tested range."""
    lo, hi = float(np.min(x)), float(np.max(x))
    c2, c1, _ = np.polyfit(x, y, 2)
    if c2 >= 0:  # no interior maximum: best end of the range
        return hi if np.polyval([c2, c1, 0], hi) >= np.polyval([c2, c1, 0], lo) else lo
    return float(np.clip(-c1 / (2 * c2), lo, hi))


def best_h(g: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = sorted(g["subject"].unique())
    rows, boots = [], {}
    for d, r in g.groupby("density"):
        curve = r.groupby("x")["gain"].mean()
        piv = r.pivot(index="subject", columns="x", values="gain")
        bs = []
        for _ in range(n_boot):
            pick = rng.choice(subjects, size=len(subjects), replace=True)
            m = piv.reindex(pick).mean()
            if m.notna().sum() >= 3:
                bs.append(vertex(m.dropna().index.to_numpy(), m.dropna().to_numpy()))
        boots[d] = np.array(bs)
        xv = vertex(curve.index.to_numpy(), curve.to_numpy())
        lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
        rows.append({"density": d, "argmax_h_mm": H_MAX / 2 ** curve.idxmax(),
                     "vertex_h_mm": H_MAX / 2 ** xv,
                     "vertex_h_95ci_mm": f"{H_MAX / 2 ** hi:.1f}-{H_MAX / 2 ** lo:.1f}"})
    out = pd.DataFrame(rows).set_index("density")
    return out, boots


def _vertex_fit(g: pd.DataFrame):
    """OLS gain ~ 1 + x + x^2 + s + x s + x^2 s (s = log2 k); returns dx*/ds and x*(s) at each k."""
    x, s2 = g["x"].to_numpy(), np.log2(g["k"].to_numpy())
    X = np.column_stack([np.ones_like(x), x, x ** 2, s2, x * s2, x ** 2 * s2])
    b, *_ = np.linalg.lstsq(X, g["gain"].to_numpy(), rcond=None)
    lo, hi = float(x.min()), float(x.max())

    def xstar(s):
        a1, a2 = b[1] + b[4] * s, b[2] + b[5] * s
        if a2 >= 0:  # no interior maximum at this density: best end
            return hi if a1 * hi + a2 * hi ** 2 >= a1 * lo + a2 * lo ** 2 else lo
        return float(np.clip(-a1 / (2 * a2), lo, hi))
    ks = sorted(g["k"].unique())
    xs = {k: xstar(np.log2(k)) for k in ks}
    slope = (xs[ks[-1]] - xs[ks[0]]) / (np.log2(ks[-1]) - np.log2(ks[0]))
    return slope, xs


def vertex_regression(g: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    """Pooled vertex regression with a bootstrap over held-out people."""
    rng = np.random.default_rng(seed)
    people = sorted(g["subject"].unique())
    slope, xs = _vertex_fit(g)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(people, size=len(people), replace=True)
        gb = pd.concat([g[g["subject"] == p] for p in pick])
        if gb["k"].nunique() >= 2 and gb["x"].nunique() >= 3:
            bs.append(_vertex_fit(gb)[0])
    return slope, xs, np.array(bs)


def loo_r2(g: pd.DataFrame, with_k: bool) -> float:
    """Leave-one-person-out R^2 of gain ~ x * coverage (+ density dummies)."""
    def design(r):
        x, c = r["x"].to_numpy(), r["coverage"].to_numpy()
        cols = [np.ones(len(r)), x, c, x * c]
        if with_k:
            for d in sorted(g["density"].unique())[1:]:
                cols.append((r["density"] == d).to_numpy(float))
        return np.column_stack(cols)
    g = g.dropna(subset=["gain", "coverage"])
    pred = np.full(len(g), np.nan)
    for s in g["subject"].unique():
        tr, te = g["subject"] != s, g["subject"] == s
        beta, *_ = np.linalg.lstsq(design(g[tr]), g.loc[tr, "gain"].to_numpy(), rcond=None)
        pred[te.to_numpy()] = design(g[te]) @ beta
    y = g["gain"].to_numpy()
    return float(1 - np.nansum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def plot(g: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for d, r in sorted(g.groupby("density"), key=lambda t: level(t[0])):
        c = r.groupby("h")["gain"].agg(["mean", "sem"]).sort_index(ascending=False)
        ax[0].errorbar(c.index, c["mean"], yerr=c["sem"], marker="o", capsize=3, label=f"density {d}")
        ax[1].scatter(r["coverage"], r["gain"], s=18, alpha=0.7, label=f"density {d}")
    ax[0].set_xscale("log", base=2)
    ax[0].invert_xaxis()
    ax[0].set_xticks(sorted(g["h"].unique()))
    ax[0].set_xticklabels([f"{h:g}" for h in sorted(g["h"].unique())])
    ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set_xlabel("cube size (mm), finer ->")
    ax[0].set_ylabel("AUC gain over no location")
    ax[0].set_title("Gain vs resolution (mean +/- SE over people)")
    ax[0].legend()
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set_xlabel("coverage (held-out weight in cubes training saw)")
    ax[1].set_ylabel("AUC gain over no location")
    ax[1].set_title("Gain vs coverage (one dot = person x density x cube)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(df: pd.DataFrame, out_md: str, out_png: str | None) -> str:
    L = ["# Density x resolution\n",
         "Outcome: held-out person's test AUC at the last epoch. Gain = AUC(cube) - AUC(none),"
         " paired on person, density and seed, then averaged over seeds. Analysis fixed in"
         " configs/sweeps/density_resolution.yaml.\n"]
    if df.empty:
        L.append("No runs yet.")
        return "\n".join(L)
    g = gains(df)
    L.append(f"{df['run'].nunique()} runs; {g['subject'].nunique()} held-out people; "
             f"seeds per cell: {int(g['n_seeds'].min())}-{int(g['n_seeds'].max())}.\n")
    tab = lambda t: "```\n" + t.to_string() + "\n```\n"  # noqa: E731
    L.append("Mean gain over people (rows: cube size mm; columns: density level):\n")
    L.append(tab(g.pivot_table(index="h", columns="density", values="gain", aggfunc="mean")
                 .sort_index(ascending=False).round(4)))
    L.append("Mean coverage:\n")
    L.append(tab(g.pivot_table(index="h", columns="density", values="coverage", aggfunc="mean")
                 .sort_index(ascending=False).round(3)))

    L.append("## H1 (primary): does finer resolution pay off more with more data?\n")
    sl = slopes(g)
    lo, hi = ends(g["density"])
    if g["density"].nunique() >= 2 and {lo, hi} <= set(sl.columns):
        L.append("Slope of gain per halving of cube size, per person:\n")
        L.append(tab(sl.round(4)))
        pm, pp, nb = perm_test(df)
        L.append(f"**Interaction slope({hi}) - slope({lo}) = {pm:+.4f} AUC per halving; "
                 f"permutation p = {pp:.4f}** (density labels permuted within {nb} person x seed blocks; "
                 f"prediction: > 0)\n")
        m, p, npos, n = signflip((sl[hi] - sl[lo]).to_numpy())
        L.append(f"- population level: {npos}/{n} people > 0; exact sign-flip p = {p:.3f} "
                 f"(n = 6 caps this at p >= 0.031)")
        mids = sorted(set(sl.columns), key=level)
        for b, a in zip(mids[:-1], mids[1:]) if len(mids) > 2 else []:
            if {a, b} <= set(sl.columns):
                m2, p2, np2, n2 = signflip((sl[a] - sl[b]).to_numpy())
                L.append(f"- secondary: slope({a}) - slope({b}) = {m2:+.4f}; {np2}/{n2} > 0; p = {p2:.3f}")
        L.append("")
    else:
        L.append("Needs >= 2 density levels with >= 3 cube sizes.\n")

    L.append("## H2: how far does the best cube size move?\n")
    if g["x"].nunique() >= 3 and g["k"].nunique() >= 2:
        slope, xs, bs = vertex_regression(g)
        L.append("Pooled vertex regression, best cube size at each density: "
                 + ", ".join(f"level {k:g}: {H_MAX / 2 ** v:.1f} mm" for k, v in xs.items()))
        if len(bs):
            L.append(f"\n**dx*/ds = {slope:+.2f} octaves finer per doubling of the density level** "
                     f"(bootstrap over people 95% CI {np.percentile(bs, 2.5):+.2f} to {np.percentile(bs, 97.5):+.2f}; "
                     f"P(> 0) = {np.mean(bs > 0):.2f}). Prediction: > 0; variance-only theory 0.2-0.33.\n")
        L.append("Descriptive, per density:\n")
        bh, boots = best_h(g)
        L.append(tab(bh.round(1)))
        if lo in boots and hi in boots and len(boots[lo]) and len(boots[hi]):
            n = min(len(boots[lo]), len(boots[hi]))
            shift = boots[hi][:n] - boots[lo][:n]
            L.append(f"Vertex shift {hi} vs {lo}: {np.mean(shift):+.2f} halvings finer "
                     f"(bootstrap 95% CI {np.percentile(shift, 2.5):+.2f} to {np.percentile(shift, 97.5):+.2f}; "
                     f"P(shift > 0) = {np.mean(shift > 0):.2f}). Prediction: > 0.\n")
    L.append("## H3: does density matter once coverage is known?\n")
    if g["coverage"].notna().any() and g["density"].nunique() > 1 and g["subject"].nunique() > 2:
        r_a, r_b = loo_r2(g, False), loo_r2(g, True)
        L.append(f"Leave-one-person-out R^2: gain ~ log2 h * coverage = {r_a:.3f}; "
                 f"+ density = {r_b:.3f} (adds {r_b - r_a:+.3f}). Small or negative: density acts through coverage.\n")
    if out_png:
        plot(g, out_png)
        L.append(f"Figure: {os.path.basename(out_png)}\n")
    text = "\n".join(L)
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w") as f:
        f.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="reports/density_resolution.md")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--noise-only", action="store_true",
                    help="print only the seed SD of gain (for the staged-seeds rule), no effects")
    a = ap.parse_args(argv)
    if a.noise_only:
        sg = seed_gains(load_runs(a.results))
        sd = sg.groupby(["subject", "density", "h"])["gain"].std(ddof=1)
        print(f"seed SD of gain per cell: mean {np.sqrt((sd ** 2).mean()):.4f} over {sd.notna().sum()} cells "
              f"({sg['seed'].nunique()} seeds)")
        return
    png = None if a.no_plot else os.path.splitext(a.out)[0] + ".png"
    print(report(load_runs(a.results), a.out, png))


if __name__ == "__main__":
    main()
