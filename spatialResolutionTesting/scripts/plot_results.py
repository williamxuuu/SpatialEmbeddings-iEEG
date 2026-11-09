"""Plots from collect_results.py output.

    python scripts/plot_results.py results/exp1_resolution --kind curve   # resolution x performance
    python scripts/plot_results.py results/exp2_soft_regions --kind dots  # every setting, ranked

Saves a PNG next to the CSVs. summary.csv is the table view of the same numbers.
"""
import argparse
import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, fixed order


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def curve(df, out):
    models = sorted(df.model.unique()) if "model" in df else ["all"]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.4), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, model in zip(axes[0], models):
        style(ax)
        g = df[df.model == model] if "model" in df else df
        grp = g.spatial.map(group_of)
        shuffled = grp == "control"
        hard = grp == "hard"                    # the curve uses one-region-per-electrode settings only
        grid = g.spatial.str.match(r"grid\d") & hard
        atlas = hard & ~grid
        for mask, color, label in [(grid, BLUE, "Cubes (no anatomy)"), (atlas, ORANGE, "Brain atlases")]:
            s = g[mask].sort_values("n_regions")
            ax.errorbar(s.n_regions, s.auc_mean, yerr=s.auc_sd.fillna(0), color=color, lw=2,
                        marker="o", ms=8, mec=SURFACE, mew=2, capsize=0, elinewidth=1.2, label=label)
            for _, r in s.iterrows():
                ax.annotate(r.spatial, (r.n_regions, r.auc_mean), textcoords="offset points",
                            xytext=(5, 6), fontsize=7.5, color=INK2)
        s = g[shuffled]
        if len(s):
            ax.scatter(s.n_regions, s.auc_mean, s=64, facecolors="none", edgecolors=INK2, lw=1.5,
                       label="Control (location link broken)", zorder=3)
            for _, r in s.iterrows():
                ax.annotate(r.spatial, (r.n_regions, r.auc_mean), textcoords="offset points",
                            xytext=(5, -11), fontsize=7.5, color=INK2)
        for name, ls in [("none", "--"), ("coords", ":")]:
            r = g[g.spatial == name]
            if len(r):
                y = float(r.auc_mean.iloc[0])
                ax.axhline(y, color=MUTED, ls=ls, lw=1.5)
                ax.annotate({"none": "No location", "coords": "xyz coordinates"}[name],
                            (0, y), xycoords=("axes fraction", "data"), xytext=(4, 4),
                            textcoords="offset points", ha="left", fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Number of regions (log scale)  →  finer", color=INK2, fontsize=9)
        ax.set_title(f"{model}", color=INK, fontsize=11, loc="left")
    axes[0][0].set_ylabel("Test AUC (mean ± SD over seeds)", color=INK2, fontsize=9)
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.suptitle("How detailed should location be?", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=SURFACE)


SOFT_TAGS = ("sphere", "gauss", "trilinear", "prob_soft")


def group_of(name):
    if name in ("none", "coords"):
        return "reference"
    if "shuffled" in name or "randnbr" in name:
        return "control"
    if any(t in name for t in SOFT_TAGS):
        return "soft"
    return "hard"   # plain atlas/grid names use mode=hard


def dots(df, out):
    models = sorted(df.model.unique()) if "model" in df else ["all"]
    colors = {"soft": BLUE, "hard": ORANGE, "control": AQUA, "reference": MUTED}
    n = df.spatial.nunique()
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 0.32 * n + 1.6),
                             facecolor=SURFACE, squeeze=False)
    for ax, model in zip(axes[0], models):
        style(ax)
        g = (df[df.model == model] if "model" in df else df).sort_values("auc_mean")
        y = np.arange(len(g))
        for grp, color in colors.items():
            m = g.spatial.map(group_of).values == grp
            if m.any():
                ax.errorbar(g.auc_mean[m], y[m], xerr=g.auc_sd.fillna(0)[m], fmt="o", ms=7, color=color,
                            mec=SURFACE, mew=1.5, elinewidth=1.2, label=grp)
        ax.set_yticks(y, g.spatial, fontsize=8, color=INK2)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Test AUC (mean ± SD over seeds)", color=INK2, fontsize=9)
        ax.set_title(model, color=INK, fontsize=11, loc="left")
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=SURFACE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--kind", choices=["curve", "dots"], default="dots")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.exp_dir, "summary.csv"))
    out = os.path.join(a.exp_dir, f"{a.kind}.png")
    (curve if a.kind == "curve" else dots)(df, out)
    print("saved", out)
