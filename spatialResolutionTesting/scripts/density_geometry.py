"""Electrode density vs spatial resolution, from coordinates alone (no training).

    python scripts/density_geometry.py            # writes reports/density_geometry.{md,csv,png}

For every encoding (atlases coarse -> fine, cubes 80 -> 5 mm; hard and soft
Gaussian sigma 3 mm) and every number of training people k = 1, 2, 5, this runs
the same assignment code training uses (locemb.spatial.assign.build_assignment),
with the same nested training subsets the density_resolution sweep draws
(held-out person x train_subset_seed 0-5), and reports:

  n_regions      regions the training people occupy (the embedding table that gets trained)
  e_per_region   median training electrodes per occupied region
  frac_lt3       share of occupied regions with < 3 training electrodes
  coverage       share of the held-out person's weight on regions training saw
                 (1 - weight on the shared <unknown> vector)

These are the x-axes of the density x resolution story: what "more data" changes
at each resolution, and where soft weights should matter (the coverage soft adds).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

PAIRS = [(1, 2), (2, 4), (3, 1), (4, 1), (7, 1), (10, 1)]
ATLASES = ["lobes", "ho", "destrieux", "schaefer100", "schaefer200", "schaefer400", "schaefer1000"]
CUBES = [80, 57, 40, 28, 20, 14, 10, 5]
KS = [1, 2, 5]
SEEDS = range(6)


def load_subjects(root: str) -> dict:
    os.environ["ROOT_DIR_BRAINTREEBANK"] = root
    import neuroprobe as npb
    import torch
    from locemb.data.base import SubjectInfo
    out = {}
    for s, _ in PAIRS:
        sub = npb.BrainTreebankSubject(s, cache=False, dtype=torch.float32, coordinates_type="mni")
        labels = list(sub.electrode_labels)
        xyz = np.asarray(sub.get_electrode_coordinates(), dtype=np.float64)
        out[f"btb{s:02d}"] = SubjectInfo(f"btb{s:02d}", xyz, labels)
    return out


def train_people(test: int, k: int, seed: int) -> list[int]:
    """Same nested draw as the neuroprobe adapter (data.n_train_subjects)."""
    eligible = sorted(s for s, _ in PAIRS if s != test)   # all 5 eligible (checked: no shared movies)
    key = f"train_{[test]}_{seed}"
    order = np.random.default_rng(zlib.crc32(key.encode())).permutation(eligible)
    return [int(x) for x in order[:k]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Brain Treebank root (default: config / $ROOT_DIR_BRAINTREEBANK)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "reports", "density_geometry"))
    ap.add_argument("--from-csv", action="store_true", help="redraw tables/figure from <out>.csv")
    a = ap.parse_args(argv)
    if a.from_csv:
        return summarize(pd.read_csv(a.out + ".csv"), a.out)
    import yaml
    from locemb.spatial import assign
    root = a.root or yaml.safe_load(open(os.path.join(HERE, "..", "configs", "neuroprobe.yaml")))["data"].get("root") \
        or os.environ["ROOT_DIR_BRAINTREEBANK"]
    subjects = load_subjects(root)

    cache, raw = {}, assign.raw_votes            # region lookups depend only on (spec, person)

    def cached(spec, subj):
        key = (json.dumps(spec, sort_keys=True), subj.subject_id)
        if key not in cache:
            cache[key] = raw(spec, subj)
        return cache[key]
    assign.raw_votes = cached

    specs = {}
    for at in ATLASES:
        specs[at] = {"type": "region", "atlas": at, "mode": "hard"}
        specs[f"{at}_gauss3"] = {"type": "region", "atlas": at, "mode": "gaussian", "sigma_mm": 3}
    for h in CUBES:
        specs[f"grid{h}"] = {"type": "region", "atlas": f"grid{h}", "mode": "hard"}

    rows = []
    for name, spec in specs.items():
        for k in KS:
            for t, _ in PAIRS:
                for seed in (SEEDS if k < 5 else [0]):
                    tr = train_people(t, k, seed)
                    sub = {f"btb{s:02d}": subjects[f"btb{s:02d}"] for s in tr + [t]}
                    st = assign.build_assignment(spec, sub, {f"btb{s:02d}" for s in tr}, seed=seed).stats
                    rows.append({"encoding": name, "soft": name.endswith("_gauss3"),
                                 "base": name.replace("_gauss3", ""), "k": k, "test": t, "seed": seed,
                                 "n_regions": st["n_regions"], "e_per_region": st["electrodes_per_region_median"],
                                 "frac_lt3": st["frac_regions_lt3_electrodes"],
                                 "coverage": 1 - st["frac_weight_unknown_test_subjects"]})
        print(f"done {name}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out + ".csv", index=False)
    summarize(df, a.out)


def summarize(df: pd.DataFrame, out: str):

    m = df.groupby(["encoding", "k"])[["n_regions", "e_per_region", "frac_lt3", "coverage"]].mean().unstack("k")
    order = ATLASES + [f"{x}_gauss3" for x in ATLASES] + [f"grid{h}" for h in CUBES]
    m = m.reindex(order)
    hard = df[~df["soft"] & df["base"].isin(ATLASES)].groupby(["base", "k"])["coverage"].mean().unstack()
    soft = df[df["soft"]].groupby(["base", "k"])["coverage"].mean().unstack()
    gain = (soft - hard).reindex(ATLASES)
    tab = lambda t: "```\n" + t.to_string() + "\n```\n"  # noqa: E731
    L = ["# Electrode density vs spatial resolution (coordinates only)\n",
         "Same assignment code and nested training subsets as the density_resolution sweep; "
         "mean over 6 held-out people (and 6 subset draws for k = 1, 2). k = training people.\n",
         "## Regions training occupies\n", tab(m["n_regions"].round(1)),
         "## Median training electrodes per occupied region\n", tab(m["e_per_region"].round(1)),
         "## Share of occupied regions with < 3 training electrodes\n", tab(m["frac_lt3"].round(2)),
         "## Coverage: held-out weight on regions training saw\n", tab(m["coverage"].round(3)),
         "## Coverage soft adds (gauss3 - hard)\n", tab(gain.round(3))]
    with open(out + ".md", "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in KS:
        for fam, mk in ((ATLASES, "o"), ([f"grid{h}" for h in CUBES], "s")):
            r = m.reindex(fam)
            ax[0].plot(r[("n_regions", k)], r[("coverage", k)], marker=mk, label=f"k={k} {'atlas' if mk == 'o' else 'cubes'}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("regions training occupies (log)")
    ax[0].set_ylabel("coverage of held-out electrodes")
    ax[0].set_title("Finer resolution -> less coverage; more people -> more")
    ax[0].legend(fontsize=7)
    for k in KS:
        pts = pd.DataFrame({"n": m.reindex(ATLASES)[("n_regions", k)], "g": gain[k]}).sort_values("n")
        ax[1].plot(pts["n"], pts["g"], marker="o", label=f"k={k}")
    ax[1].set_xscale("log")
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set_xlabel("regions training occupies (log), atlases")
    ax[1].set_ylabel("coverage soft adds (gauss3 - hard)")
    ax[1].set_title("Where soft weights can help")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=150)


if __name__ == "__main__":
    main()
