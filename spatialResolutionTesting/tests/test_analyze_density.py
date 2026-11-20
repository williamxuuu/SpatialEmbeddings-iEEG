"""analyze_density on synthetic runs where the true relationship is known."""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import analyze_density as ad  # noqa: E402

PEOPLE = ["01", "02", "03", "04", "07", "10"]
H = [80, 57, 40, 28, 20, 14, 10, 5]
# coverage measured on the real electrodes (56 mm used for 57), k -> h -> coverage
COV = {1: [0.57, 0.52, 0.43, 0.36, 0.34, 0.25, 0.16, 0.04],
       2: [0.85, 0.81, 0.69, 0.61, 0.58, 0.45, 0.29, 0.08],
       5: [1.00, 0.98, 0.88, 0.87, 0.83, 0.71, 0.55, 0.18]}


def synth(true_gain, noise=0.002, seed=0, seeds=(0, 1), ks=(1, 2, 5)):
    rng = np.random.default_rng(seed)
    rows = []
    for p, k, s in itertools.product(PEOPLE, ks, seeds):
        base = 0.60 + rng.normal(0, 0.02)
        common = dict(fold=f"ho{p}", density=f"k{k}", seed=s, subject=f"btb{p}")
        rows.append(dict(run=f"{p}{k}{s}none", spatial="none", auc=base, auc_last=base,
                         frac_unknown_test=None, **common))
        for i, h in enumerate(H):
            x, c = np.log2(80 / h), COV[k][i]
            a = base + true_gain(x, c, k) + rng.normal(0, noise)
            rows.append(dict(run=f"{p}{k}{s}{h}", spatial=f"grid{h}", auc=a, auc_last=a,
                             frac_unknown_test=1 - c, **common))
    return pd.DataFrame(rows)


def _run(df, tmp_path):
    return ad.report(df, str(tmp_path / "r.md"), str(tmp_path / "r.png"))


def test_density_through_coverage(tmp_path):
    # finer helps only where held-out electrodes are covered: gain = 0.03 * x * cov - 0.01 * x
    t = _run(synth(lambda x, c, k: 0.03 * x * c - 0.01 * x), tmp_path)
    g = ad.gains(synth(lambda x, c, k: 0.03 * x * c - 0.01 * x))
    sl = ad.slopes(g)
    assert (sl["k5"] > sl["k1"]).all()
    assert "6/6 people > 0; exact sign-flip p = 0.031" in t
    pm, pp, nb = ad.perm_test(synth(lambda x, c, k: 0.03 * x * c - 0.01 * x))
    assert pm > 0 and pp < 0.001 and nb == 12
    bh, _ = ad.best_h(g, n_boot=200)
    assert bh.loc["k5", "vertex_h_mm"] < bh.loc["k1", "vertex_h_mm"]   # optimum finer with more data
    assert abs(ad.loo_r2(g, True) - ad.loo_r2(g, False)) < 0.02        # density adds ~nothing
    slope, xs, bs = ad.vertex_regression(g, n_boot=200)
    assert slope > 0.2 and (bs > 0).mean() > 0.95 and xs[5] > xs[1]
    assert os.path.exists(tmp_path / "r.png")


def test_null(tmp_path):
    t = _run(synth(lambda x, c, k: 0.01 * x, noise=0.01), tmp_path)
    m = float(t.split("Interaction slope(k=5) - slope(k=1) = ")[1].split(" ")[0])
    assert abs(m) < 0.01
    ps = [ad.perm_test(synth(lambda x, c, k: 0.01 * x, noise=0.05, seed=r), n_perm=500)[1] for r in range(40)]
    assert 0.0 <= np.mean(np.array(ps) < 0.05) <= 0.15          # ~5% false positives


def test_vertex_regression_null():
    # peaked at 20 mm for every density: no shift
    g = ad.gains(synth(lambda x, c, k: 0.02 - 0.005 * (x - 2) ** 2, noise=0.003))
    slope, xs, bs = ad.vertex_regression(g, n_boot=200)
    assert abs(slope) < 0.15 and all(abs(80 / 2 ** v - 20) < 4 for v in xs.values())


def test_density_beyond_coverage():
    g = ad.gains(synth(lambda x, c, k: 0.02 * x * (k == 5) + 0.005 * x * c))
    assert ad.loo_r2(g, True) - ad.loo_r2(g, False) > 0.1


def test_signflip_and_vertex():
    m, p, npos, n = ad.signflip(np.array([1, 2, 3, 4, 5, 6.0]))
    assert (npos, n) == (6, 6) and abs(p - 2 / 64) < 1e-9
    x = np.arange(5.0)
    assert abs(ad.vertex(x, -(x - 2.5) ** 2) - 2.5) < 1e-6
    assert ad.vertex(x, x) == 4.0          # monotone: best end
