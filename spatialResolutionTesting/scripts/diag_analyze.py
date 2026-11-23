"""Analyse the band-power diagnostic (scripts/diag_bandpower.py).

    python scripts/diag_analyze.py reports/diag/bandpower > reports/diag/bandpower_report.txt

A  per-electrode speech AUC of each band, per session (how many electrodes respond, which sign)
B  pooled (mean over electrodes) band power: speech AUC per session, sign across people/movies
C  linear cross-subject transfer of pooled features (5 people -> held-out person), per fold
D  across-person similarity of electrode responses vs MNI distance
E  atlas transfer: does a region's mean response in the training people predict the
   held-out person's electrode response in that region? (per atlas / resolution)
F  drift: pooled power, validation half vs test half of the held-out recordings
"""

from __future__ import annotations

import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BANDS = ["4-8", "8-16", "16-32", "32-64", "64-128", "128-200"]
HG = 4  # 64-128 Hz
HELD_OUT = ["btb01", "btb02", "btb03", "btb04", "btb07", "btb10"]


def auc_cols(F, y):
    """AUC of every column of F (N, ...) for labels y -> array of F.shape[1:]."""
    N = len(y)
    flat = F.reshape(N, -1).astype(np.float64)
    r = np.argsort(np.argsort(flat, 0), 0) + 1.0
    n1 = y.sum(); n0 = N - n1
    return ((r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)).reshape(F.shape[1:])


def load(d):
    S = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        sid, part = os.path.basename(f)[:-4].split("_")
        S[(sid, part)] = dict(feat=z["feat"].astype(np.float32), y=z["y"].astype(int), start=z["start"],
                              electrodes=list(z["electrodes"]), coords=z["coords"])
    return S


def main(d):
    S = load(d)
    sids = sorted({k[0] for k in S})
    # sessions: "A" = the cross_session train trial, "B" = the test trial (val + test halves)
    sess = {}
    for s in sids:
        sess[(s, "A")] = S[(s, "train")]
        v, t = S[(s, "val")], S[(s, "test")]
        sess[(s, "B")] = dict(feat=np.concatenate([v["feat"], t["feat"]]), y=np.concatenate([v["y"], t["y"]]),
                              start=np.concatenate([v["start"], t["start"]]), electrodes=v["electrodes"],
                              coords=v["coords"], nval=len(v["y"]))
    np.set_printoptions(precision=3, suppress=True, linewidth=160)

    print("=" * 100)
    print("A. Per-electrode speech AUC (within session). Columns = bands", BANDS)
    print("   frac electrodes with AUC > 0.6 / < 0.4, and the median, per band")
    el_auc = {}
    for k, z in sess.items():
        a = auc_cols(z["feat"], z["y"])          # (E, B)
        el_auc[k] = a
        print(f"  {k[0]} sess {k[1]}  E={a.shape[0]:3d} N={len(z['y'])}  >0.6: {(a > 0.6).mean(0)}  <0.4: {(a < 0.4).mean(0)}  med: {np.median(a, 0)}")
    print("  Sign agreement of electrode HG AUC between a person's two sessions (same electrodes):")
    for s in sids:
        a, b = el_auc[(s, "A")][:, HG] - 0.5, el_auc[(s, "B")][:, HG] - 0.5
        print(f"    {s}: r = {np.corrcoef(a, b)[0, 1]:.3f}   same sign (|d|>.05 both) = "
              f"{np.mean(np.sign(a) == np.sign(b))[()] if True else 0:.3f}")

    print("=" * 100)
    print("B. Pooled band power (mean over electrodes): speech AUC per session")
    for k, z in sess.items():
        print(f"  {k[0]} sess {k[1]}: {auc_cols(z['feat'].mean(1), z['y'])}")

    print("=" * 100)
    print("C. Linear cross-subject transfer of pooled features (train on other 5 people, both sessions;")
    print("   features z-scored per session; test on held-out person's test-trial val / test halves)")
    def pooled(z):
        f = z["feat"].mean(1)
        return (f - f.mean(0)) / (f.std(0) + 1e-6)
    for ho in HELD_OUT:
        if ho not in sids:
            continue
        Xtr = np.concatenate([pooled(sess[k]) for k in sess if k[0] != ho])
        ytr = np.concatenate([sess[k]["y"] for k in sess if k[0] != ho])
        zt = sess[(ho, "B")]; Xt = pooled(zt); nv = zt["nval"]
        for name, cols in (("HG only", [HG]), ("all bands", list(range(len(BANDS))))):
            m = LogisticRegression(max_iter=2000).fit(Xtr[:, cols], ytr)
            p = m.decision_function(Xt[:, cols])
            print(f"  {ho} {name:9s}: val {roc_auc_score(zt['y'][:nv], p[:nv]):.3f}  test {roc_auc_score(zt['y'][nv:], p[nv:]):.3f}"
                  + (f"   coef {m.coef_.ravel()}" if name == "all bands" else ""))
    print("  (reference: the same with features fit within the held-out person = upper bound for pooled)")
    for ho in HELD_OUT:
        if ho not in sids:
            continue
        zt = sess[(ho, "B")]; Xt = pooled(zt); nv = zt["nval"]
        m = LogisticRegression(max_iter=2000).fit(Xt[:nv], zt["y"][:nv])
        print(f"  {ho} within (fit val, score test): {roc_auc_score(zt['y'][nv:], m.decision_function(Xt[nv:])):.3f}")

    print("=" * 100)
    print("D. Across-person similarity of electrode HG response (AUC-0.5) vs MNI distance")
    print("   (electrode response averaged over the person's two sessions)")
    R, C, P = [], [], []
    for i, s in enumerate(sids):
        r = (el_auc[(s, "A")][:, HG] + el_auc[(s, "B")][:, HG]) / 2 - 0.5
        R.append(r); C.append(sess[(s, "A")]["coords"]); P += [i] * len(r)
    R, C, P = np.concatenate(R), np.concatenate(C), np.array(P)
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    cross = P[:, None] != P[None]
    same = ~cross & ~np.eye(len(R), dtype=bool)
    bins = [0, 3, 5, 8, 12, 20, 40, 200]
    print(f"   {'bin mm':>10s} {'cross-person r':>15s} {'n pairs':>8s} {'within-person r':>16s} {'n pairs':>8s}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        out = []
        for msk in (cross, same):
            ii, jj = np.where(msk & (D >= lo) & (D < hi))
            out.append((np.corrcoef(R[ii], R[jj])[0, 1] if len(ii) > 10 else np.nan, len(ii)))
        print(f"   {lo:3d}-{hi:<4d}mm  {out[0][0]:15.3f} {out[0][1]:8d} {out[1][0]:16.3f} {out[1][1]:8d}")

    print("=" * 100)
    print("E. Atlas transfer: predict each held-out electrode's HG response from the mean response of")
    print("   training-person electrodes with the same (hard) region. r over held-out electrodes that")
    print("   have a seen region; coverage = share of held-out electrodes with a seen region.")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from locemb.spatial.assign import raw_votes
        atlases = ["lobes", "destrieux", "schaefer400", "schaefer1000", "grid80", "grid40", "grid20", "grid10", "grid5"]
        lab = {}
        for a in atlases:
            try:
                lab[a] = {s: [max(v, key=v.get) if v else None for v in
                              raw_votes({"type": "region", "atlas": a, "mode": "hard", "fallback_mm": 5,
                                         "include_subcortical": True},
                                        SimpleNamespace(coords=sess[(s, "A")]["coords"]))] for s in sids}
            except Exception as ex:
                print(f"   [skip {a}: {ex}]")
        rs = {s: (el_auc[(s, "A")][:, HG] + el_auc[(s, "B")][:, HG]) / 2 - 0.5 for s in sids}
        print(f"   {'atlas':12s} " + " ".join(f"{ho:>16s}" for ho in HELD_OUT) + "   mean r")
        for a, L in lab.items():
            row, rr = [], []
            for ho in HELD_OUT:
                tr = {}
                for s in sids:
                    if s == ho:
                        continue
                    for reg, r in zip(L[s], rs[s]):
                        if reg is not None:
                            tr.setdefault(reg, []).append(r)
                pred = np.array([np.mean(tr[reg]) if reg in tr else np.nan for reg in L[ho]])
                ok = ~np.isnan(pred)
                r = np.corrcoef(pred[ok], rs[ho][ok])[0, 1] if ok.sum() > 5 else np.nan
                rr.append(r); row.append(f"r={r:+.2f} cov={ok.mean():.2f}")
            print(f"   {a:12s} " + " ".join(f"{x:>16s}" for x in row) + f"   {np.nanmean(rr):+.3f}")
    except Exception as ex:
        print("   E failed:", ex)

    print("=" * 100)
    print("F. Drift in the held-out recordings: pooled band power, val half minus test half (in SD),")
    print("   per class, and speech AUC of pooled HG in each half")
    for s in sids:
        z = sess[(s, "B")]; nv = z["nval"]; f = z["feat"].mean(1); y = z["y"]
        sd = f.std(0) + 1e-6
        dpos = (f[:nv][y[:nv] == 1].mean(0) - f[nv:][y[nv:] == 1].mean(0)) / sd
        dneg = (f[:nv][y[:nv] == 0].mean(0) - f[nv:][y[nv:] == 0].mean(0)) / sd
        print(f"  {s}: speech {dpos}  non-speech {dneg}  HG AUC val {auc_cols(f[:nv, HG:HG+1], y[:nv])[0]:.3f} "
              f"test {auc_cols(f[nv:, HG:HG+1], y[nv:])[0]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
