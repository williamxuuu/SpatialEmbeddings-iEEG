"""Check finished run folders and print OK / FAIL per run, then "N/M OK".

    python scripts/check_run.py results/smoke_real/* --expect-cuda --epochs 2

For each results/<exp>/<run>/ folder:
  - metrics.json and config.yaml exist and parse
  - device is cuda and a GPU name was recorded           (--expect-cuda)
  - exactly N epochs ran                                 (--epochs N)
  - every epoch ran >0 mini-batches (== train.max_batches if that was set)
  - every train loss and val AUC is finite; test AUC is finite and in [0, 1]
  - test set is non-empty; per-subject AUCs are present
  - ckpt.pt (if train.keep_ckpt) loads on CPU, is from the last epoch, all weights finite
  - the run's Slurm log (logs/*_<array job>_<task>.out) has no Traceback / CUDA error /
    out-of-memory / NaN-loss lines
Exit code 0 only if every run is OK.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import yaml

BAD_LOG = re.compile(r"Traceback|CUDA error|cuda runtime error|out of memory|OutOfMemoryError|"
                     r"NCCL error|Segmentation fault|Bus error|core dumped|illegal memory access|"
                     r"loss=nan|CANCELLED|DUE TO TIME LIMIT|oom-kill", re.I)


def find_log(slurm: dict, logs_dir: str) -> str | None:
    a, t = slurm.get("SLURM_ARRAY_JOB_ID"), slurm.get("SLURM_ARRAY_TASK_ID")
    pats = [f"*_{a}_{t}.out"] if a and t else []
    if slurm.get("SLURM_JOB_ID"):
        pats.append(f"*_{slurm['SLURM_JOB_ID']}.out")
    for p in pats:
        hits = glob.glob(os.path.join(logs_dir, p))
        if hits:
            return hits[0]
    return None


def check_one(d: str, a) -> list[str]:
    errs = []
    mp, cp = os.path.join(d, "metrics.json"), os.path.join(d, "config.yaml")
    if not os.path.exists(mp):
        return ["no metrics.json (run unfinished or crashed)"]
    try:
        m = json.load(open(mp))
    except Exception as e:  # noqa: BLE001
        return [f"metrics.json unreadable: {e}"]
    cfg = {}
    if not os.path.exists(cp):
        errs.append("no config.yaml")
    else:
        cfg = yaml.safe_load(open(cp)) or {}
    tr = cfg.get("train", {})

    if a.expect_cuda:
        if m.get("device") != "cuda":
            errs.append(f"device={m.get('device')} (expected cuda)")
        if not m.get("gpu"):
            errs.append("no GPU name recorded")
    hist = m.get("history", [])
    if a.epochs is not None and (m.get("epochs_run") != a.epochs or len(hist) != a.epochs):
        errs.append(f"epochs_run={m.get('epochs_run')} history={len(hist)} (expected {a.epochs})")
    want_b = tr.get("max_batches")
    for h in hist:
        nb = h.get("batches")
        if nb is None:
            errs.append(f"epoch {h['epoch']}: no batch count (old train.py?)")
        elif nb <= 0 or (want_b and nb != int(want_b)):
            errs.append(f"epoch {h['epoch']}: {nb} mini-batches (expected {want_b or '>0'})")
        if not math.isfinite(h.get("train_loss", float("nan"))):
            errs.append(f"epoch {h['epoch']}: train loss {h.get('train_loss')}")
        if not math.isfinite(h.get("val_auc", float("nan"))):
            errs.append(f"epoch {h['epoch']}: val AUC {h.get('val_auc')}")
    auc = m.get("test_auc")
    if auc is None or not math.isfinite(auc) or not 0 <= auc <= 1:
        errs.append(f"test_auc={auc}")
    if not m.get("test_n"):
        errs.append("empty test set")
    if not m.get("test_per_subject_auc"):
        errs.append("no per-subject test AUCs")

    ck = os.path.join(d, "ckpt.pt")
    if cfg.get("keep_ckpt"):
        if not os.path.exists(ck):
            errs.append("keep_ckpt=true but no ckpt.pt")
        else:
            try:
                import torch
                c = torch.load(ck, map_location="cpu", weights_only=False)
                if c["epoch"] != len(hist) - 1:
                    errs.append(f"ckpt epoch {c['epoch']} != last epoch {len(hist) - 1}")
                for part in ("model", "best_state"):
                    sd = c.get(part)
                    if sd is None:
                        errs.append(f"ckpt has no {part}")
                    elif not all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()):
                        errs.append(f"ckpt {part} has non-finite weights")
            except Exception as e:  # noqa: BLE001
                errs.append(f"ckpt.pt does not load: {e}")

    log = find_log(m.get("slurm", {}), a.logs_dir)
    if log is None:
        if not a.no_log:
            errs.append(f"Slurm log not found in {a.logs_dir}/ (use --no-log for local runs)")
    else:
        bad = [ln.strip() for ln in open(log, errors="replace") if BAD_LOG.search(ln)]
        if bad:
            errs.append(f"{os.path.basename(log)}: {len(bad)} error line(s), first: {bad[0][:160]}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders, e.g. results/smoke_real/*")
    ap.add_argument("--expect-cuda", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--no-log", action="store_true", help="don't require a Slurm log")
    a = ap.parse_args()
    runs = sorted(r.rstrip("/") for r in a.runs if os.path.isdir(r))
    n_ok = 0
    for r in runs:
        errs = check_one(r, a)
        if errs:
            print(f"FAIL {r}")
            for e in errs:
                print(f"       - {e}")
        else:
            n_ok += 1
            m = json.load(open(os.path.join(r, "metrics.json")))
            print(f"OK   {r}  [{m['gpu']}, {m['epochs_run']} ep, {m['runtime_sec']:.0f}s]")
    print(f"\n{n_ok}/{len(runs)} OK")
    return 0 if runs and n_ok == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
