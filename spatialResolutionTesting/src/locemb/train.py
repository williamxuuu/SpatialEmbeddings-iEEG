"""Train + evaluate one run.

    # one run from a config, with overrides
    python -m locemb.train --config configs/smoke.yaml spatial.atlas=grid20 seed=1

    # one line of a sweep file (this is what each Slurm array task does)
    python -m locemb.train --sweep-file runs/exp1_resolution.jsonl --index 7

Writes results/<exp_name>/<run_name>/{metrics.json, config.yaml, preds.npz, ckpt.pt}.

Model selection (train.select):
  best   test the epoch with the best validation AUC (validation = first half of
         the test recording, Neuroprobe's convention)
  last   test the final epoch. No test-person labels touch model selection.
Both are always computed and saved (metrics["test_best"], metrics["test_last"]);
`select` only decides which one fills the top-level test_auc.

Early stopping: never before train.min_epochs; after that, stop when the
validation AUC has not beaten its best by more than train.min_delta for
train.patience epochs. The pilot (patience 8, no minimum) stopped most runs
during the initial loss plateau (loss = ln 2 for 10-17 epochs), before the
model had started to learn.
If metrics.json already exists the run is skipped, so re-submitting a sweep
only re-runs what failed. If ckpt.pt exists (job was killed / preempted)
training resumes from it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .config import apply_overrides, load_config
from .data import Collator, build_data
from .models import build_model
from .spatial import build_assignment
from .utils import pick_device, set_seed, slurm_info, write_json


def _one_thread_worker(_):
    """DataLoader workers: one math thread each. Otherwise every worker starts a
    full BLAS/OpenMP pool (7 workers x 8 threads on 8 CPUs), which slowed data
    loading ~20x in the v2 smoke test."""
    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def evaluate(model, loader, device, n_classes):
    model.eval()
    ys, ps, subs = [], [], []
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            p = torch.softmax(model(b).float(), -1)
            ys.append(batch["y"].numpy()); ps.append(p.cpu().numpy()); subs += batch["subject"]
    y, p, subs = np.concatenate(ys), np.concatenate(ps), np.array(subs)

    def auc(yy, pp):
        if len(np.unique(yy)) < 2:
            return float("nan")
        if n_classes == 2:
            return float(roc_auc_score(yy, pp[:, 1]))
        return float(roc_auc_score(yy, pp, multi_class="ovr", average="macro", labels=list(range(n_classes))))

    out = {"auc": auc(y, p), "acc": float((p.argmax(1) == y).mean()), "n": int(len(y))}
    out["per_subject_auc"] = {s: auc(y[subs == s], p[subs == s]) for s in np.unique(subs)}
    out["per_subject_acc"] = {s: float((p[subs == s].argmax(1) == y[subs == s]).mean())
                              for s in np.unique(subs)}
    vals = [v for v in out["per_subject_auc"].values() if not math.isnan(v)]
    out["mean_subject_auc"] = float(np.mean(vals)) if vals else float("nan")
    out["_y"], out["_p"], out["_subject"] = y, p, subs
    return out


def extract_features(model, loader, device):
    """Frozen readout vectors (N, d), labels and subject ids for every window."""
    model.eval()
    fs, ys, subs = [], [], []
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            fs.append(model.features(b).float().cpu().numpy())
            ys.append(batch["y"].numpy()); subs += batch["subject"]
    return np.concatenate(fs), np.concatenate(ys), np.array(subs)


def calibrate(model, calib_loader, test_loader, device, fracs, C: float = 1.0) -> dict:
    """Calibration on a new person (train.calibrate, cross_subject only).

    The trained model is frozen; only a new linear readout (standardised features,
    L2 logistic regression with fixed C, chosen before any result) is fit on the
    first `frac` of the person's calibration windows (the first half of their test
    recording, in recording order) and scored on the test half. The location codes
    and features cannot adapt, so what transfers is what the model learned from the
    training people. frac -> {"auc", "n_calib", "per_subject_auc"}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xc, yc, sc = extract_features(model, calib_loader, device)
    Xt, yt, st = extract_features(model, test_loader, device)
    out = {}
    for frac in fracs:
        per, ns = {}, {}
        for s in np.unique(st):
            ic = np.nonzero(sc == s)[0]
            ic = ic[:max(2, int(round(float(frac) * len(ic))))]
            it = st == s
            if len(np.unique(yc[ic])) < 2 or len(np.unique(yt[it])) < 2:
                per[s] = float("nan"); ns[s] = int(len(ic)); continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
            clf.fit(Xc[ic], yc[ic])
            per[s] = float(roc_auc_score(yt[it], clf.predict_proba(Xt[it])[:, 1]))
            ns[s] = int(len(ic))
        vals = [v for v in per.values() if not math.isnan(v)]
        out[f"{float(frac):g}"] = {"auc": float(np.mean(vals)) if vals else float("nan"),
                                   "per_subject_auc": per, "n_calib": ns}
    return out


def _public(ev: dict) -> dict:
    return {k: v for k, v in ev.items() if not k.startswith("_")}


def permuted_table(table: dict, seed: int) -> dict:
    """The spatial table with each subject's electrodes shuffled among themselves:
    every electrode gets another electrode's location code. Used at test time
    only, to measure how much the trained model relies on location."""
    rng = np.random.default_rng(seed + 4242)
    out = {}
    for sid, t in table.items():
        perm = torch.from_numpy(rng.permutation(t["idx"].shape[0]))
        out[sid] = {k: v[perm] for k, v in t.items()}
    return out


def run(cfg: dict) -> dict:
    out_dir = os.path.join(cfg.get("results_dir", "results"), cfg["exp_name"], cfg["run_name"])
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path) and not cfg.get("force", False):
        print(f"[skip] {metrics_path} exists")
        with open(metrics_path) as f:
            return json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    t0 = time.time()
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = pick_device(cfg.get("device", "auto"))
    tr = cfg["train"]

    bundle = build_data(cfg, seed)
    assign = build_assignment(cfg["spatial"], bundle.subjects, bundle.train_subjects, seed)
    print(f"[data] {bundle.description}: train={len(bundle.train)} val={len(bundle.val)} "
          f"test={len(bundle.test)}  [spatial] {json.dumps(assign.stats)}", flush=True)

    collate = Collator(assign.table)
    p_drop = float(cfg["spatial"].get("code_dropout", 0.0) or 0.0)
    if p_drop and cfg["spatial"].get("type", "none") not in ("region", "electrode_id"):
        raise ValueError("spatial.code_dropout only applies to spatial.type region / electrode_id")
    train_collate = Collator(assign.table, code_dropout=p_drop)
    nw = int(tr.get("num_workers", 4))
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=tr.get("batch_size", 32), shuffle=shuffle,
                                        num_workers=nw, collate_fn=train_collate if shuffle else collate,
                                        pin_memory=device.type == "cuda",
                                        persistent_workers=nw > 0, drop_last=False,
                                        worker_init_fn=_one_thread_worker if nw > 0 else None)
    train_dl, val_dl, test_dl = mk(bundle.train, True), mk(bundle.val, False), mk(bundle.test, False)
    mon_dl = mk(bundle.monitor, False) if bundle.monitor is not None else None
    if bundle.val_subjects:
        print(f"[data] model selection on validation people {bundle.val_subjects}; "
              f"test-recording first half = monitor only", flush=True)

    n_samples = int(bundle.train[0]["x"].shape[-1])
    model = build_model(cfg, assign.n_regions, bundle.sfreq, bundle.n_classes, n_samples).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tr.get("lr", 1e-3),
                            weight_decay=tr.get("weight_decay", 1e-2))
    epochs = int(tr.get("epochs", 30))
    steps_total = epochs * max(1, len(train_dl))
    warm = int(tr.get("warmup_frac", 0.05) * steps_total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm))))
    use_amp = device.type == "cuda" and tr.get("amp", True)

    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    start_epoch, best, best_state, bad, history = 0, -1.0, None, 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best, best_state, bad, history = ck["epoch"] + 1, ck["best"], ck["best_state"], ck["bad"], ck["history"]
        print(f"[resume] from epoch {start_epoch}", flush=True)

    patience = int(tr.get("patience", 8))
    min_epochs = int(tr.get("min_epochs", 0))
    min_delta = float(tr.get("min_delta", 0.0))
    select = tr.get("select", "best")
    if select not in ("best", "last"):
        raise ValueError(f"train.select must be best or last, not {select!r}")
    # Smoke tests only: stop each epoch after this many mini-batches (null = full epoch).
    max_batches = tr.get("max_batches")
    max_batches = int(max_batches) if max_batches else None
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(start_epoch, epochs):
        model.train()
        tl, nb, te = 0.0, 0, time.time()
        for batch in train_dl:
            if max_batches is not None and nb >= max_batches:
                break
            b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model(b).float(), b["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr.get("grad_clip", 1.0))
            opt.step(); sched.step()
            tl += loss.item(); nb += 1
        val = evaluate(model, val_dl, device, bundle.n_classes)
        history.append({"epoch": epoch, "train_loss": tl / max(1, nb), "val_auc": val["auc"],
                        "batches": nb, "sec": time.time() - te})
        print(f"[epoch {epoch}] batches={nb} loss={tl / max(1, nb):.4f} val_auc={val['auc']:.4f} "
              f"({time.time() - te:.1f}s)", flush=True)
        history[-1]["val_mean_subject_auc"] = val["mean_subject_auc"]
        if mon_dl is not None:   # diagnostics only: never used to select or stop
            history[-1]["monitor_auc"] = evaluate(model, mon_dl, device, bundle.n_classes)["auc"]
        score = val["auc"] if not math.isnan(val["auc"]) else val["acc"]
        if score > best + min_delta:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history[-1]["best"] = True
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "best_state": best_state, "bad": bad,
                    "history": history}, ckpt_path)
        if bad >= patience and epoch + 1 >= min_epochs:
            print(f"[early stop] at epoch {epoch}", flush=True)
            break

    test_last = evaluate(model, test_dl, device, bundle.n_classes)
    last_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_best = evaluate(model, test_dl, device, bundle.n_classes)
    if select == "last":   # leave the model at its final epoch for the probe below
        model.load_state_dict(last_state)
    test = test_last if select == "last" else test_best

    # Calibration on the new person (cross_subject): freeze the selected model, refit
    # only a linear readout on part of the held-out person's first half, test on the
    # second half. Zero-shot (test) and calibrated numbers come from the same model.
    calib = None
    cal_fracs = tr.get("calibrate") or []
    if cal_fracs:
        if cfg["data"].get("split") != "cross_subject":
            raise ValueError("train.calibrate only applies to data.split=cross_subject")
        if not hasattr(model, "features"):
            raise ValueError("train.calibrate needs a model with .features() (transformer)")
        cal_ds = bundle.monitor if bundle.monitor is not None else bundle.val
        calib = calibrate(model, mk(cal_ds, False), test_dl, device, cal_fracs,
                          C=float(tr.get("calibrate_C", 1.0)))
        print("[calibrate] " + "  ".join(f"{k}: {v['auc']:.4f} (n={sum(v['n_calib'].values())})"
                                         for k, v in calib.items()), flush=True)

    # How much does the trained model rely on location? Re-test with each
    # subject's location codes shuffled among its electrodes.
    probe = None
    if cfg["spatial"].get("type", "none") != "none" and tr.get("location_probe", True):
        pdl = DataLoader(bundle.test, batch_size=tr.get("batch_size", 32), shuffle=False,
                         num_workers=nw, collate_fn=Collator(permuted_table(assign.table, seed)),
                         worker_init_fn=_one_thread_worker if nw > 0 else None)
        pe = evaluate(model, pdl, device, bundle.n_classes)
        probe = {"auc": pe["auc"], "mean_subject_auc": pe["mean_subject_auc"],
                 "per_subject_auc": pe["per_subject_auc"],
                 "drop_mean_subject_auc": test["mean_subject_auc"] - pe["mean_subject_auc"]}

    np.savez_compressed(os.path.join(out_dir, "preds.npz"),
                        y=test_best["_y"], p_best=test_best["_p"][:, 1:].squeeze(-1) if bundle.n_classes == 2 else test_best["_p"],
                        p_last=test_last["_p"][:, 1:].squeeze(-1) if bundle.n_classes == 2 else test_last["_p"],
                        subject=test_best["_subject"])
    metrics = {
        "exp_name": cfg["exp_name"], "run_name": cfg["run_name"], "seed": seed,
        "select": select,
        "test_auc": test["auc"], "test_acc": test["acc"], "test_n": test["n"],
        "test_per_subject_auc": test["per_subject_auc"],
        "test_mean_subject_auc": test["mean_subject_auc"],
        "test_best": _public(test_best), "test_last": _public(test_last),
        "location_probe": probe, "calibration": calib, "best_val": best,
        "n_train_windows": len(bundle.train),
        "train_window_frac": float(cfg["data"].get("train_window_frac", 1.0) or 1.0),
        "best_epoch": max([h["epoch"] for h in history if h.get("best")], default=None),
        "val_subjects": bundle.val_subjects, "code_dropout": p_drop,
        "spatial": assign.stats, "n_params": n_params, "epochs_run": len(history),
        "runtime_sec": time.time() - t0, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "host": platform.node(), "slurm": slurm_info(), "history": history,
    }
    write_json(metrics_path, metrics)
    if not cfg.get("keep_ckpt", False) and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"[done] select={select} test_auc={test['auc']:.4f} mean_subject_auc={test['mean_subject_auc']:.4f} "
          f"(best-val {test_best['mean_subject_auc']:.4f} / last {test_last['mean_subject_auc']:.4f}) "
          f"test_acc={test['acc']:.4f} -> {metrics_path}", flush=True)
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config (supports _base: inheritance)")
    ap.add_argument("--sweep-file", help="JSONL from `python -m locemb.sweep`")
    ap.add_argument("--index", type=int, help="line of the sweep file (0-based)")
    ap.add_argument("overrides", nargs="*", help="key=value overrides, e.g. spatial.atlas=grid20")
    a = ap.parse_args(argv)
    if a.sweep_file:
        if a.index is None:
            a.index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        with open(a.sweep_file) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        job = lines[a.index]
        cfg = load_config(job["config"], job["overrides"])
        cfg = apply_overrides(cfg, a.overrides)
    elif a.config:
        cfg = load_config(a.config, a.overrides)
    else:
        ap.error("give --config or --sweep-file")
    cfg.setdefault("exp_name", "scratch")
    cfg.setdefault("run_name", f"run_seed{cfg.get('seed', 0)}")
    run(cfg)


if __name__ == "__main__":
    sys.exit(main())
