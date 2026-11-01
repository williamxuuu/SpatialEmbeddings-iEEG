"""Train + evaluate one run.

    # one run from a config, with overrides
    python -m locemb.train --config configs/smoke.yaml spatial.atlas=grid20 seed=1

    # one line of a sweep file (this is what each Slurm array task does)
    python -m locemb.train --sweep-file runs/exp1_resolution.jsonl --index 7

Writes results/<exp_name>/<run_name>/{metrics.json, config.yaml, ckpt.pt}.
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
    nw = int(tr.get("num_workers", 4))
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=tr.get("batch_size", 32), shuffle=shuffle,
                                        num_workers=nw, collate_fn=collate,
                                        pin_memory=device.type == "cuda",
                                        persistent_workers=nw > 0, drop_last=False)
    train_dl, val_dl, test_dl = mk(bundle.train, True), mk(bundle.val, False), mk(bundle.test, False)

    model = build_model(cfg, assign.n_regions, bundle.sfreq, bundle.n_classes).to(device)
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
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(start_epoch, epochs):
        model.train()
        tl, nb, te = 0.0, 0, time.time()
        for batch in train_dl:
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
                        "sec": time.time() - te})
        print(f"[epoch {epoch}] loss={tl / max(1, nb):.4f} val_auc={val['auc']:.4f} "
              f"({time.time() - te:.1f}s)", flush=True)
        score = val["auc"] if not math.isnan(val["auc"]) else val["acc"]
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "best_state": best_state, "bad": bad,
                    "history": history}, ckpt_path)
        if bad >= patience:
            print(f"[early stop] at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_dl, device, bundle.n_classes)
    metrics = {
        "exp_name": cfg["exp_name"], "run_name": cfg["run_name"], "seed": seed,
        "test_auc": test["auc"], "test_acc": test["acc"], "test_n": test["n"],
        "test_per_subject_auc": test["per_subject_auc"], "best_val": best,
        "spatial": assign.stats, "n_params": n_params, "epochs_run": len(history),
        "runtime_sec": time.time() - t0, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "host": platform.node(), "slurm": slurm_info(), "history": history,
    }
    write_json(metrics_path, metrics)
    if not cfg.get("keep_ckpt", False) and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"[done] test_auc={test['auc']:.4f} test_acc={test['acc']:.4f} -> {metrics_path}", flush=True)
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
