"""Cluster smoke test: proves the GPU stack, a few training mini-batches,
checkpoint saving/loading and the evaluation code all work. Run INSIDE a GPU
job (slurm/smoke.sbatch via slurm/submit_smoke.sh), never on a login node.

    python scripts/smoke_test.py            # synthetic data (no download needed)
    python scripts/smoke_test.py --real     # real Brain Treebank (needs the cache built)

Stages (each prints PASS/FAIL; the last line is "ALL STAGES PASSED" or
"FAILED STAGES: ..."; exit code 0 only if everything passed):

  1 env      torch/CUDA versions, driver, GPU name, bf16 support
  2 gpu      fp32 + bf16 matmul and conv on the GPU agree with the CPU; no NaN
  3 atlases  the pilot's atlases load from the offline cache (compute nodes may lack internet)
  4 eval     evaluate() returns the right AUC/accuracy on hand-built predictions
             (perfect -> 1, inverted -> 0, random -> equals sklearn, per subject, 3-class)
  5 train    a real train.run(): 2 epochs x 20 mini-batches on the GPU, finite losses,
             metrics.json written with device=cuda
  6 weights  ckpt.pt reloads; weights changed from init; reloaded best weights reproduce
             the saved test AUC exactly; a fresh model + optimizer resumes from it

The AUCs from stage 5 mean nothing (20 batches); only pass/fail matters.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def stage(name):
    def deco(fn):
        def wrapped(*a, **k):
            print(f"\n===== STAGE {name} =====", flush=True)
            t = time.time()
            try:
                out = fn(*a, **k)
                RESULTS.append((name, True, f"{time.time() - t:.1f}s"))
                print(f"[PASS] {name} ({time.time() - t:.1f}s)", flush=True)
                return out
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
                return None
        return wrapped
    return deco


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}", flush=True)


@stage("1-env")
def stage_env():
    print(f"  python {sys.version.split()[0]}  torch {torch.__version__}  "
          f"built for CUDA {torch.version.cuda}  cuDNN {torch.backends.cudnn.version()}")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        print(f"  nvidia-smi: {smi.stdout.strip() or smi.stderr.strip()}")
    except Exception as e:  # noqa: BLE001
        print(f"  nvidia-smi not runnable: {e}")
    check(torch.cuda.is_available(), "torch.cuda.is_available() (driver works with this torch build)")
    n = torch.cuda.device_count()
    check(n >= 1, f"{n} GPU(s) visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"  GPU 0: {name}, compute capability {cap[0]}.{cap[1]}")
    check(torch.cuda.is_bf16_supported(), "bf16 supported (training uses bf16 autocast)")
    if cap[0] < 8:
        print("  WARN: pre-Ampere GPU: bf16 is emulated and slow; pin an A100/H100 in LOCEMB_GRES")


@stage("2-gpu")
def stage_gpu():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(512, 512, generator=g), torch.randn(512, 512, generator=g)
    ref = a @ b
    torch.backends.cuda.matmul.allow_tf32 = False
    out = (a.cuda() @ b.cuda()).cpu()
    check(torch.allclose(out, ref, rtol=1e-4, atol=1e-3), f"fp32 matmul GPU==CPU (max err {(out - ref).abs().max():.2e})")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outb = (a.cuda() @ b.cuda()).float().cpu()
    rel = ((outb - ref).norm() / ref.norm()).item()
    check(rel < 2e-2, f"bf16 autocast matmul close to fp32 (rel err {rel:.2e})")
    x = torch.randn(4, 8, 1024, generator=g)
    conv = torch.nn.Conv1d(8, 16, 7)
    yc = conv(x)
    yg = conv.cuda()(x.cuda()).cpu()
    check(torch.allclose(yg, yc, rtol=1e-3, atol=1e-3), "conv1d GPU==CPU")
    # backward on GPU
    w = torch.randn(64, 64, device="cuda", requires_grad=True)
    loss = (torch.randn(32, 64, device="cuda") @ w).pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    check(torch.isfinite(w.grad).all().item(), "backward pass on GPU gives finite gradients")
    print(f"  GPU memory in use: {torch.cuda.memory_allocated() / 1e6:.1f} MB")


@stage("3-atlases")
def stage_atlases(names):
    import warnings
    warnings.filterwarnings("ignore")
    from locemb.spatial.atlases import get_labeler
    print(f"  LOCEMB_ATLAS_DIR={os.environ.get('LOCEMB_ATLAS_DIR')}")
    pt = np.array([[-50.0, -20.0, 5.0], [40.0, 20.0, 30.0]])  # left temporal, right frontal (MNI mm)
    for n in names:
        lb = get_labeler(n, True)
        codes = lb(pt)
        check(len(lb.names) > 1, f"{n}: {len(lb.names)} regions; test points -> "
              f"{[lb.name_of(int(c)) for c in codes]}")


@stage("4-eval")
def stage_eval():
    from sklearn.metrics import roc_auc_score
    from locemb.train import evaluate

    class ScoreModel(torch.nn.Module):
        """Returns given logits: batch['x'] holds them directly."""
        def forward(self, b):
            return b["x"]

    def loader(logits, y, subs, bs=16):
        out = []
        for i in range(0, len(y), bs):
            out.append({"x": torch.as_tensor(logits[i:i + bs], dtype=torch.float32),
                        "y": torch.as_tensor(y[i:i + bs]), "subject": list(subs[i:i + bs])})
        return out

    dev = torch.device("cuda")
    m = ScoreModel().to(dev)
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    subs = np.array(["sA", "sB", "sC"])[rng.integers(0, 3, n)]
    two = lambda s: np.stack([np.zeros_like(s), s], 1)  # p(class 1) = sigmoid(s), monotone in s

    r = evaluate(m, loader(two(2.0 * y - 1), y, subs), dev, 2)
    check(r["auc"] == 1.0 and r["acc"] == 1.0, f"perfect predictor: auc={r['auc']} acc={r['acc']}")
    r = evaluate(m, loader(two(1.0 - 2.0 * y), y, subs), dev, 2)
    check(r["auc"] == 0.0 and r["acc"] == 0.0, f"inverted predictor: auc={r['auc']} acc={r['acc']}")
    r = evaluate(m, loader(two(np.zeros(n)), y, subs), dev, 2)
    check(r["auc"] == 0.5, f"constant predictor: auc={r['auc']}")
    s = rng.normal(size=n) + 0.8 * y
    r = evaluate(m, loader(two(s), y, subs), dev, 2)
    ref = roc_auc_score(y, s)
    check(abs(r["auc"] - ref) < 1e-6, f"random scores: auc={r['auc']:.6f} sklearn={ref:.6f}")
    check(abs(r["acc"] - ((s > 0) == y).mean()) < 1e-9, f"accuracy={r['acc']:.4f} matches threshold at 0")
    check(r["n"] == n, f"all {n} examples counted (none dropped by batching)")
    for sub in ["sA", "sB", "sC"]:
        k = subs == sub
        check(abs(r["per_subject_auc"][sub] - roc_auc_score(y[k], s[k])) < 1e-6,
              f"per-subject AUC {sub}={r['per_subject_auc'][sub]:.4f} matches sklearn")
    # one subject with a single class -> nan, not a crash
    y1 = y.copy(); y1[subs == "sA"] = 1
    r = evaluate(m, loader(two(s), y1, subs), dev, 2)
    check(math.isnan(r["per_subject_auc"]["sA"]), "single-class subject gives NaN AUC (not a crash)")
    # 3 classes: macro one-vs-rest on softmax probabilities
    y3 = rng.integers(0, 3, n)
    l3 = rng.normal(size=(n, 3)); l3[np.arange(n), y3] += 1.0
    r = evaluate(m, loader(l3, y3, subs), dev, 3)
    p3 = torch.softmax(torch.tensor(l3, dtype=torch.float32), -1).numpy()
    ref3 = roc_auc_score(y3, p3, multi_class="ovr", average="macro", labels=[0, 1, 2])
    check(abs(r["auc"] - ref3) < 1e-6, f"3-class macro AUC={r['auc']:.6f} sklearn={ref3:.6f}")


def smoke_cfg(real: bool, results_dir: str) -> dict:
    from locemb.config import load_config
    if real:
        cfg = load_config("configs/neuroprobe.yaml", {
            "spatial.type": "region", "spatial.atlas": "destrieux",
            "data.split": "cross_session", "data.electrodes": "lite"})
    else:
        cfg = load_config("configs/smoke.yaml", {
            "spatial.type": "region", "spatial.atlas": "grid40",
            "synthetic.windows_per_subject": 400})   # enough windows for 20 batches/epoch
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    return {**cfg, "results_dir": results_dir, "exp_name": "smoke_gpu_real" if real else "smoke_gpu",
            "run_name": "run", "force": True, "keep_ckpt": True, "device": "cuda",
            "train": {**cfg["train"], "epochs": 2, "max_batches": 20, "patience": 100,
                      "num_workers": max(0, ncpu - 1)}}


@stage("5-train")
def stage_train(cfg):
    from locemb.train import run
    out = os.path.join(cfg["results_dir"], cfg["exp_name"], cfg["run_name"])
    shutil.rmtree(out, ignore_errors=True)       # fresh run every time
    m = run(cfg)
    check(m["device"] == "cuda" and m["gpu"], f"trained on {m['device']} ({m['gpu']})")
    check(m["epochs_run"] == 2, f"epochs_run={m['epochs_run']}")
    for h in m["history"]:
        check(h["batches"] == 20, f"epoch {h['epoch']}: {h['batches']} mini-batches")
        check(math.isfinite(h["train_loss"]), f"epoch {h['epoch']}: train loss {h['train_loss']:.4f} is finite")
        check(math.isfinite(h["val_auc"]), f"epoch {h['epoch']}: val AUC {h['val_auc']:.4f} is finite")
    check(math.isfinite(m["test_auc"]) and 0 <= m["test_auc"] <= 1, f"test AUC {m['test_auc']:.4f} in [0,1]")
    check(m["test_n"] > 0, f"test set has {m['test_n']} windows")
    saved = json.load(open(os.path.join(out, "metrics.json")))
    check(saved["test_auc"] == m["test_auc"], "metrics.json written and matches")
    check(os.path.exists(os.path.join(out, "config.yaml")), "config.yaml written")
    return m


@stage("6-weights")
def stage_weights(cfg, metrics):
    from torch.utils.data import DataLoader
    from locemb.data import Collator, build_data
    from locemb.models import build_model
    from locemb.spatial import build_assignment
    from locemb.train import evaluate
    from locemb.utils import set_seed

    if metrics is None:
        raise RuntimeError("stage 5 failed, nothing to check")
    path = os.path.join(cfg["results_dir"], cfg["exp_name"], cfg["run_name"], "ckpt.pt")
    check(os.path.exists(path), f"checkpoint saved ({os.path.getsize(path) / 1e6:.2f} MB)")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    check(ck["epoch"] == 1 and len(ck["history"]) == 2, f"checkpoint is from the last epoch (epoch={ck['epoch']})")
    check(ck["best_state"] is not None, "best-epoch weights stored in checkpoint")
    check(all(torch.isfinite(v).all() for v in ck["model"].values() if v.is_floating_point()),
          "all saved weights finite")

    # Rebuild exactly as train.run does (same seed -> same init, same data, same test order).
    seed = int(cfg["seed"]); set_seed(seed)
    bundle = build_data(cfg, seed)
    assign = build_assignment(cfg["spatial"], bundle.subjects, bundle.train_subjects, seed)
    dev = torch.device("cuda")
    model = build_model(cfg, assign.n_regions, bundle.sfreq, bundle.n_classes).to(dev)
    init = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    changed = sum(not torch.equal(init[k], ck["model"][k]) for k in init if init[k].is_floating_point())
    check(changed > 0, f"training changed {changed}/{sum(v.is_floating_point() for v in init.values())} weight tensors")

    model.load_state_dict(ck["best_state"])
    reloaded = {k: v.cpu() for k, v in model.state_dict().items()}
    check(all(torch.equal(reloaded[k], ck["best_state"][k]) for k in reloaded), "weights load back bit-for-bit")
    nw = int(cfg["train"]["num_workers"])
    test_dl = DataLoader(bundle.test, batch_size=cfg["train"]["batch_size"], shuffle=False,
                         num_workers=nw, collate_fn=Collator(assign.table), persistent_workers=False)
    t = evaluate(model, test_dl, dev, bundle.n_classes)
    check(abs(t["auc"] - metrics["test_auc"]) < 1e-6,
          f"reloaded weights reproduce test AUC ({t['auc']:.6f} vs saved {metrics['test_auc']:.6f})")

    # Resume: a fresh optimizer must accept the saved optimizer state and take a step.
    model.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    opt.load_state_dict(ck["opt"])
    batch = next(iter(test_dl))
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(model(b).float(), b["y"])
    opt.zero_grad(); loss.backward(); opt.step()
    check(math.isfinite(loss.item()), f"resumed model + optimizer take a step (loss {loss.item():.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use real Brain Treebank instead of synthetic data")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    os.chdir(os.path.join(HERE, ".."))
    if not os.environ.get("SLURM_JOB_ID"):
        print("WARN: not inside a Slurm job. Don't run this on a login node.")
    print(f"[smoke] mode={'real' if a.real else 'synthetic'} host={os.uname().nodename} "
          f"job={os.environ.get('SLURM_JOB_ID')}", flush=True)

    stage_env()
    cuda_ok = all(ok for _, ok, _ in RESULTS)
    if cuda_ok:
        stage_gpu()
    stage_atlases(["lobes", "ho", "destrieux", "schaefer400"])
    if cuda_ok:
        stage_eval()
        cfg = smoke_cfg(a.real, a.results_dir)
        m = stage_train(cfg)
        stage_weights(cfg, m)
    else:
        print("\nCUDA unavailable: skipping GPU stages 2, 4, 5, 6")

    print("\n===== SUMMARY =====")
    for name, ok, info in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:10s} {info}")
    expected = 6 if cuda_ok else 1
    failed = [n for n, ok, _ in RESULTS if not ok]
    if not failed and len(RESULTS) == expected:
        print("ALL STAGES PASSED")
        return 0
    print("FAILED STAGES: " + (", ".join(failed) or "GPU stages skipped (no CUDA)"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
