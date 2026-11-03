"""First look at the real Brain Treebank data on the cluster. Cheap (no
filtering, reads no signal). Run on a login node or a small CPU job:

    source slurm/activate_env.sh
    python scripts/inspect_neuroprobe.py            # uses $ROOT_DIR_BRAINTREEBANK

Checks that the 12 Lite recordings and the label/localization files are
there, prints electrode counts (lite vs all) and the number of examples per
split, and builds the pilot data exactly as the sweep will (with
--build, which also filters one recording and times it).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--eval", default="speech")
ap.add_argument("--build", action="store_true", help="also filter subject 10's two Lite recordings and time it")
a = ap.parse_args()

root = os.environ.get("ROOT_DIR_BRAINTREEBANK")
if not root:
    sys.exit("set ROOT_DIR_BRAINTREEBANK (slurm/cluster.env)")
import neuroprobe as npb  # noqa: E402
from neuroprobe import config as npc  # noqa: E402

print("neuroprobe", npb.__version__, "| root", root)
ok = True
for s, t in npc.NEUROPROBE_LITE_SUBJECT_TRIALS:
    p = os.path.join(root, f"sub_{s}_trial{t:03d}.h5")
    gb = os.path.getsize(p) / 1e9 if os.path.exists(p) else None
    ok &= gb is not None
    print(f"  sub {s:2d} trial {t}: {'MISSING' if gb is None else f'{gb:.1f} GB'}")
for f in ["corrupted_elec.json", "localization", "electrode_labels", "transcripts"]:
    e = os.path.exists(os.path.join(root, f))
    ok &= e
    print(f"  {f}: {'ok' if e else 'MISSING'}")
if not ok:
    sys.exit("missing files: check the download (braintreebank_download_extract.py --lite)")

for s in sorted({s for s, _ in npc.NEUROPROBE_LITE_SUBJECT_TRIALS}):
    subj = npb.BrainTreebankSubject(s, cache=False, dtype=torch.float32, coordinates_type="mni")
    lite = [e for e in npc.NEUROPROBE_LITE_ELECTRODES[subj.subject_identifier] if e in subj.electrode_labels]
    xyz = subj.get_electrode_coordinates()
    trials = [t for ss, t in npc.NEUROPROBE_LITE_SUBJECT_TRIALS if ss == s]
    sizes = []
    for t in trials:
        ds = npb.BrainTreebankSubjectTrialBenchmarkDataset(subj, t, torch.float32, a.eval,
                                                           output_indices=True, lite=True)
        item = ds[0]
        sizes.append(f"trial {t}: {len(ds)} windows of {item['data'][1] - item['data'][0]} samples")
    print(f"sub {s:2d}: {len(subj.electrode_labels)} usable electrodes, {len(lite)} lite, "
          f"MNI NaN rows: {int(torch.isnan(xyz).any(1).sum())} | " + "; ".join(sizes))

if a.build:
    from locemb.config import load_config
    from locemb.data import build_data
    cfg = load_config("configs/neuroprobe.yaml", {"data.root": root, "data.eval_name": a.eval,
                                                   "data.subject_trials": [[10, 1]]})
    t0 = time.time()
    b = build_data(cfg, seed=0)
    print(f"built {b.description} in {time.time() - t0:.0f}s: train={len(b.train)} "
          f"val={len(b.val)} test={len(b.test)}; one window: {tuple(b.test[0]['x'].shape)}")
