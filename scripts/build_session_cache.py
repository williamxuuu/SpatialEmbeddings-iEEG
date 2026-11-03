"""Filter whole Brain Treebank recordings once and save them (see
src/locemb/data/session_cache.py). Run this before a sweep so the GPU jobs
don't each wait on filtering. CPU only.

    python scripts/build_session_cache.py --config configs/neuroprobe.yaml            # all 12 Lite sessions
    python scripts/build_session_cache.py --config configs/neuroprobe.yaml --index 3  # just one (Slurm array task)

Uses the config's `preprocess` filter settings (notch / bandpass / resample),
`data.root` and `data.cache_dir`. Every usable electrode is saved, so the
same cache serves electrodes=lite and electrodes=all.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from locemb.config import load_config  # noqa: E402
from locemb.data.preprocess import filter_spec  # noqa: E402
from locemb.data.session_cache import build_session, default_cache_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/neuroprobe.yaml")
ap.add_argument("--index", type=int, help="build only this one of the Lite sessions (0-11)")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, dict(o.split("=", 1) for o in a.overrides))
root = cfg["data"].get("root") or os.environ.get("ROOT_DIR_BRAINTREEBANK")
if not root:
    sys.exit("set data.root or ROOT_DIR_BRAINTREEBANK")
os.environ["ROOT_DIR_BRAINTREEBANK"] = root
import neuroprobe as npb  # noqa: E402
from neuroprobe import config as npc  # noqa: E402

spec = filter_spec(cfg.get("preprocess"))
cache_dir = cfg["data"].get("cache_dir") or default_cache_dir(root)
todo = list(npc.NEUROPROBE_LITE_SUBJECT_TRIALS)
if a.index is not None:
    todo = [todo[a.index]]
print(f"filter settings {spec} -> {cache_dir}")
for s, t in todo:
    subj = npb.BrainTreebankSubject(s, cache=False, dtype=torch.float32, coordinates_type="mni")
    print(build_session(subj, t, spec, cache_dir, root), flush=True)
