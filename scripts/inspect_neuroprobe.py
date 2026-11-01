"""Print what the installed `neuroprobe` package actually returns, to resolve
the VERIFY notes in src/locemb/data/neuroprobe_adapter.py. Run on a node that
can see the data:
    python scripts/inspect_neuroprobe.py --subject 1 --trial 2 --eval speech
"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--subject", type=int, default=1)
ap.add_argument("--trial", type=int, default=2)
ap.add_argument("--eval", default="speech")
a = ap.parse_args()

import neuroprobe as npb

print("neuroprobe version:", getattr(npb, "__version__", "?"))
print("public names:", [n for n in dir(npb) if not n.startswith("_")])
for attr in ["NEUROPROBE_LITE_SUBJECT_TRIALS", "NEUROPROBE_TASKS"]:
    if hasattr(npb, attr):
        print(attr, "=", getattr(npb, attr))
subj = npb.BrainTreebankSubject(subject_id=a.subject, cache=True, dtype=torch.float32, coordinates_type="mni")
print("subject attrs:", [n for n in dir(subj) if not n.startswith("_")])
splits = npb.generate_splits_cross_session(test_subject=subj, test_trial_id=a.trial,
                                           eval_name=a.eval, output_indices=False)
print("type(splits):", type(splits), "len:", len(splits))
fold = splits[0]
print("fold keys:", list(fold.keys()) if isinstance(fold, dict) else type(fold))
for k in (fold.keys() if isinstance(fold, dict) else []):
    ds = fold[k]
    print(f"  {k}: {type(ds).__name__} len={len(ds)}")
    item = ds[0]
    if isinstance(item, dict):
        print("    item keys:", {kk: getattr(v, 'shape', v) for kk, v in item.items()})
    else:
        print("    item:", [getattr(v, "shape", v) for v in item])
    for at in ["electrode_labels", "electrode_coordinates"]:
        v = getattr(ds, at, None)
        print(f"    .{at}:", None if v is None else (len(v), getattr(v, 'shape', None)))
