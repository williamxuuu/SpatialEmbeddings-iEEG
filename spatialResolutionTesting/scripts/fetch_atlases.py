"""Download/cache every atlas the configs use. Run once on a login node:
    LOCEMB_ATLAS_DIR=$PWD/nilearn_data python scripts/fetch_atlases.py
"""
import warnings

from locemb.spatial.atlases import get_labeler, get_prob_labeler

warnings.filterwarnings("ignore")
for name in ["lobes", "ho", "destrieux", "schaefer100", "schaefer200", "schaefer400", "schaefer1000"]:
    lb = get_labeler(name, True)
    print(f"{name:14s} {len(lb.names):5d} regions (incl. subcortical)")
pl = get_prob_labeler("ho_prob", True)
print(f"{'ho_prob':14s} {len(pl.names):5d} regions (probabilistic)")
