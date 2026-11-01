"""Brain atlases as "labelers": functions from MNI points (mm) to region codes.

All labelers work on points in millimetres, so atlases on different voxel
grids can be mixed freely, and soft assignment can sample many points around
an electrode.

Available atlas names (each can be combined with Harvard-Oxford subcortical
structures via include_subcortical=True, since sEEG has many hippocampus /
amygdala contacts):

    lobes          ~6 cortical groups per hemisphere (derived from Harvard-Oxford)
    ho             Harvard-Oxford cortical, 48 regions per hemisphere
    destrieux      Destrieux, ~74 per hemisphere (the atlas BaRISTA used for parcels)
    schaefer{N}    Schaefer 2018, N in {100,200,...,1000} (functional parcels, 7 networks)
    schaefer{N}_17 same, 17-network naming
    grid{S}        cubes of side S mm (no anatomy at all; cleanest resolution knob)

Probabilistic atlas (for soft assignment):

    ho_prob        Harvard-Oxford probability maps: for each point, the % of
                   people in the atlas sample in whom this point was region X

Cluster note: compute nodes often have no internet. Run
`python scripts/fetch_atlases.py` once on a login node to cache everything.
"""

from __future__ import annotations

import functools
import os
import re

import numpy as np

UNKNOWN = "<unknown>"


# ----------------------------------------------------------------------------
# Lobe grouping for Harvard-Oxford cortical labels. Rough, rule-based; check it
# against your lab's preferred definition (BaRISTA used Desikan-Killiany lobes).
# ----------------------------------------------------------------------------
_LOBE_RULES = [  # first match wins; order matters
    ("cingulate", ["Cingulate", "Paracingulate"]),
    ("insula", ["Insular"]),
    ("temporal", ["Temporal", "Heschl", "Planum", "Parahippocampal"]),
    ("occipital", ["Occipital", "Calcarine", "calcarine", "Cuneal", "Lingual"]),
    ("parietal", ["Parietal", "Postcentral", "Supramarginal", "Angular", "Precuneous"]),
    ("frontal", ["Frontal", "Precentral", "Juxtapositional", "Subcallosal", "Central Opercular"]),
]
_SUBCORT_GROUPS = {
    "Hippocampus": "hippocampus", "Amygdala": "amygdala", "Thalamus": "thalamus",
    "Caudate": "basal_ganglia", "Putamen": "basal_ganglia", "Pallidum": "basal_ganglia",
    "Accumbens": "basal_ganglia",
}
_SUBCORT_EXCLUDE = ["White Matter", "Cerebral Cortex", "Ventricle", "Brain-Stem"]


def lobe_of(name: str) -> str | None:
    for lobe, keys in _LOBE_RULES:
        if any(k in name for k in keys):
            return lobe
    return None


def _strip_side(name: str) -> str:
    return re.sub(r"^(Left|Right)\s+", "", name)


# ----------------------------------------------------------------------------
# Labelers
# ----------------------------------------------------------------------------
class VolumeLabeler:
    """Integer label volume -> region codes. Code -1 means background / no label."""

    def __init__(self, data, affine, raw_names: dict[int, str], split_hemi: bool,
                 rename=None, exclude=()):
        self.data = np.asarray(data)
        self.inv = np.linalg.inv(affine)
        self.split_hemi = split_hemi
        final = {}
        for raw, nm in raw_names.items():
            if raw == 0 or nm.lower() in ("background", "unknown", "medial_wall") \
                    or "Medial_wall" in nm or any(e in nm for e in exclude):
                continue
            nm2 = rename(nm) if rename else nm
            if nm2:
                final[raw] = nm2
        base_names = sorted(set(final.values()))
        base_idx = {n: i for i, n in enumerate(base_names)}
        max_raw = int(max(max(raw_names), self.data.max())) + 1
        self.lut = np.full(max_raw, -1, dtype=np.int64)
        for raw, nm in final.items():
            self.lut[raw] = base_idx[nm]
        self.names = ([f"{n}_L" for n in base_names] + [f"{n}_R" for n in base_names]
                      if split_hemi else base_names)
        self.n_base = len(base_names)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        ijk = np.rint(pts @ self.inv[:3, :3].T + self.inv[:3, 3]).astype(np.int64)
        shape = np.array(self.data.shape[:3])
        ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
        out = np.full(len(pts), -1, dtype=np.int64)
        v = self.data[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]].astype(np.int64)
        v = np.where(v < len(self.lut), v, 0)
        out[ok] = self.lut[v]
        if self.split_hemi:
            right = pts[:, 0] >= 0  # MNI: x >= 0 is the right hemisphere
            out = np.where(out >= 0, out + right * self.n_base, -1)
        return out

    def name_of(self, code: int) -> str:
        return self.names[code]


class GridLabeler:
    """Cubes of side `size_mm`. No anatomy: only 'how fine is the grid'."""

    def __init__(self, size_mm: float, extent_mm: float = 160.0):
        self.size = float(size_mm)
        self.off = int(np.ceil(extent_mm / self.size)) + 1
        self.n = 2 * self.off + 1

    def cube_ijk(self, pts):
        return np.floor(pts / self.size).astype(np.int64)

    def encode(self, ijk):
        ijk = np.clip(ijk + self.off, 0, self.n - 1)
        return (ijk[..., 0] * self.n + ijk[..., 1]) * self.n + ijk[..., 2]

    def __call__(self, pts):
        return self.encode(self.cube_ijk(pts))

    def name_of(self, code: int) -> str:
        i, r = divmod(int(code), self.n * self.n)
        j, k = divmod(r, self.n)
        return f"cube{self.size:g}_{i - self.off}_{j - self.off}_{k - self.off}"


class CombinedLabeler:
    """Use the first labeler that gives a label (e.g. cortex first, then subcortex)."""

    def __init__(self, labelers):
        self.labelers = labelers
        self.offsets = np.cumsum([0] + [len(lb.names) for lb in labelers])[:-1]
        self.names = [n for lb in labelers for n in lb.names]

    def __call__(self, pts):
        out = np.full(len(pts), -1, dtype=np.int64)
        for lb, off in zip(self.labelers, self.offsets):
            todo = out < 0
            if not todo.any():
                break
            c = lb(pts[todo])
            out[todo] = np.where(c >= 0, c + off, -1)
        return out

    def name_of(self, code: int) -> str:
        return self.names[code]


class ProbLabeler:
    """4D probability maps -> (N, K) region probabilities (rows may sum to < 1)."""

    def __init__(self, data4d, affine, names: list[str], split_hemi: bool, keep=None):
        data4d = np.asarray(data4d, dtype=np.float32)
        if data4d.max() > 1.5:
            data4d = data4d / 100.0  # Harvard-Oxford stores percent
        keep = keep if keep is not None else list(range(len(names)))
        self.data = data4d[..., keep]
        base = [names[i] for i in keep]
        self.inv = np.linalg.inv(affine)
        self.split_hemi = split_hemi
        self.n_base = len(base)
        self.names = [f"{n}_L" for n in base] + [f"{n}_R" for n in base] if split_hemi else base

    def __call__(self, pts):
        ijk = np.rint(pts @ self.inv[:3, :3].T + self.inv[:3, 3]).astype(np.int64)
        shape = np.array(self.data.shape[:3])
        ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
        p = np.zeros((len(pts), self.n_base), dtype=np.float32)
        p[ok] = self.data[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        if not self.split_hemi:
            return p
        right = (pts[:, 0] >= 0)[:, None]
        return np.concatenate([p * ~right, p * right], axis=1)


class StackedProbLabeler:
    def __init__(self, labelers):
        self.labelers = labelers
        self.names = [n for lb in labelers for n in lb.names]

    def __call__(self, pts):
        return np.concatenate([lb(pts) for lb in self.labelers], axis=1)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
def _load(fetch_result):
    import nibabel as nib
    maps = fetch_result.maps
    img = nib.load(maps) if isinstance(maps, (str, os.PathLike)) else maps
    return np.asanyarray(img.dataobj), img.affine


def _raw_names(fetch_result) -> dict[int, str]:
    lut = fetch_result.get("lut") if hasattr(fetch_result, "get") else None
    if lut is not None:
        return {int(i): str(n) for i, n in zip(lut["index"], lut["name"])}
    return {i: str(n) for i, n in enumerate(fetch_result.labels)}


def _data_dir():
    return os.environ.get("LOCEMB_ATLAS_DIR") or os.environ.get("NILEARN_DATA") or None


def _subcortical(group_basal_ganglia: bool):
    from nilearn import datasets
    f = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm", data_dir=_data_dir(), verbose=0)
    data, aff = _load(f)

    def rename(n):
        n = _strip_side(n)
        if group_basal_ganglia:
            return _SUBCORT_GROUPS.get(n)
        return n

    return VolumeLabeler(data, aff, _raw_names(f), split_hemi=True,
                         rename=rename, exclude=_SUBCORT_EXCLUDE)


@functools.lru_cache(maxsize=None)
def get_labeler(atlas: str, include_subcortical: bool = True):
    m = re.fullmatch(r"grid(\d+(?:\.\d+)?)", atlas)
    if m:
        return GridLabeler(float(m.group(1)))

    from nilearn import datasets
    dd = _data_dir()
    if atlas in ("ho", "lobes"):
        f = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm", data_dir=dd, verbose=0)
        data, aff = _load(f)
        cort = VolumeLabeler(data, aff, _raw_names(f), split_hemi=True,
                             rename=lobe_of if atlas == "lobes" else None)
    elif atlas == "destrieux":
        f = datasets.fetch_atlas_destrieux_2009(data_dir=dd, verbose=0)
        data, aff = _load(f)
        cort = VolumeLabeler(data, aff, _raw_names(f), split_hemi=False)
    elif (m := re.fullmatch(r"schaefer(\d+)(?:_(7|17))?", atlas)):
        f = datasets.fetch_atlas_schaefer_2018(n_rois=int(m.group(1)),
                                               yeo_networks=int(m.group(2) or 7),
                                               resolution_mm=1, data_dir=dd, verbose=0)
        data, aff = _load(f)
        cort = VolumeLabeler(data, aff, _raw_names(f), split_hemi=False)
    else:
        raise ValueError(f"Unknown atlas {atlas!r}")
    if include_subcortical:
        return CombinedLabeler([cort, _subcortical(group_basal_ganglia=(atlas == "lobes"))])
    return cort


@functools.lru_cache(maxsize=None)
def get_prob_labeler(atlas: str, include_subcortical: bool = True):
    from nilearn import datasets
    if atlas != "ho_prob":
        raise ValueError(f"Unknown probabilistic atlas {atlas!r} (only ho_prob for now)")
    dd = _data_dir()
    f = datasets.fetch_atlas_harvard_oxford("cort-prob-2mm", data_dir=dd, verbose=0)
    data, aff = _load(f)
    names = [str(n) for n in f.labels[1:]]  # labels[0] is Background; volume j <-> labels[j+1]
    parts = [ProbLabeler(data, aff, names, split_hemi=True)]
    if include_subcortical:
        f = datasets.fetch_atlas_harvard_oxford("sub-prob-2mm", data_dir=dd, verbose=0)
        data, aff = _load(f)
        raw = [str(n) for n in f.labels[1:]]
        keep = [i for i, n in enumerate(raw) if not any(e in n for e in _SUBCORT_EXCLUDE)]
        # Subcortical prob maps are already left/right specific; don't split again.
        sub = ProbLabeler(data, aff, raw, split_hemi=False, keep=keep)
        parts.append(sub)
    return StackedProbLabeler(parts)
