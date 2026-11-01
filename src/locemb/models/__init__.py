from torch import nn

from ..spatial.encodings import build_spatial_encoder
from .linear import RegionPooledLinear
from .transformer import PopulationTransformer


def build_model(cfg: dict, n_regions: int, sfreq: float, n_classes: int) -> nn.Module:
    m = cfg["model"]
    sp = cfg["spatial"]
    if m["name"] == "transformer":
        d = m.get("d", 64)
        spatial = build_spatial_encoder(sp, n_regions, d)
        return PopulationTransformer(spatial, n_classes, d=d, n_patches=m.get("n_patches", 4),
                                     layers=m.get("layers", 4), heads=m.get("heads", 4),
                                     dropout=m.get("dropout", 0.1))
    if m["name"] == "linear":
        return RegionPooledLinear(n_regions, sfreq, n_classes, sp.get("type", "none"))
    raise ValueError(f"Unknown model {m['name']!r}. Add new architectures here.")


__all__ = ["build_model"]
