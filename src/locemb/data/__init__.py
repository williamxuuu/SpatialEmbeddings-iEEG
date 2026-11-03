from .base import Collator, DataBundle, SubjectInfo


def build_data(cfg: dict, seed: int) -> DataBundle:
    name = cfg["data"]["name"]
    if name == "synthetic":
        from .synthetic import make_synthetic
        return make_synthetic(cfg, seed)
    if name == "neuroprobe":
        from .neuroprobe_adapter import make_neuroprobe
        return make_neuroprobe(cfg, seed)
    raise ValueError(f"Unknown dataset {name!r}")


__all__ = ["build_data", "Collator", "DataBundle", "SubjectInfo"]
