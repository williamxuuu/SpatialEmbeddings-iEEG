import numpy as np
import pytest
import torch

from locemb.data.base import SubjectInfo
from locemb.spatial.assign import ball_offsets, build_assignment
from locemb.spatial.atlases import GridLabeler, lobe_of


def _subjects(n=3, E=25, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        c = rng.uniform([-60, -90, -30], [60, 50, 60], size=(E, 3))
        out[f"s{i}"] = SubjectInfo(f"s{i}", c, [f"e{j}" for j in range(E)])
    return out


def _check_rows(a):
    for t in a.table.values():
        s = t["w"].sum(1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
        assert (t["idx"] >= 0).all() and (t["idx"] < a.n_regions).all()


def test_grid_hard_is_one_hot():
    subs = _subjects()
    a = build_assignment({"type": "region", "atlas": "grid20", "mode": "hard"}, subs, set(subs))
    _check_rows(a)
    assert a.stats["frac_electrodes_multi_region"] == 0.0


def test_grid_codes_roundtrip():
    g = GridLabeler(10)
    pts = np.array([[-35.0, 12.0, 0.1], [99.0, -99.0, 5.0]])
    names = [g.name_of(c) for c in g(pts)]
    assert names == ["cube10_-4_1_0", "cube10_9_-10_0"]


def test_trilinear_is_soft_and_normalized():
    subs = _subjects()
    a = build_assignment({"type": "region", "atlas": "grid20", "mode": "trilinear"}, subs, set(subs))
    _check_rows(a)
    assert a.stats["frac_electrodes_multi_region"] > 0.5


def test_sphere_radius_zero_equals_hard():
    subs = _subjects()
    h = build_assignment({"type": "region", "atlas": "grid20", "mode": "hard"}, subs, set(subs))
    s = build_assignment({"type": "region", "atlas": "grid20", "mode": "sphere", "radius_mm": 0}, subs, set(subs))
    for sid in subs:
        assert torch.equal(h.table[sid]["idx"][:, 0], s.table[sid]["idx"][:, 0])


def test_unseen_regions_map_to_unknown_for_test_subjects():
    subs = _subjects()
    fit = {"s0", "s1"}
    a = build_assignment({"type": "region", "atlas": "grid5", "mode": "hard"}, subs, fit)
    # 5 mm cubes: most of a new person's cubes were never seen in training
    assert a.stats["frac_weight_unknown_test_subjects"] > 0.5
    assert a.stats["frac_weight_unknown_train"] == 0.0


def test_shuffle_keeps_region_multiset():
    subs = _subjects()
    spec = {"type": "region", "atlas": "grid20", "mode": "hard"}
    a = build_assignment(spec, subs, set(subs))
    b = build_assignment({**spec, "control": "shuffle_electrodes"}, subs, set(subs), seed=3)
    for sid in subs:
        assert sorted(a.table[sid]["idx"][:, 0].tolist()) == sorted(b.table[sid]["idx"][:, 0].tolist())
        assert not torch.equal(a.table[sid]["idx"], b.table[sid]["idx"])


def test_random_neighbors_keeps_top_region_and_weights():
    subs = _subjects()
    spec = {"type": "region", "atlas": "grid20", "mode": "trilinear"}
    a = build_assignment(spec, subs, set(subs))
    b = build_assignment({**spec, "control": "random_neighbors"}, subs, set(subs), seed=1)
    for sid in subs:
        assert torch.equal(a.table[sid]["idx"][:, 0], b.table[sid]["idx"][:, 0])
        assert torch.equal(a.table[sid]["w"], b.table[sid]["w"])


def test_ball_offsets_sorted():
    off, d = ball_offsets(3)
    assert np.all(np.diff(d) >= 0) and d[0] == 0


def test_lobe_rules():
    assert lobe_of("Superior Temporal Gyrus, anterior division") == "temporal"
    assert lobe_of("Cingulate Gyrus, anterior division") == "cingulate"
    assert lobe_of("Precentral Gyrus") == "frontal"
    assert lobe_of("Lateral Occipital Cortex, superior division") == "occipital"
    assert lobe_of("Angular Gyrus") == "parietal"


# ---- tests that need nilearn atlases (downloaded on first use) ----
nilearn = pytest.importorskip("nilearn")


@pytest.mark.parametrize("atlas", ["lobes", "destrieux"])
@pytest.mark.parametrize("mode", ["hard", "sphere", "gaussian"])
def test_atlas_modes(atlas, mode):
    subs = _subjects()
    try:
        a = build_assignment({"type": "region", "atlas": atlas, "mode": mode,
                              "radius_mm": 5, "sigma_mm": 3}, subs, set(subs))
    except OSError as e:  # offline
        pytest.skip(f"atlas download failed: {e}")
    _check_rows(a)
    assert a.stats["n_regions"] > 3


def test_prob_atlas_soft():
    subs = _subjects()
    try:
        a = build_assignment({"type": "region", "source": "prob_atlas", "atlas": "ho_prob",
                              "mode": "prob"}, subs, set(subs))
    except OSError as e:
        pytest.skip(f"atlas download failed: {e}")
    _check_rows(a)
    assert a.stats["frac_electrodes_multi_region"] > 0.2


def test_nearest_seen_replaces_unknown_for_test_subjects():
    subs = _subjects()
    fit = {"s0", "s1"}
    spec = {"type": "region", "atlas": "grid5", "mode": "hard"}
    a = build_assignment(spec, subs, fit)
    b = build_assignment({**spec, "unseen": "nearest_seen", "unseen_max_mm": 60}, subs, fit)
    assert a.stats["frac_weight_unknown_test_subjects"] > 0.5
    assert b.stats["frac_weight_unknown_test_subjects"] == 0.0
    assert b.stats["frac_test_electrodes_remapped"] > 0.5
    for sid in fit:                                   # training people are untouched
        assert torch.equal(a.table[sid]["idx"], b.table[sid]["idx"])
    # the replacement really is the nearest cube centre that training saw
    seen = {n: i for i, n in enumerate(b.names)}
    g = GridLabeler(5)
    centres = {i: (np.array([int(v) for v in n.split("_")[1:]]) + 0.5) * 5
               for n, i in seen.items() if n.startswith("cube")}
    for e in range(subs["s2"].n_electrodes):
        x = subs["s2"].coords[e]
        if g.name_of(int(g(x[None])[0])) in seen:
            continue
        got = centres[int(b.table["s2"]["idx"][e, 0])]
        best = min(np.linalg.norm(c - x) for c in centres.values())
        assert np.linalg.norm(got - x) <= best + 5 * np.sqrt(3) / 2 + 1e-6
