"""Run the Neuroprobe adapter through the real `neuroprobe` package, on a fake
data folder laid out like Brain Treebank's (tiny, synthetic signals).

The fake signal of raw electrode i is 100*(i+1) + (seconds since start), so a
window's values show which electrode and which time it came from. Neuroprobe's
word tables are cut to the first 20 minutes so the fake recordings stay small.
Skipped if neuroprobe or h5py isn't installed.
"""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/nonexistent")  # neuroprobe reads it at import
npb = pytest.importorskip("neuroprobe")
h5py = pytest.importorskip("h5py")
pd = pytest.importorskip("pandas")

SESSIONS = [(4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1)]
MINUTES = 20
SF = 2048


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    from neuroprobe import config as npc
    from neuroprobe import datasets as npd

    root = tmp_path_factory.mktemp("btb")
    words_dir = tmp_path_factory.mktemp("words")
    n = MINUTES * 60 * SF
    mni_dir = npc.MNI_COORDS_DIR
    raw_labels = {}
    for s in sorted({s for s, _ in SESSIONS}):
        mni = pd.read_csv(os.path.join(mni_dir, f"btbank{s}_mni_coords.csv"))
        known = [str(e).replace("*", "").replace("#", "") for e in mni.electrode_label]
        lite = [e for e in npc.NEUROPROBE_LITE_ELECTRODES[f"btbank{s}"] if e in known][:6]
        other = [e for e in known if e not in npc.NEUROPROBE_LITE_ELECTRODES[f"btbank{s}"]][:4]
        corrupted = other[3]
        labels = lite[:3] + ["DC1"] + other[:3] + lite[3:] + [corrupted]
        raw_labels[s] = labels
        (root / "electrode_labels" / f"sub_{s}").mkdir(parents=True)
        json.dump(labels, open(root / "electrode_labels" / f"sub_{s}" / "electrode_labels.json", "w"))
        (root / "localization" / f"sub_{s}").mkdir(parents=True)
        pd.DataFrame({"Electrode": [e for e in labels if e != "DC1"],
                      "DesikanKilliany": ["ctx-lh-superiortemporal"] * (len(labels) - 1),
                      "L": 1.0, "P": 2.0, "I": 3.0}).to_csv(
            root / "localization" / f"sub_{s}" / "depth-wm.csv", index=False)
    json.dump({f"sub_{s}": ([raw_labels[s][-1]] if s in raw_labels else []) for s in range(1, 11)},
              open(root / "corrupted_elec.json", "w"))

    t = np.arange(n)
    for s, tr in SESSIONS:
        with h5py.File(root / f"sub_{s}_trial{tr:03d}.h5", "w") as f:
            g = f.create_group("data")
            for i, _ in enumerate(raw_labels[s]):
                x = (100.0 * (i + 1) + t // SF).astype(np.float32)
                g.create_dataset(f"electrode_{i}", data=x, compression="gzip", shuffle=True)
        for kind in ("words", "nonverbal"):
            df = pd.read_csv(os.path.join(npc.SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{s}_trial{tr}_{kind}_df.csv"))
            df = df[df.est_idx < n - 2 * SF]
            df.to_csv(words_dir / f"subject{s}_trial{tr}_{kind}_df.csv", index=False)
            if kind == "words":
                movie = npc.BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{s}_{tr}"]
                (root / "transcripts" / movie).mkdir(parents=True, exist_ok=True)
                full = pd.read_csv(os.path.join(npc.SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{s}_trial{tr}_words_df.csv"))
                pd.DataFrame({"dummy": 0}, index=range(int(full.original_index.max()) + 1)).to_csv(
                    root / "transcripts" / movie / "features.csv")

    old_words = npd.SAVE_SUBJECT_TRIAL_DF_DIR
    old_roots = (npc.ROOT_DIR, npd.ROOT_DIR)
    import neuroprobe.braintreebank_subject as nps
    old_subj_root = nps.ROOT_DIR
    npd.SAVE_SUBJECT_TRIAL_DF_DIR = str(words_dir)
    npc.ROOT_DIR = npd.ROOT_DIR = nps.ROOT_DIR = str(root)
    yield root, raw_labels
    npd.SAVE_SUBJECT_TRIAL_DF_DIR = old_words
    npc.ROOT_DIR, npd.ROOT_DIR = old_roots
    nps.ROOT_DIR = old_subj_root


def _cfg(root, tmp_path, **data):
    from locemb.config import load_config
    ov = {"data.root": str(root), "data.cache_dir": str(tmp_path / "cache"),
          "data.subject_trials": [[4, 1], [7, 1], [10, 1]], "data.eval_name": "speech",
          "preprocess.reref": "none", "preprocess.notch_hz": None, "preprocess.bandpass": None,
          "preprocess.resample_hz": None, "preprocess.zscore": False, "preprocess.clip": None}
    ov.update({f"data.{k}": v for k, v in data.items()})
    return load_config("configs/neuroprobe.yaml", ov)


def test_windows_come_from_the_right_electrodes_and_times(fake_root, tmp_path):
    from locemb.data import build_data
    from neuroprobe import config as npc

    root, raw = fake_root
    for which in ("lite", "all"):
        b = build_data(_cfg(root, tmp_path, electrodes=which, split="cross_session"), seed=0)
        info = b.subjects["btb04"]
        lite = set(npc.NEUROPROBE_LITE_ELECTRODES["btbank4"])
        usable = [e for e in raw[4][:-1] if e != "DC1"]  # no trigger, no corrupted
        if which == "lite":
            assert set(info.electrode_names) == lite & set(usable) and len(info.electrode_names) == 6
        else:
            assert sorted(info.electrode_names) == sorted(usable)
        assert info.coords.shape == (len(info.electrode_names), 3) and np.isfinite(info.coords).all()
        assert info.dataset_labels["DesikanKilliany"][0] == "ctx-lh-superiortemporal"
        ds = b.test.datasets[0]
        for k in (0, len(ds) - 1):
            x = ds[k]["x"].numpy()
            sec = ds.starts[k] / SF
            for row, e in enumerate(info.electrode_names):
                i = raw[4].index(e)
                expect = 100 * (i + 1) + np.floor(sec)
                assert abs(x[row, 0] - expect) < 1.0, (which, e, x[row, 0], expect)
        assert x.shape[1] == SF  # 1 s windows at the raw rate (no resampling here)
    assert len(b.train) > 0 and len(b.val) > 0 and len(b.test) > 0


def test_same_windows_for_lite_and_all(fake_root, tmp_path):
    from locemb.data import build_data
    root, _ = fake_root
    a = build_data(_cfg(root, tmp_path, electrodes="lite"), seed=0)
    b = build_data(_cfg(root, tmp_path, electrodes="all"), seed=0)
    for da, db in zip(a.test.datasets, b.test.datasets):
        assert np.array_equal(da.starts, db.starts) and np.array_equal(da.labels, db.labels)


def test_cross_subject_holds_out_people_and_movies(fake_root, tmp_path):
    from locemb.data import build_data
    root, _ = fake_root
    b = build_data(_cfg(root, tmp_path, split="cross_subject", test_subjects=[10],
                        subject_trials=[[4, 1], [7, 1], [10, 0]]), seed=0)
    assert b.train_subjects == {"btb04", "btb07"}
    assert {d.subject for d in b.test.datasets} == {"btb10"}
    # 10_0 is cars-2, and so is 7_0, so 7_0 must not be used for training
    trained = {(d.subject, d.session.meta["trial_id"]) for d in b.train.datasets}
    assert ("btb07", 0) not in trained and ("btb07", 1) in trained


def test_filtered_end_to_end_run(fake_root, tmp_path):
    from locemb.config import load_config
    from locemb.train import run
    root, _ = fake_root
    cfg = load_config("configs/neuroprobe.yaml", {
        "data.root": str(root), "data.cache_dir": str(tmp_path / "cache"),
        "data.subject_trials": [[4, 1], [7, 1], [10, 1]], "data.electrodes": "all",
        "results_dir": str(tmp_path / "res"), "run_name": "t", "train.epochs": 1,
        "train.num_workers": 0, "spatial.type": "region", "spatial.atlas": "lobes"})
    m = run(cfg)
    assert 0 <= m["test_auc"] <= 1


@pytest.mark.skipif(not os.environ.get("LOCEMB_SLOW"), reason="~7 min; set LOCEMB_SLOW=1 to run")
def test_pilot_sweep_lines_run_and_report(fake_root, tmp_path):
    """A few real lines of configs/sweeps/pilot_lite*.yaml, on the fake data,
    then the pilot report on their results."""
    import subprocess
    import sys

    import yaml

    from locemb.config import load_config
    from locemb.sweep import expand
    from locemb.train import run
    root, _ = fake_root
    for sweep_file in ("pilot_lite", "pilot_lite_xsubj"):
        jobs = expand(yaml.safe_load(open(f"configs/sweeps/{sweep_file}.yaml")))
        assert len(jobs) == 42
        pick = [j for j in jobs if j["labels"]["seed"] in ("0", "1")
                and j["labels"]["spatial"] in ("none", "destrieux", "destrieux_shuffled", "coords")]
        for j in pick:
            ov = dict(j["overrides"])
            ov.update({"data.root": str(root), "data.cache_dir": str(tmp_path / "cache"),
                       "results_dir": str(tmp_path / "res"), "train.epochs": 1, "train.num_workers": 0})
            if sweep_file == "pilot_lite":
                ov["data.subject_trials"] = [[4, 1], [7, 1], [10, 1]]
            else:
                ov["data.subject_trials"] = [[4, 1], [7, 1], [10, 1]]
                ov["data.test_subjects"] = [10]
            m = run(load_config(j["config"], ov))
            assert 0 <= m["test_auc"] <= 1
        out = subprocess.run([sys.executable, "scripts/pilot_report.py", str(tmp_path / "res" / jobs[0]["overrides"]["exp_name"])],
                             capture_output=True, text=True, check=True).stdout
        assert "Gate 1" in out and "Does Lite hold up" in out
