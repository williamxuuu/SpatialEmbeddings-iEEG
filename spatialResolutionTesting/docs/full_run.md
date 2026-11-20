# Full run: protocol, design, and statistics

Written 2026-11-19, after the Nov 18-19 pilots (174 runs: `pilot_lite`, `pilot_lite_xsubj`,
`pilot_soft`, `pilot_soft_xsubj`). The rules below are fixed **before** any full-run result
exists. Change them only with a dated note saying why.

## 1. What the pilots showed

| Finding | Evidence | Consequence |
|---|---|---|
| Most runs never learned | ~80% ended at test AUC 0.51-0.55 with accuracy exactly 0.500 (one class for everything); train loss sat at ln 2 | The pilot compares seed luck, not encodings |
| Early stopping killed runs inside the plateau | `patience: 8`, no minimum: 107/174 runs stopped by epoch 14. The best run (`ho`, seed 0, 0.85) left the plateau at epoch 17-18 and survived only because val AUC beat its best by 0.0001 at epoch 14 | No early stopping during the plateau |
| The signal was there; the model didn't find it | Linear model on log band power, pilot preprocessing, subject 3 cross-session: test AUC **0.929** with per-window z-scoring, **0.935** with recording-level z-scoring (scratch diagnostic, 2026-11-19). The pilot transformer's best run got 0.94 on subject 3 but most runs stayed at chance | Normalization is NOT the main problem (the leaderboard's per-window vs global gap, e.g. PopT 0.913 vs 0.719, is for STFT-normalized pretrained models, a different setting). The bottleneck is the raw-voltage CNN having to learn spectral features from scratch |
| Raw-voltage input is weak | Linear model: raw voltage 0.616 vs spectrogram 0.860 (leaderboard, cross-session speech); GRU from scratch on raw voltage sits at chance on the leaderboard | Spectral front end (`frontend: stft`) is the main Phase 0 candidate; `zscore: session` is tested there too (r0 vs r1) |
| Cross-subject design could not reach significance | Only subjects 2 and 7 tested, and 7 is near chance for every leaderboard model, so n=2 (smallest sign-flip p = 0.5) | Leave-one-subject-out: n=6 (p >= 0.031) |
| Hints worth testing properly | No-region encodings (`none`, `coords`) left the plateau in 0/44 runs vs 18/130 for region encodings; hard Destrieux 0/27 vs Gaussian Destrieux 6/25 (Fisher p = 0.009); random-neighbour control ≈ real neighbours | Report "leaves the plateau" as its own outcome; keep the controls |

## 2. What changed in the code (all backward compatible; old configs behave as before)

- `preprocess.zscore: window | session | none` (`true` = window). `session` uses each electrode's
  mean/SD over its whole recording (estimated from 256 fixed random windows, no labels).
- `preprocess.reref: laplacian`: each contact minus the mean of its shaft neighbours (contact
  number ±1), read from all usable contacts even when the model sees only Lite's. A contact with
  one usable neighbour gets a bipolar difference; with none it is left as is (count logged).
- `model.frontend: cnn | stft`: `stft` = log power spectrum of each 250 ms patch (Hann window,
  4 Hz bins up to 200 Hz, no per-token normalization) instead of the raw-voltage CNN.
  `model.readout: cls | mean`.
- `train.min_epochs`, `train.min_delta`, `train.select: best | last`. Both the best-validation and
  the last epoch are tested and saved every run (`test_best`, `test_last`).
- `spatial.type: electrode_id`: one learned vector per electrode (identity-tag control).
- `spatial.jitter_mm`: moves every electrode by N(0, σ) mm per axis before encoding (fixed per seed).
- Every run saves `preds.npz` (labels, scores, subject) and a **location probe**: test AUC again
  with each person's location codes shuffled among their electrodes; the drop measures how much
  the trained model relies on location.
- **I/O fix:** each cached recording is read once front to back when a run opens it
  (`SessionData.prefetch`, ~3 s each, `data.prefetch: false` to turn off). Cold random reads on
  PACE scratch cost ~0.5 s per window; they made the pilot's first epoch take ~400 s and made the
  first v2 smoke job time out (1 h) while building datasets. After the fix a full 6-subject build
  takes ~70 s and a warm window ~1.6 ms.
- `slurm/array.sbatch`: `LOCEMB_PER_TASK=K` runs K sweep lines one after another per array task.
  A failed run no longer stops the others in its task.
- `scripts/runs_table.py`, `scripts/phase0_report.py`, `scripts/analyze_full.py`.
- `tests/test_fixes.py`: 11 tests for the above.

## 3. Protocol (`configs/neuroprobe_v2.yaml`)

- Data: Neuroprobe Lite sessions and windows (benchmark-exact), **all usable electrodes**
  (Lite's 120 were chosen by linear decodability, which can hide location effects; Tier 4b
  checks Lite). MNI152 coordinates (`coordinates_type="mni"`, true 3D positions for depth contacts).
- Normalization: `zscore: session` by default (it keeps between-window power changes and costs
  nothing), but Phase 0's r0 vs r1 arms measure whether it matters for the transformer.
- Training: 40 epochs, cosine LR with 5% warmup, **no early stopping**, report the **last** epoch.
  Validation (first half of each test recording, as in Neuroprobe) only picks the Phase 0 recipe.
  No test-person labels touch model selection in the full run.
- Primary metric: per-subject test AUC (the mean over subjects, as on the Neuroprobe leaderboard),
  not the AUC pooled across subjects.

## 4. Phases

| Phase | Sweep(s) | Runs | Purpose | Gate to continue |
|---|---|---|---|---|
| 0 | `phase0_recipe.yaml` + `phase0_xsubj.yaml` | 45 + 24 | Pick the recipe (front end, reference, readout) on validation data; check it also learns on new people | `phase0_report.py`: leaves the plateau in ≥8/9 cross-session and ≥6/8 cross-subject runs; copy it into `neuroprobe_v2.yaml` |
| 1 | `full/full_xsess.yaml`, `full/full_xsubj.yaml` | 120 + 432 | The paper: 10-12 encodings, speech + onset, both cross-session directions, LOSO, 3 seeds | none (analyse) |
| 2 | `full/tier2_*.yaml` | 120 + 360 | Finer resolution curve (+ `lobes`); soft width σ = 1.5 / 6 mm (with Tier 1's 3 mm); `ho_prob` soft vs argmax | after Phase 1; width rows only if Tier 1's gauss3 beats hard |
| 3 | `full/tier3_*.yaml` | 216 + 648 | Task dependence (delta_volume, volume, word_index, global_flow) | optional |
| 4 | `full/tier4_*.yaml` | 216 + 90 + 90 + 216 | Robustness: localization jitter, Lite electrodes, CNN front end, linear model | after Phase 1 |

**Cost cut (2026-11-20, before any Phase 0 result):** cross-session seeds 5 → 3 (n is still 6
people; seeds only make each person's mean less noisy), and `lobes` + the `ho_prob_soft` /
`ho_prob_argmax` pair moved from Tier 1 to Tier 2. Chosen by scientific priority, not by any
result. Phase 1 went from 800 to 552 runs. Leave-one-subject-out is kept (pairs would train on 4
people and inflate unseen regions for fine atlases, confounding the resolution question).

## 5. Statistics (pre-registered; `scripts/analyze_full.py`)

- **Unit = test subject.** For each (split, task, encoding, subject): mean AUC over seeds and
  folds/directions.
- **Contrasts** (`CONTRASTS` in the script): destrieux vs none / shuffled / electrode_id / coords;
  coords vs none; gauss3 vs hard; gauss3 vs random neighbours; gauss3_nearest vs hard_nearest;
  prob-soft vs prob-argmax; schaefer400 vs destrieux; lobes vs none. Plus jitter interactions.
- **Test:** exact two-sided paired sign-flip over the 6 subjects (a subject's tasks flip
  together); BH-FDR within each split. With 6 subjects the smallest p is 0.031, so a contrast can
  survive BH only if nearly all subjects agree. State this limit in the paper.
- **Also reported:** bootstrap 95% CIs (subjects, then seeds); share of runs that leave the plateau
  and the epoch they do; location-probe drop; fraction of held-out weight on `<unknown>`;
  Spearman ρ of AUC vs log(#regions) (descriptive).
- **Leaderboard format** (`leaderboard_format.csv`): mean test AUROC over test sessions per
  (task, encoding), as Neuroprobe reports it, for seed 0 alone (one run, like a leaderboard entry)
  and mean / range over seeds. Neuroprobe's own protocol is the floor we hold ourselves to (fixed
  benchmark splits and windows, no test data in model selection); ours adds seeds, a fixed
  last-epoch rule, and subject-level tests. Comparable to neuroprobe.dev: cross-session only, and
  even there our model is one model for all 6 people, not one per subject. Our LOSO cross-subject
  is not the official split (train on subject 2 trial 4 only).
- A "no difference" claim needs an equivalence argument (e.g. 95% CI inside ±0.01 AUC), not p > 0.05.
- Analyse Tier 1 + Tier 2 + Tier 4a together; Tier 3, 4b, 4c, 4d each separately (the script
  refuses to average the same encoding name across different protocols).

## 6. Cost

Pilot: 174 runs, 18.9 A100-hours (6.5 min per run; most stopped early, Lite electrodes). Full
runs train all 40 epochs on up to 246 electrodes, so expect several times that per run.
**Measure in Phase 0**, then project:

```bash
python scripts/estimate_compute.py configs/sweeps/full/full_xsess.yaml configs/sweeps/full/full_xsubj.yaml \
  --pilot results/phase0_recipe --parallel 8
```

## 7. Commands

```bash
cd spatialResolutionTesting && source slurm/activate_env.sh
python -m pytest -q                                              # must pass
# Phase 0 (3 runs per array task; one run is ~10-30 min)
LOCEMB_PER_TASK=3 bash slurm/submit_sweep.sh configs/sweeps/phase0_recipe.yaml 8 --time=03:00:00
python scripts/phase0_report.py results/phase0_recipe            # -> edit neuroprobe_v2.yaml
# Phase 1
LOCEMB_PER_TASK=3 bash slurm/submit_sweep.sh configs/sweeps/full/full_xsess.yaml 16 --time=03:00:00
LOCEMB_PER_TASK=3 bash slurm/submit_sweep.sh configs/sweeps/full/full_xsubj.yaml 16 --time=03:00:00
python scripts/check_run.py results/full_xsess/* results/full_xsubj/* --expect-cuda --epochs 40
python scripts/analyze_full.py results/full_xsess results/full_xsubj --out reports/full_tier1
```

## 8. Known limits

- 6 subjects (children and adolescents), linear registration to an adult MNI template: coordinate
  errors of a few mm are likely, the same size as the soft kernels (Tier 4a measures sensitivity).
- One model family (a small population transformer trained from scratch). The question "does the
  best encoding depend on the architecture?" needs a pretrained model (BaRISTA / PopT) behind
  `build_model`. That is the natural next step after Tier 1.
- Cross-session codes can act as electrode name tags; `electrode_id` measures how far that goes.
- Session z-scoring uses unlabeled statistics of the test recording (as Neuroprobe's "global
  z-scoring" baselines do).
