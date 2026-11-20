# Density x resolution

**Question.** Does the best spatial resolution for electrode-location embeddings
depend on how many electrodes each region gets? Prediction: with denser regions,
finer resolution pays off, and the best cube size shrinks.

This is the paper's main relationship. Earlier work compares a few fixed scales
(BaRISTA: channel / parcel / lobe, parcel best) but never crosses scale with data
density. A literature check (Nov 2026; `../reports/Spatial encoding resolution prior work.md`,
notes in `../research_notes/Density resolution deep dive/`) found no iEEG, EEG or
spiking-model paper that does; residual risk is a follow-up from BaRISTA's lab.

## Why cross-subject, why cubes, why thinning

- **Cross-subject only.** In cross-session the same electrodes are in training and
  test, so a finer code just acts like an electrode ID; no tradeoff is expected.
- **Cubes, not atlases.** Cube side h is a pure resolution knob with no anatomy.
- **Thin training electrodes, keep 5 people.** The first design varied the number of
  training people (1/2/5); that mostly changed coverage, not electrodes per region,
  because electrodes come in shafts. It was replaced (before running) by keeping all
  5 training people and randomly thinning their electrodes to 10%
  (`data.train_electrode_frac`); the held-out person keeps all electrodes.
  `reports/density_geometry.md` and the sweep header have the measured numbers.

## Theory (what to cite, what it predicts)

A cell embedding is a piecewise-constant estimate. Variance ~ 1/(n h^d), squared
bias ~ h^2, so h* ~ n^(-1/(d+2)) (partitioning estimates, Györfi et al. 2002 ch. 4,
theorem number to verify; same rate as Scott's multivariate histogram rule).

Two things specific to our data:
- **n is roughly the number of people, not electrodes.** Electrodes cluster on shafts
  and within people. Measured: median electrodes per occupied 20 mm cube is 3.5 with
  1 training person and 4.3 with 5.
- **Coverage.** A held-out electrode in a cube no training person occupied gets the
  shared unknown vector. Coverage is what density changes most (20 mm: 0.34 / 0.58 / 0.83
  for 1 / 2 / 5 people; 5 mm: 0.04 / 0.08 / 0.18), and it pushes the optimum coarser.

Variance-only theory predicts a small shift: 1 -> 5 people moves h* by 0.46-0.77
octaves (d = 3 to 1). Hence half-octave cube sizes.

## Design (`configs/sweeps/density_resolution.yaml`)

| axis | values |
|---|---|
| held-out person | all 6 (leave-one-subject-out), trained on the other 5 |
| density | e100 (all training electrodes), e10 (random 10%, redrawn per seed) |
| encoding | none; cubes 80, 40, 20, 10, 5 mm |
| seed | 0-23 |

1728 runs, recipe r2_stft_car (Phase 0 pick), 40 epochs, last epoch.

## Analysis (`scripts/analyze_density.py`, fixed before running)

gain(h) = AUC(cube h) - AUC(none), paired on person, density and seed.

- **H1 (primary):** slope of gain per halving of cube size; interaction
  slope(k=5) - slope(k=1). Permutation test (density labels within person x seed);
  population-level sign-flip over 6 people as secondary.
- **H2:** pooled vertex regression gain ~ x + x^2 + s + xs + x^2 s (x = log2(80/h),
  s = log2 density level): how many octaves finer the optimum moves per doubling of people,
  bootstrap CI over people. Theory: 0.2-0.33.
- **H3 (mechanism):** leave-one-person-out R^2 of gain ~ x * coverage, with and without
  density. If density adds nothing, it acts through coverage.

Figure: gain vs cube size per k, and gain vs coverage.

**Power** (simulated at Phase 0's cross-subject run SD 0.059, best size moving
0.25 octave per doubling of electrodes): e10 vs e100 with 24 seeds detects a 0.03 AUC
peak gain with power 0.80 (H1) / 0.95 (H2); e25 would give 0.38 / 0.65. Table in the
sweep header.

## Run

```bash
LOCEMB_PER_TASK=8 bash slurm/submit_sweep.sh configs/sweeps/density_resolution.yaml 16 --time=16:00:00   # 500-job QOS limit
python scripts/analyze_density.py results/density_resolution --noise-only   # seed SD only
python scripts/analyze_density.py results/density_resolution               # full analysis
```

Smoke test (real data, 2 epochs, 4 runs): `configs/sweeps/smoke_density.yaml`.
