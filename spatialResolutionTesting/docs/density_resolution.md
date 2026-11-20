# Density x resolution

**Question.** Does the best spatial resolution for electrode-location embeddings
depend on how much training data there is? Prediction: with more training people,
finer resolution pays off, and the best cube size shrinks.

This is the paper's main relationship. Earlier work compares a few fixed scales
(BaRISTA: channel / parcel / lobe, parcel best) but never crosses scale with data
amount. A literature check (Nov 2026; `../reports/Spatial encoding resolution prior work.md`,
notes in `../research_notes/Density resolution deep dive/`) found no iEEG, EEG or
spiking-model paper that does; residual risk is a follow-up from BaRISTA's lab.

## Why cross-subject, why cubes

- **Cross-subject only.** In cross-session the same electrodes are in training and
  test, so a finer code just acts like an electrode ID; no tradeoff is expected.
  With a held-out person, a fine cell only helps if training people had electrodes there.
- **Cubes, not atlases.** Cube side h is a pure resolution knob with no anatomy, and it
  maps onto the theory directly. Atlases are the follow-up.

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
| held-out person | all 6 (leave-one-subject-out) |
| training people k | 1, 2, 5 (nested subsets; who is drawn changes with seed) |
| encoding | none; cubes 80, 57, 40, 28, 20, 14, 10, 5 mm |
| seed | 0, 1 (stage 1; more by the staged rule) |

324 runs, recipe r2_stft_car (Phase 0 pick), 40 epochs, last epoch.
Knobs: `data.n_train_subjects`, `data.train_subset_seed` (cross-subject),
`data.electrode_frac` (cross-session, used by Experiment 2).

## Analysis (`scripts/analyze_density.py`, fixed before running)

gain(h) = AUC(cube h) - AUC(none), paired on person, density and seed.

- **H1 (primary):** slope of gain per halving of cube size; interaction
  slope(k=5) - slope(k=1). Permutation test (density labels within person x seed);
  population-level sign-flip over 6 people as secondary.
- **H2:** pooled vertex regression gain ~ x + x^2 + s + xs + x^2 s (x = log2(80/h),
  s = log2 k): how many octaves finer the optimum moves per doubling of people,
  bootstrap CI over people. Theory: 0.2-0.33.
- **H3 (mechanism):** leave-one-person-out R^2 of gain ~ x * coverage, with and without
  density. If density adds nothing, it acts through coverage.

Figure: gain vs cube size per k, and gain vs coverage.

**Power** (simulated at Phase 0's cross-subject run SD 0.059): with 2 seeds the
primary test detects the interaction only when the k=5 peak gain is >= ~0.06 AUC.
Seeds are therefore staged: after stage 1, look only at the seed SD
(`--noise-only`) and add seeds 2, 3 if needed.

## Run

```bash
LOCEMB_PER_TASK=3 bash slurm/submit_sweep.sh configs/sweeps/density_resolution.yaml 8 --time=06:00:00
python scripts/analyze_density.py results/density_resolution --noise-only   # staged-seed rule
python scripts/analyze_density.py results/density_resolution               # full analysis
```

Smoke test (real data, 2 epochs, 4 runs): `configs/sweeps/smoke_density.yaml`.
