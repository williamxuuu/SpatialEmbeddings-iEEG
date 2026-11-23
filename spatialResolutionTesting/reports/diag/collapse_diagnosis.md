# Why cross-subject runs collapse and are so noisy (2026-11-23)

Evidence:
- phase0_soft_xsubj: 212 runs (held-out people 1, 2, 3). The pending tasks were put on hold on 2026-11-23 and none were cancelled.
- phase0_xsubj from Exp 1, which includes a `none` arm.
- phase0_soft_xsess: 84 runs.
- A model-free band-power diagnostic: `scripts/diag_bandpower.py` and `scripts/diag_analyze.py`. Output is in `bandpower_report.txt`.

## Symptoms

| | cross-session | cross-subject |
|---|---|---|
| mean val AUC by epoch (ep 0 / 4 / 12 / 39) | 0.56 / 0.73 / 0.75 / 0.72 | 0.58 / 0.67 / 0.63 / 0.57 |
| runs below 0.5 at the last epoch | 0% | 25% |
| seed SD at the last epoch | ~0.03-0.04 | 0.07-0.13 (power analysis assumed 0.059) |
| train loss at epoch 39 | | 0.15-0.19 |

## Findings

1. **Seed noise is real model variance, not measurement noise.**
   - Across seeds, the last-epoch AUC on the first half of the held-out recording correlates with the AUC on the second half: median r = 0.95.
   - Bootstrap measurement SD is 0.014, and the SD over epochs 30-39 within a run is 0.007.
   - So each seed converges to a genuinely different solution.
   - Averaging seed predictions lifts AUC by 0.05-0.10 (e.g., 0.66 to 0.74, 0.72 to 0.80).

2. **Below-chance runs are systematic.**
   - On held-out person 1, Schaefer 1000 hard: single seeds average 0.39, and the 12-seed ensemble is 0.24. The seeds agree on the wrong ranking.
   - The same data with gauss3 codes gives 0.66 (ensemble 0.74).
   - Shuffling the held-out person's codes takes every setting to about 0.5. The model's output is entirely region-code-specific readout.

3. **Region codes act as person/electrode IDs, and the model memorises them.**
   - At Schaefer 400/1000 the median region has about 4 training electrodes, mostly from one person.
   - Cross-session (same person) tells the story:
     - with codes, AUC is 0.70-0.89 for all 6 people;
     - without codes, it is 0.50-0.55 for people 4, 7 and 10, even within the same person.
   - The speech information in those people is electrode-specific. It is not a broad high-gamma response:
     - only 1-3% of their electrodes have a high-gamma (HG) AUC above 0.6, against 18-33% for people 1, 2 and 3;
     - a pooled linear model within the person reaches 0.52-0.64, against 0.81-0.84 for people 1 and 3.
   - To fit people 4, 7 and 10 in training, the model has to learn per-code detectors, and training loss falls to 0.16.

4. **These detectors barely transfer across people.**
   - Electrode HG responses correlate across people at only r ≈ 0.26-0.35 at 3-12 mm. The correlation is 0.19 at 12-20 mm and about 0 beyond 20 mm.
   - Within a person, the same correlation is 0.83 at under 5 mm.
   - A region's mean training response predicts a new person's electrode response with r = 0.45 at Schaefer 400. The correlation is 0.30 at Schaefer 1000, 0.21-0.25 at lobes/Destrieux/grid80, and 0.39-0.40 at grid20/40.
   - Person 7 is negative at every resolution.
   - This is a model-free estimate of the resolution curve. The peak is near Schaefer 400 / 20-40 mm.

5. **The generic (pooled) contrast is weak and changes sign across people and movies.**
   - A pooled linear model trained on 5 people gets 0.71-0.77 on held-out people 2 and 3, 0.54-0.61 on person 1, and 0.46-0.54 on people 4, 7 and 10.
   - The pooled HG speech AUC is below 0.5 in several sessions (4B, 10A, 10B).
   - Even the no-code model decays, from 0.65 at epoch 0 to 0.54.

6. **Drift inside the held-out recordings.**
   - The pooled power in the first half (val) differs from the second half (test) by up to 0.6-1.2 SD in people 2, 3 and 10, and the drift differs by class.
   - It adds noise between halves but is not the main cause.

## Mechanism

- Early epochs learn generic features that partly transfer, so held-out AUC peaks at epochs 1-5.
- Fitting 5 training people then pushes the model onto per-code detectors, which barely transfer or transfer inversely to a new person.
- With 40 epochs and last-epoch selection, every run is measured fully overfit. That produces the decline, the below-chance runs (which are data-determined, hence systematic) and the seed variance (which detectors win depends on the initialisation).
- Fine atlases (Schaefer 1000 hard: 31-52% of test electrodes unknown, fewer electrodes per code) are hit hardest.

## Fix being piloted (`configs/sweeps/fix_pilot.yaml`, not yet run)

- `data.val_subjects=auto`: pick the epoch on a held-out *training-side* person. This is a cross-person signal and uses no test labels. It costs one training person (5 to 4).
- `spatial.code_dropout`: during training, each electrode's code becomes `<unknown>` with probability p. This forces generic features and trains `<unknown>`, which covers 2% of training electrodes but 19-52% of a held-out person's electrodes.
- Analyse seed ensembles, or more seeds, rather than single seeds.
- `density_resolution` (864 runs) used the collapsing recipe. It must not run until the pilot picks the recipe.
