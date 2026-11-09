# softRegionTesting: Experiment 2, overlapping regions

**Question.** Every model we've read gives each electrode exactly **one** brain region. An electrode near a border may pick up signal from both sides, and border positions differ from person to person. Does it help to give each electrode a **weighted mix** of the regions around it instead (for example 80% one region, 20% its neighbour)?

The model is the same either way: one learned vector per region, and an electrode gets the weighted sum of its regions' vectors. Only the weights change, so model size and speed don't. The idea, the ways to compute weights, and the other explanations to rule out are in [`../spatialResolutionTesting/docs/experiments.md`](../spatialResolutionTesting/docs/experiments.md) (Experiment 2).

This folder has the **pilot**, a smaller test with written-down pass/fail rules, plus the full sweep for later. The code, Python env, data, and filtered recordings are shared with [`../spatialResolutionTesting`](../spatialResolutionTesting); jobs run from there.

## What runs

Small transformer, the 12 Neuroprobe Lite recordings, `speech` task, Lite electrodes, **5 seeds**. Two sweeps, **90 runs** in total:

- `configs/sweeps/pilot_soft.yaml` (**40 runs**): tested on each person's other movie (`cross_session`).
- `configs/sweeps/pilot_soft_xsubj.yaml` (**50 runs**): tested on people the model never trained on (subjects 2 and 7; `cross_subject`). This is the more trustworthy test (see "new movies vs new people" below).

| Setting | What each electrode gets | Why it's there |
|---|---|---|
| `none` | no location | floor |
| `destrieux_hard` | its one Destrieux region (~160 small regions) | what papers do |
| `destrieux_gauss3` | **main soft setting**: share of each region nearby, where map points count less the farther they are (half as much at 3.5 mm, nothing past 9 mm) | the thing being tested; the fall-off follows the signal spread measured by McCarty et al. 2022 (see below) |
| `destrieux_sphere10` | share of each region within 10 mm, every point counting equally | a much wider mix: does width matter at all? |
| `destrieux_gauss3_randnbr` | same weights as gauss3, but the mixed-in regions are swapped for random ones | is it the *right* neighbours, or just spreading weight? |
| `ho_prob_argmax` / `ho_prob_soft` | Harvard-Oxford probabilistic atlas: most likely region vs. the mix of how often each region is found at that spot across people | a second, independent kind of "soft" |
| `coords` | exact x, y, z position | does soft help only by adding position detail? |
| `destrieux_hard_nearest`, `destrieux_gauss3_nearest` | new-people sweep only: like hard / gauss3, but see below | is soft helping only because it reduces "unknown" regions? |

Exactly how each mix is computed, step by step, with worked examples: [`METHOD.md`](METHOD.md).

**Why 3 mm weighted.** McCarty et al. (eNeuro 2022; 8,752 contacts, 71 patients) measured how fast the signals of two sEEG contacts in grey matter stop resembling each other with distance. With the average-of-all-electrodes reference we also use, the resemblance halves at about 3.6 mm (full width 7.14 mm). A bell curve with 3 mm width halves at 3.5 mm, so map points count in the same proportion. It's an anchor, not a proof: similarity between two contacts spreads wider than what one contact picks up, and their figure is for high-frequency activity.

**Why the "nearest" rows.** A new person's electrode can land in a region none of the training people had. It then gets a shared "unknown" code. Soft weights reach into neighbouring regions, so they hit "unknown" less often. Soft could then win just by covering new people better, not because mixing at borders helps. The `_nearest` rows give such electrodes the nearest region training did see (new option `spatial.unseen: nearest_seen`), so hard and soft are compared with the same coverage.

## Measured on the real electrode positions (no recordings needed)

707 Lite electrodes, from Neuroprobe's MNI coordinates. "Unknown weight" is for held-out subjects 2 and 7, with the model trained on subjects 1, 3, 4, 10.

| Setting | Electrodes with >1 region | Avg. weight on main region | Unknown weight, held-out people |
|---|---|---|---|
| destrieux_hard | 0% | 1.00 | 10.1% |
| destrieux_hard_nearest | 0% | 1.00 | 0.4% |
| destrieux_gauss3 | 77% | 0.75 | 7.2% |
| destrieux_gauss3_nearest | 76% | 0.76 | 0.0% |
| destrieux_gauss3_randnbr | 77% | 0.75 | 6.5% |
| destrieux_sphere10 | 92% | 0.60 | 3.5% |
| ho_prob_argmax | 0% | 1.00 | 5.0% |
| ho_prob_soft | 76% | 0.76 | 3.5% |

So:

- **The soft settings really mix:** 77% of electrodes get more than one region with 3 mm weighting.
- **The coverage worry is real:** soft cuts held-out "unknown" weight from 10.1% to 7.2%.
- **The `_nearest` rows remove most of that difference:** 0.4% for hard, 0.0% for soft. The small gap left is electrodes in white matter more than 5 mm from any cortex, which soft (reaching 9 mm) still places and hard doesn't.

## Rules, written before seeing results

`soft_report.py` applies these and prints a reading. It never submits anything. "Most seeds" = at least 4 of 5, paired by seed.

- **Check 0, location helps at all:** `destrieux_hard` beats `none` on average. If not, soft vs. hard can't be read; look at the Experiment 1 pilot.
- **Check 1, soft beats hard:** `destrieux_gauss3` minus `destrieux_hard` is at least **+0.01 AUC** on average and above 0 in most seeds.
- **Check 2, right neighbours:** `destrieux_gauss3` beats `destrieux_gauss3_randnbr` on average and in most seeds. If it fails, the gain looks like a smoothing effect, not real neighbours.
- **Check 3, not just coverage** (new-people sweep only): `gauss3_nearest` minus `hard_nearest` is at least +0.01 on average and above 0 in most seeds.

The report also shows, without pass/fail:

- `coords` vs. soft.
- Probabilistic soft vs. hard.
- The 10 mm ball vs. 3 mm weighted, and vs. hard.
- Soft minus hard for each test person.
- Cost per run.

**New movies vs. new people.** On new movies, the test electrodes were also in training. A fine mix of regions is close to unique per electrode, so it can work as an electrode name tag. If soft helps on new movies but not on new people, the report flags it; trust the new-people result.

## When and how to run

**Recommended: after the Experiment 1 pilot.**
- If Experiment 1's Gate 2 shows location doesn't help at all, this test can't say much.
- If Experiment 1 shows Lite doesn't hold up, change `data.electrodes: lite` to `all` in both sweep files first.

Steps 0–3 of `../spatialResolutionTesting/docs/pilot.md` cover the env, data, and filtered recordings, and they serve both experiments. Then, from the repo root:

```bash
bash softRegionTesting/submit.sh pilot_soft 8
bash softRegionTesting/submit.sh pilot_soft_xsubj 8
# when done:
source spatialResolutionTesting/slurm/activate_env.sh
python softRegionTesting/soft_report.py softRegionTesting/results/pilot_soft softRegionTesting/results/pilot_soft_xsubj
```

**Cost (a guess until the Experiment 1 pilot measures it):** at the assumed 0.5–1 GPU-hour per run, about 45–90 GPU-hours. The report prints the measured cost and projects it to the full 117-run sweep (`configs/sweeps/exp2_soft_regions.yaml`; submit with `bash softRegionTesting/submit.sh exp2_soft_regions 8`).

## Known limits

- **Small effect, few seeds.** The BaRISTA paper reports a seed-to-seed SD of about 0.016 AUC. A real gain of 0.01–0.02 may or may not show up with 5 seeds; the rules only give a rough signal.
- **One task, one atlas for the main test.** `speech` only, and Destrieux only. The probabilistic pair uses Harvard-Oxford (69–85 regions here), a coarser level.
- **The random-neighbour control picks from all regions**, not from regions near the electrode. It tests "right neighbours vs. any regions", not "these neighbours vs. other nearby ones".
- **Location error isn't tested.** Electrode positions and the warp to the template brain are off by a few mm. Soft weights may help just by absorbing that error, and nothing here separates the two.
- **Run-level scores only.** We can't yet tell which electrodes benefit. Testing whether border electrodes gain most needs a per-electrode score (planned later with BaRISTA).
- **`nearest_seen` is our own control**, not from a paper. It only changes codes for test people's electrodes in regions training never saw (about 10% of them here).

## Files

```
METHOD.md                             how each mix is computed, step by step
configs/sweeps/pilot_soft.yaml        40 runs, new movies
configs/sweeps/pilot_soft_xsubj.yaml  50 runs, new people
configs/sweeps/exp2_soft_regions.yaml full sweep (117 runs), after the pilot
soft_report.py                        checks above -> soft_report.md
submit.sh                             submits through ../spatialResolutionTesting/slurm
tests/                                sweep sizes, report rules, real sweep lines on fake data (LOCEMB_SLOW=1)
results/                              created by the runs (not in git)
```

Tests: `cd softRegionTesting && python -m pytest -q tests` (uses the shared code; set `LOCEMB_SLOW=1` for the 4-minute fake-data run).
