# Experiment plan

Both experiments use the same pipeline. Only the **spatial encoding** changes from run to run: the extra input that tells the model where each electrode sits. Everything else stays fixed: data, model, training, and seeds.

---

## Experiment 1: how detailed should location be?

**Question.** As we go from coarse locations (lobes) to fine ones (small parcels, then exact coordinates), how does decoding performance change? Is the best level the same for every model?

**What changes** (`configs/sweeps/exp1_resolution.yaml`, 93 runs):

| Group | Settings | Why |
|---|---|---|
| Reference | none, xyz coordinates | floor, and the "infinitely fine" end |
| Cubes | 80, 40, 20, 10, 5 mm | pure resolution knob; no anatomy |
| Atlases | lobes → Harvard-Oxford → Destrieux → Schaefer 100/200/400/1000 | real brain regions, coarse → fine |
| Controls | Destrieux and 20 mm cubes with locations **shuffled** between electrodes | same number of regions, but the location link is broken |

These run on two models (the small transformer and the linear baseline) with 3 seeds each.

**Logged with every run, because these can fake a result:**

- `electrodes_per_region_median` and `frac_regions_lt3_electrodes`. At fine levels, most regions have only 1–2 electrodes, so their vectors barely get trained. A drop at fine levels may mean "too little data per region", not "too much detail".
- `frac_weight_unknown_test_subjects`. In cross-subject tests, a new person's electrodes often land in regions never seen in training. These fall back to one shared "unknown" vector.

**Comparing cubes and atlases** at the same number of regions separates two things:

- "finer helps", and
- "real anatomical borders help".

---

## Experiment 2: overlapping regions (soft spatial encoding)

### The idea

Every model in our library gives each electrode exactly **one** region. But an electrode on a border sits between regions. Its signal may reflect both sides, and atlas borders are themselves fuzzy across people.

The proposal: give each electrode a **weighted mix** of the regions around it. For example: 48% precentral gyrus, 41% central sulcus, 11% postcentral gyrus.

The model still has one learned vector per region. An electrode's spatial vector becomes the weighted sum of its regions' vectors. Hard assignment is the special case where one weight is 1. So the model, its size, and its compute are **unchanged**. Only the weights change.

### Ways to get the weights (all implemented in `src/locemb/spatial/assign.py`)

| Mode | What the weight means |
|---|---|
| `sphere` (radius r) | share of brain points within r mm of the electrode that belong to each region |
| `gaussian` (sigma) | same, but closer points count more |
| `prob` (probabilistic atlas) | across the people the atlas was built from, how often this exact spot was region X |
| `trilinear` (cubes) | blend of the nearest cube centres; a no-anatomy version of "soft" |

A real check at known MNI spots (Destrieux, 5 mm sphere) gave sensible mixes:

- Precentral spot: precentral 0.48, central sulcus 0.41, postcentral 0.11.
- Fusiform spot: lateral occipito-temporal sulcus 0.65, fusiform 0.35.

### Main comparisons (`softRegionTesting/configs/sweeps/exp2_soft_regions.yaml`, 117 runs)

A smaller pilot with written-down pass/fail rules comes first: [`softRegionTesting/README.md`](../../softRegionTesting/README.md).

- **Soft vs hard, same atlas:** Destrieux (fine) and lobes (coarse), several radii.
- **Probabilistic atlas vs its own hard version** (`ho_prob_soft` vs `ho_prob_argmax`).
- **Random-neighbour control:** keep each electrode's main region and the same weights, but swap the mixed-in regions for random ones. If this does as well as real soft weights, the gain comes from "spreading weight" (a smoothing or regularising effect), not from the right neighbours.
- **xyz coordinates.** Soft weights partly say *where inside* a region an electrode sits, which makes them part coordinate encoding. If plain coordinates do as well, the "overlapping function" story gets weaker.

`collect_results.py` prints soft-minus-hard differences paired by seed.

### Other explanations to rule out

A win for soft encoding would **not by itself** show that border areas have mixed functions. Each of these would also explain a win:

1. **Location error.** Electrode positions and the warp to the template brain are off by a few mm. Soft weights may just absorb that error.
2. **Sharing between regions.** Rare regions get trained through their neighbours' electrodes. Compare at matched `electrodes_per_region`.
3. **Extra position information** (see the coordinates point above).
4. **White-matter contacts.** Many sEEG contacts sit in white matter with no cortical label. Soft modes find nearby grey matter more often than hard mode does. Check `frac_weight_unknown_*` and compare runs where this fraction is similar.

The random-neighbour control and the coordinates baseline address 2 and 3. For 1 there is no clean fix, so state it as a limit.

A stronger test of the biological story is **per-electrode**: does the soft-minus-hard gain grow with how mixed an electrode is (its weight entropy, which is already logged)? That needs an electrode-level score, such as BaRISTA's masked-channel reconstruction error. It's a good phase-2 analysis once the real models are plugged in.

### Natural extensions (not built yet)

- **Learned overlap:** let the model learn each region's width, e.g. a learnable sigma.
- **Functional atlases:** Schaefer parcels come from fMRI connectivity. Soft weights over a functional atlas fit the "multiple functions" idea more directly than an anatomical atlas.
- **Bipolar referencing** (a preprocessing choice) makes each channel the *difference* of two neighbouring contacts, so the channel itself mixes two places. Soft encoding of the two contacts' regions is the natural match. This links Experiment 2 to the preprocessing axis.

### Feasibility

| | Verdict |
|---|---|
| Code | **Easy, done.** Weights are computed once per electrode before training. The model adds up to 8 vector lookups per electrode instead of 1. |
| Compute | **Negligible extra cost** over hard encoding. Same model size, same speed. |
| Data | Brain Treebank: 10 subjects. Neuroprobe provides MNI coordinates per electrode, which is what the atlas lookups need. |
| Statistics | **The real risk.** BaRISTA reports seed-to-seed SD of about ±0.016 AUC, so a soft-vs-hard gain of 0.01–0.02 would be hard to see with 3 seeds. Use paired comparisons (same seed, same subject) and per-subject AUCs (already saved). Add seeds for the settings that look promising. |

---

## Compute estimates

**These are guesses until a pilot runs.** Replace them with measured times:

```bash
python scripts/estimate_compute.py configs/sweeps/exp1_resolution.yaml ../softRegionTesting/configs/sweeps/exp2_soft_regions.yaml \
  --pilot results/<pilot exp> --parallel 8
```

### Route A: this repo's small model, trained from scratch per setting

Assumed time per run:

- **Transformer: ~0.5–1 GPU-hour.** Based on 1 s windows, about 100–250 electrodes, a 4-layer d=64 transformer, and 40 epochs. Data loading may be the bottleneck rather than the GPU; the pilot will measure it.
- **Linear model: ~0.25 hour.**

| Sweep | Runs | Rough GPU-hours |
|---|---|---|
| Exp 1 resolution | 93 (48 transformer, 45 linear) | ~35–60 |
| Exp 2 soft regions | 117 (60 transformer, 57 linear) | ~45–75 |
| **Total** | 210 | **~80–135**, i.e. ~10–17 hours of wall-clock time at 8 GPUs at once, plus queue wait |

### Route B: real foundation models with pretraining per setting

Changing the spatial encoding of a pretrained model usually means **redoing the pretraining**. The BaRISTA paper reports "<< 4 hours" of pretraining on 4 GPUs (RTX 6000 Ada / A6000) for a 1M-parameter model. That's at most ~16 GPU-hours per pretraining.

- 16 Exp-1 settings × 3 seeds = 48 pretrainings → **at most ~770 GPU-hours**, likely much less.
- Cheaper option to discuss with your PI: pretrain once per model without a spatial code, then add and fine-tune only the spatial encoding. That's cheaper, but it answers a slightly different question.

### Ways to cut cost

- Whole recordings are filtered once and cached (done: `data/session_cache.py`).
- Run the linear model on CPU nodes.
- Start with 1 seed to find interesting regions of the curve, then add seeds only there.

---

## Order of work

1. **Smoke sweep on the cluster** (synthetic data, minutes): proves the account, env, and GPU setup work.
2. **Pilot on Neuroprobe Lite** (`docs/pilot.md`): 42 runs + 42 cross-subject runs. Checks that location helps at all, that Lite's 120-electrode cap doesn't change the answer, and measures time per run.
3. **Decide how to scale** from the pilot report, then re-estimate the budget.
4. **Exp 1** with 1 seed, then 3 seeds.
5. **Exp 2:** first its pilot (`softRegionTesting/`, 90 runs; best after the Exp 1 pilot shows location helps), then the full sweep.
6. **Plug in a real architecture** (BaRISTA is the natural first one; its code is public at github.com/ShanechiLab/BaRISTA) behind the same `build_model` interface, then repeat the key settings.
7. **Preprocessing axis:** repeat the best and worst settings under 2–3 `preprocess:` settings.
