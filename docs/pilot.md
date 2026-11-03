# Pilot on Neuroprobe Lite

## What it asks

1. On real data, does telling the model where electrodes are help at all?
2. Does Neuroprobe Lite's cap of 120 electrodes per person change the answer?
   Lite kept the electrodes a linear model decoded best from, which could bias
   a study about location.

## What runs

`configs/sweeps/pilot_lite.yaml`: 7 settings × 2 electrode sets × 3 seeds = **42 small-transformer runs**.

| Setting | What the model is told about each electrode |
|---|---|
| none | nothing |
| lobes | which of ~20 big areas it's in |
| ho (Harvard-Oxford) | which of ~110 regions |
| destrieux | which of ~160 smaller regions (closest to BaRISTA's best level) |
| schaefer400 | which of ~400 regions |
| coords | its exact x, y, z position |
| destrieux_shuffled | Destrieux regions swapped randomly between electrodes (control) |

- **Electrode sets:**
  - `lite`: Neuroprobe's ≤120 electrodes per person.
  - `all`: every usable electrode.
- **Same windows and labels in both sets.** Only the electrodes differ, so the two can be compared directly.
- **Task:** `speech` (speech vs. no speech, 1 s windows), on the 12 Lite recordings.
- **Split:** `cross_session`. Each person trains on one Lite movie and is tested on their other one. All 6 people are trained into one model, so location codes are shared across people.

`configs/sweeps/pilot_lite_xsubj.yaml` (another 42 runs, so **84 in total**) is the same but tests on **people the model never trained on**: subjects 2 and 7. This is the harder and more relevant test for location.

Why run both: in `cross_session`, the test electrodes were also in training, so a very fine location code can work as an electrode ID tag. That would make fine levels look good for reasons that aren't about anatomy. `cross_subject` removes that.

## Real electrode positions (from Neuroprobe's MNI coordinates; no recordings needed)

|  | lite (707 electrodes) | all (1,108 electrodes) |
|---|---|---|
| Electrodes per person | 109–120 | 134–246 |
| Lobes: regions used per person | 6–9 | 6–9 |
| Harvard-Oxford | 22–28 | 22–37 |
| Destrieux | 24–34 (90 total) | 24–47 (99 total) |
| Schaefer 400 | 35–41 (140 total) | 40–72 (179 total) |
| Schaefer 400: share of regions with < 3 electrodes | 33% | 28% |

BaRISTA reported 19–47 parcels per person, which matches.

The "all" counts are approximate: corrupted electrodes can't be removed without the data download.

**Choosing the held-out people for `cross_subject`:** I checked every pair of the 6 subjects. For each pair, I measured how many of their electrodes land in a region that none of the other 4 people have, so the model has no learned vector for it.

- Subjects 2 + 7 were best: Destrieux 6% (all) / 10% (lite); Schaefer 400 20% / 28%.
- 3 + 10 (my earlier default) was among the worst.

Picking the best-covered pair is a deliberate choice. We want to measure resolution, not missing regions. Even so, at Schaefer 400 a fifth or more of the held-out electrodes have no learned region vector. That will drag the fine levels down in this split.

## Rules, written before seeing results

`scripts/pilot_report.py` checks these and prints a recommendation. **It never submits anything.**

- **Gate 1: the model learns something.** The best setting's mean AUC minus 2 SD is above 0.5.
- **Gate 2: location helps.** The best location setting beats "none" by at least 0.03 AUC on average **and** in every seed.
- **Gate 3: it's real location.** Destrieux beats shuffled Destrieux in every seed.
- **Lite holds up** if, between lite and all:
  - no two settings swap order when either gap is at least 0.02 AUC, and
  - no "location minus none" gain flips sign when either gain is at least 0.02.

If Gate 2 or 3 fails, don't scale up yet. Look at why first: the task, the model, or the preprocessing.

## Steps on the cluster

```bash
# 0. Once: settings and Python env (see docs/slurm_guide.md)
cp slurm/cluster.env.example slurm/cluster.env    # account, QOS, ROOT_DIR_BRAINTREEBANK (ends in /braintreebank), LOCEMB_CACHE_DIR
bash slurm/setup_env.sh

# 1. Data: 12 Lite recordings, 75 GB zipped (~130 GB unzipped is a guess; ~200 GB peak while unzipping).
#    Skip if the lab already has a copy: just point ROOT_DIR_BRAINTREEBANK at it.
sbatch -A <acct> -q <qos> slurm/download_braintreebank.sbatch

# 2. Look before running anything big (cheap; prints counts, checks files)
source slurm/activate_env.sh
python scripts/inspect_neuroprobe.py
python scripts/inspect_neuroprobe.py --build      # also filters 2 recordings and times it

# 3. Filter the 12 recordings once (CPU jobs; ~17 GB of output, ~45 min of filtering in total)
sbatch -A <acct> -q <qos> slurm/build_cache.sbatch            # prints a job id

# 4. The pilot, after step 3 finishes
bash slurm/submit_sweep.sh configs/sweeps/pilot_lite.yaml 8 --dependency=afterok:<job id from step 3>
bash slurm/submit_sweep.sh configs/sweeps/pilot_lite_xsubj.yaml 8 --dependency=afterok:<job id from step 3>

# 5. Read the results
python scripts/pilot_report.py results/pilot_lite
python scripts/pilot_report.py results/pilot_lite_xsubj
```

## Known limits

- **One task.** `speech` is the easiest Neuroprobe task. Any electrode near auditory cortex may be enough to solve it, which could hide location effects. If the gates pass, add a harder task (e.g. `onset` or `pitch`) before the full sweep.
- **One direction per person in `cross_session`.** We test on the second Lite movie only. Neuroprobe averages both directions.
- **Validation comes from the test recording** (its first half), as in Neuroprobe. Early stopping therefore sees the test person's data, but never the test windows.
- **The shuffled control only exists for Destrieux.**
- **Filtering:**
  - Order: band-pass at 2048 Hz, then resample to 512 Hz, then notch.
  - On a simulated signal this matched the textbook order within 1.5% power in every band from 1 to 200 Hz.
  - Each recording is stored as float16 after dividing each electrode by its own SD; the SD is multiplied back when read.
- **Tests:**
  - **Tested against Neuroprobe's own code** (pinned commit `b282acd`) on a fake data folder: electrode choice, window timing, held-out people and movies, and end-to-end runs.
  - **Not yet run on the real recordings.** Step 2 is the check.
