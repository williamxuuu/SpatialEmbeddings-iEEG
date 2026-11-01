# LocationEmbeddings

How should intracranial EEG (sEEG/iEEG) models be told **where each electrode is**?

- **Experiment 1:** how detailed should that location be? Resolution vs. performance, per model and per preprocessing setup.
- **Experiment 2:** should an electrode near a border get a **weighted mix of nearby regions** instead of exactly one region?

The full plan, feasibility, and compute estimates are in [`docs/experiments.md`](docs/experiments.md). New to Slurm? Read [`docs/slurm_guide.md`](docs/slurm_guide.md).

## Layout

```
configs/
  base.yaml                 defaults for every run
  smoke.yaml                tiny synthetic run (laptop / first cluster test)
  neuroprobe.yaml           real data (Neuroprobe / Brain Treebank)
  sweeps/                   experiment grids -> one Slurm array each
src/locemb/
  data/                     synthetic data, Neuroprobe adapter, preprocessing
  spatial/atlases.py        atlases & cube grids: MNI point -> region
  spatial/assign.py         electrode -> region weights (hard / sphere / gaussian / prob / trilinear, controls)
  spatial/encodings.py      none / region-weighted / xyz-coordinate encoders
  models/                   small population transformer + region-pooled linear baseline
  train.py                  one run: train, early-stop, test, write metrics.json
  sweep.py                  sweep YAML -> runs/<exp>.jsonl (one line per run)
slurm/                      env setup, sbatch scripts, submit_sweep.sh, cluster.env.example
scripts/                    collect results, plot, estimate compute, fetch atlases, inspect Neuroprobe
tests/                      pytest (spatial logic + end-to-end tiny runs)
```

## Quick start (laptop)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                                        # ~10 s
python -m locemb.train --config configs/smoke.yaml spatial.type=region spatial.atlas=grid40
```

Any config value can be overridden on the command line as `key.subkey=value`.

## On the cluster

```bash
cp slurm/cluster.env.example slurm/cluster.env    # fill in account/QOS/paths
bash slurm/setup_env.sh                           # once, on a login node
bash slurm/submit_sweep.sh configs/sweeps/smoke.yaml 4
python scripts/collect_results.py results/smoke_sweep --sweep runs/smoke_sweep.jsonl
```

Then do the same with `configs/sweeps/exp1_resolution.yaml` and `exp2_soft_regions.yaml`. Plot with:

```bash
python scripts/plot_results.py results/exp1_resolution --kind curve
python scripts/plot_results.py results/exp2_soft_regions --kind dots
```

## Status / known gaps

- **Tested locally:** the synthetic pipeline, every spatial mode, the controls, the atlases (nilearn), result collection, and the plots.
  - On synthetic data with a known location rule, 40 mm cubes beat "no location" (test AUC 0.73 vs 0.57), and shuffling the locations removed the gain (0.55). This was a single seed and is a plumbing check only.
- **Not yet run:**
  - The Neuroprobe adapter was written from the Neuroprobe README without data access. Run `scripts/inspect_neuroprobe.py`, then fix the `VERIFY` notes.
  - The Slurm scripts have not run on PACE yet. The account, QOS, and GPU names in `cluster.env.example` need confirming.
- **The models are stand-ins.** The transformer is a small BaRISTA/PopT-style model trained from scratch. To ask "does the best resolution depend on architecture?", wrap the real models (BaRISTA, PopT, ...) behind `build_model` in `src/locemb/models/__init__.py`.
- **Lobe grouping is approximate:** keyword rules over Harvard-Oxford labels (`spatial/atlases.py`). For exact parity with BaRISTA, use the dataset's own labels (`spatial.source: dataset_labels`) once they're loaded.
