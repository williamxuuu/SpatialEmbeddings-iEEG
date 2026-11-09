# SpatialEmbeddings

How should intracranial EEG (sEEG/iEEG) models be told **where each electrode is**? Tested on Brain Treebank through the Neuroprobe benchmark.

| Folder | What's in it |
|---|---|
| [`spatialResolutionTesting/`](spatialResolutionTesting/) | **Shared code** (data loading, models, training, Slurm scripts, tests) and **Experiment 1**: how detailed should location be (lobes → small regions → exact coordinates)? Start with its [`docs/pilot.md`](spatialResolutionTesting/docs/pilot.md). |
| [`softRegionTesting/`](softRegionTesting/) | **Experiment 2**: should an electrode get a weighted mix of nearby regions instead of exactly one? Sweeps, pilot rules, and report; runs through the code in `spatialResolutionTesting/`. See its [`README.md`](softRegionTesting/README.md). |

The full plan, feasibility, and compute estimates: [`spatialResolutionTesting/docs/experiments.md`](spatialResolutionTesting/docs/experiments.md).
