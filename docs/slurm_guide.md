# Slurm, quickly

## The idea

A shared cluster has a few **login nodes** and many **compute nodes** (the ones with GPUs).

- You log in to a login node. You edit code, install packages, and submit jobs there.
- **Never train on the login node.** It's shared by everyone, and admins kill heavy work there.
- **Slurm** is the scheduler. You hand it a **job**: a shell script plus a list of what it needs (1 GPU, 8 CPUs, 48 GB RAM, 4 hours).
- Slurm puts the job in a **queue**. It runs the job on a compute node when resources free up. The script's printed output goes to a **log file**.

You don't watch the job run. You submit it, walk away, and check the logs and results later.

## Words you'll see

| Word | Meaning |
|---|---|
| job | one submitted script |
| `sbatch` | submit a script as a job |
| `#SBATCH` lines | requests written at the top of the script (time, GPUs, memory...) |
| account (`-A`) | who gets billed. On PACE Phoenix it's your PI's account, usually like `gts-<PI username>` |
| QOS (`-q`) | which queue rules apply. On Phoenix, `embers` is free but can be **preempted** (stopped when paying jobs need the GPU). `inferno` is billed to the account and is not preempted |
| partition (`-p`) | a group of nodes, e.g. a GPU type. Often you can leave this out |
| `--gres=gpu:1` | "give me 1 GPU". Some clusters accept a type, e.g. `gpu:A100:1` |
| `--time` | maximum run time. The job is killed when it runs out. Shorter limits usually start sooner |
| job array | one submission that runs N copies of a script. Each copy gets a number in `$SLURM_ARRAY_TASK_ID`. **This is how we run a sweep:** task *i* runs line *i* of the sweep file |
| `%8` in `--array=0-92%8` | at most 8 tasks run at once (be polite on a shared cluster) |

The PACE-specific values above (account format, `embers`/`inferno`, GPU names) come from memory and a third-party script, not PACE's own docs, which I couldn't reach. Confirm them with your PI or at docs.pace.gatech.edu. If you're on **ICE** (the course cluster) rather than Phoenix, you usually leave the account and QOS empty.

## Commands you'll actually use

```bash
sbatch script.sbatch            # submit; prints a job ID
squeue -u $USER                 # your queued (PD) and running (R) jobs
scancel 123456                  # cancel a job (or a whole array)
scancel 123456_7                # cancel one array task
sacct -j 123456 --format=JobID,State,Elapsed,MaxRSS,ExitCode   # what happened
seff 123456                     # how much CPU/memory a finished job actually used (if installed)
tail -f logs/<name>_<jobid>_<task>.out                         # watch a log
salloc --gres=gpu:1 --time=01:00:00 -A <acct> -q <qos>         # interactive GPU shell for debugging
```

In `squeue`, the `REASON` column explains why a job hasn't started:

- `Priority` / `Resources`: you're waiting in line.
- `QOSMaxJobsPerUserLimit`: you hit a per-user cap.

## This repo's workflow

**One time, on a login node:**

```bash
git clone <your repo url> LocationEmbeddings && cd LocationEmbeddings
cp slurm/cluster.env.example slurm/cluster.env      # fill in account / QOS / paths
bash slurm/setup_env.sh                              # makes the Python env, caches atlases, runs tests
```

**Get the data** (once; check the download script first):

```bash
sbatch -A <acct> -q <qos> slurm/download_braintreebank.sbatch
source slurm/activate_env.sh && python scripts/inspect_neuroprobe.py   # resolve the VERIFY notes
```

**Every experiment:**

```bash
bash slurm/submit_sweep.sh configs/sweeps/smoke.yaml 4          # first: tiny synthetic sweep, checks the setup
squeue -u $USER                                                 # watch it
python scripts/collect_results.py results/smoke_sweep --sweep runs/smoke_sweep.jsonl

bash slurm/submit_sweep.sh configs/sweeps/exp1_resolution.yaml 8
python scripts/collect_results.py results/exp1_resolution --sweep runs/exp1_resolution.jsonl
python scripts/plot_results.py results/exp1_resolution --kind curve
```

`submit_sweep.sh` turns the sweep YAML into `runs/<exp>.jsonl`, one line per run. It then submits one job array with your `cluster.env` settings.

- **Finished runs are skipped.** If some tasks fail or get preempted, run the same command again and only the missing runs execute.
- **Interrupted runs resume** from their last saved epoch (`ckpt.pt`).

## Things that commonly go wrong

- **"No such file" for the log:** the `logs/` folder must exist before the job starts. `submit_sweep.sh` creates it for you.
- **Job dies at exactly the time limit:** raise `LOCEMB_TIME`, or let the checkpoint resume it on resubmission.
- **Downloads fail inside jobs:** compute nodes often have no internet. That's why `setup_env.sh` caches the atlases on the login node.
- **Home directory full:** home quotas are small. Put the Python env, the data, and `results/` on project or scratch storage.
- **Scratch is usually purged after some weeks.** Copy final `summary.csv`/plots somewhere safe.
- **The job ran but used no GPU:** check the log's first lines. The job prints `nvidia-smi`, and `metrics.json` records the GPU name.
