# F9 bwUniCluster Parallel Strong-Scaling Experiment

This experiment uses the F8 pulse configuration and measures strong scaling for
parallel CN and DSLP at one fixed time step.

## Configuration

| Parameter | Value |
|---|---|
| Cluster queue | `cpu_il` |
| Nodes | `1` for up to `64` ranks, `2` for larger rank counts |
| Methods | `CN`, `DSTLP` |
| Pulse data | `ic=pulse`, `rhs=zero` |
| Mesh size | `H=0.002` |
| FEM degree | `2`, consistent mass |
| Time step | `TAU=1e-3` |
| Final time | `T=1.0` |
| Overlap | `ell=4` |
| Partitioner | `scotch` |
| DSTLP prediction rule | `gamma=1`, `minimal_pred_ells=(2,1)` |
| Solver | direct Cholesky |

The intended rank/subdomain list is:

```text
CN:    1, 2, 4, 8, 16, 24, 32, 40, 52, 64, 80, 100, 128
DSTLP:    2, 4, 8, 16, 24, 32, 40, 52, 64, 80, 100, 128
```

Each rank count is submitted as its own Slurm job. The script writes one metrics
file per rank count:

```text
metrics_raw_procs<N>.csv
```

This avoids concurrent writes to one shared CSV.

## Reference Solution

The historical F8 archive is intentionally not included in this reproduction
bundle. Leave `REF_ROOT` and `REF_BP` unset for normal F9 jobs. Each rank-count
job then computes a new fallback CN reference inside its own archive. This is
more expensive than reusing an F8 reference, but makes the experiment
self-contained.

The dedicated one-rank workaround cannot generate a reference itself. Run at
least one normal F9 job first, then point its `REF_ROOT` to the current F9
archive as described below.

## Smoke Test

Run a small smoke test first:

```bash
cd /pfs/data6/home/ka/ka_ianm/ka_wb9658/domainsplitting/parallel
RUN_ID=$(date +%Y%m%dT%H%M%S)_F9_scaling_smoke
unset REF_ROOT REF_BP

sbatch -p dev_cpu_il -N 1 -n 4 --time=00:30:00 \
  --output=../sbatch_F9_smoke_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",PROCS=4,METHODS="CN DSTLP",H=0.002,DEGREE=2,TAU=1e-3,REPEATS=1 \
  ./f9_bwunicluster.sh
```

Check:

```bash
RUN_DIR="results/F9_parallel_strong_scaling/$RUN_ID"
find "$RUN_DIR" -name 'metrics_raw_procs*.csv' -print
column -s, -t "$RUN_DIR"/metrics_raw_procs4.csv | less -S
```

## Full Run

Use a fresh `RUN_ID` and submit one job per rank count. The following loop keeps
CN and DSTLP in the same rank-count jobs, except that `PROCS=1` runs CN only.

```bash
cd /pfs/data6/home/ka/ka_ianm/ka_wb9658/domainsplitting/parallel
RUN_ID=$(date +%Y%m%dT%H%M%S)_F9_scaling_full
echo "$RUN_ID"
unset REF_ROOT REF_BP

for P in 1 2 4 8 16 24 32 40 52 64 80 100 128; do
  if [ "$P" -eq 1 ]; then
    METHODS="CN"
  else
    METHODS="CN DSTLP"
  fi
  if [ "$P" -le 64 ]; then
    NODES=1
  else
    NODES=2
  fi
  sbatch -p cpu_il -N "$NODES" -n "$P" --time=02:00:00 \
    --output=../sbatch_F9_scaling_${P}_%j.log \
    --export=ALL,RUN_ID="$RUN_ID",PROCS="$P",METHODS="$METHODS",H=0.002,DEGREE=2,TAU=1e-3,REPEATS=1 \
    ./f9_bwunicluster.sh
done
```

If many jobs run concurrently, this is okay because each job writes its own
`metrics_raw_procs<N>.csv`.

## Direct High-Rank Retry

If the two-node direct runs time out during global CN setup/factorization, rerun
the missing high-rank points with a larger wall-time limit. Keep the existing
direct `RUN_ID` so that the successful rows are written into the same archive.

```bash
cd /pfs/data6/home/ka/ka_ianm/ka_wb9658/domainsplitting/parallel
RUN_ID=20260814T154227_F9_scaling_full
unset REF_ROOT REF_BP

for P in 80 100 128; do
  sbatch -p cpu_il -N 2 -n "$P" --time=12:00:00 \
    --output=../sbatch_F9_scaling_${P}_direct12h_%j.log \
    --export=ALL,RUN_ID="$RUN_ID",RESUME=1,PROCS="$P",METHODS="CN",H=0.002,DEGREE=2,TAU=1e-3,REPEATS=1 \
    ./f9_bwunicluster.sh
done
```

After these CN jobs have completed, run the corresponding direct DSTLP points:

```bash
cd /pfs/data6/home/ka/ka_ianm/ka_wb9658/domainsplitting/parallel
RUN_ID=20260814T154227_F9_scaling_full
unset REF_ROOT REF_BP

for P in 80 100 128; do
  sbatch -p cpu_il -N 2 -n "$P" --time=04:00:00 \
    --output=../sbatch_F9_scaling_${P}_DSTLP_direct_%j.log \
    --export=ALL,RUN_ID="$RUN_ID",RESUME=1,PROCS="$P",METHODS="DSTLP",H=0.002,DEGREE=2,TAU=1e-3,REPEATS=1 \
    ./f9_bwunicluster.sh
done
```

Do not run the CN-only and DSTLP-only jobs for the same `P` at the same time
with the same `RUN_ID`; both append to the same `metrics_raw_procs<P>.csv`.

## Iterative CG-GAMG Sweep

Use a separate `RUN_ID` for the iterative comparison. This keeps the raw data
separated from the direct Cholesky archive.

```bash
cd /pfs/data6/home/ka/ka_ianm/ka_wb9658/domainsplitting/parallel
RUN_ID=$(date +%Y%m%dT%H%M%S)_F9_scaling_iter_gamg_full
echo "$RUN_ID"
unset REF_ROOT REF_BP

for P in 2 4 8 16 24 32 40 52 64 80 100 128; do
  if [ "$P" -le 64 ]; then
    NODES=1
    WALLTIME=04:00:00
  else
    NODES=2
    WALLTIME=06:00:00
  fi

  sbatch -p cpu_il -N "$NODES" -n "$P" --time="$WALLTIME" \
    --output=../sbatch_F9_scaling_iter_gamg_${P}_%j.log \
    --export=ALL,RUN_ID="$RUN_ID",PROCS="$P",METHODS="CN DSTLP",H=0.002,DEGREE=2,TAU=1e-3,REPEATS=1,GLOBAL_SOLVING_TYPE=iterative,GLOBAL_ITERATIVE_METHOD=cg,GLOBAL_PRECONDITIONER=gamg,DSTLP_SOLVING_TYPE=iterative,DSTLP_ITERATIVE_METHOD=cg,DSTLP_PRECONDITIONER=gamg \
    ./f9_bwunicluster.sh
done
```

The iterative sweep omits `P=1`; use the workaround runner below for the
single-rank CN point.

## Output

The archive is written below:

```text
results/F9_parallel_strong_scaling/<RUN_ID>/
```

Important files:

| File or directory | Purpose |
|---|---|
| `manifest_procs<N>.json` | Parameters and Slurm metadata for one rank count |
| `metrics_raw_procs<N>.csv` | Timing and reference-error rows for one rank count |
| `logs/procs<N>/` | Method logs and reference-error output |
| `timings/procs<N>/` | External wall-time wrappers |
| `meshes/procs<N>/` | Rank-specific CN/DSTLP meshes |
| `solutions/procs<N>/` | Final method solutions |

After copying the archive back, generate plot data locally with:

```bash
python3 parallel/parallel/f9_collect_scaling_data.py \
  --archive results/paper_DSTLP/f9_parallel_strong_scaling/<RUN_ID> \
  --output results/tikz_DSTLP_F9_parallel/data/f9_parallel_strong_scaling.csv
```

## Optional CN Point With One MPI Rank

The normal F9 runner can hit a one-rank submeshing edge case. To obtain the
single-rank CN point, use the dedicated workaround runner. It creates the CN
mesh with two MPI ranks and then runs CN plus reference-error postprocessing
with one MPI rank. The row is appended to `metrics_raw_procs1.csv`. Wait for at
least one normal rank-count job with the same `RUN_ID` to finish first; the
workaround locates that job's newly generated reference below the F9 archive.

```bash
cd /pfs/data6/home/ka/ka_ianm/ka_wb9658/domainsplitting/parallel
RUN_ID=<same F9 RUN_ID>
REF_ROOT="results/F9_parallel_strong_scaling/$RUN_ID"

sbatch -p cpu_il -N 1 -n 2 --time=04:00:00 \
  --output=../sbatch_F9_CN_n1_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",RESUME=1,SUBMESH_PROCS=2,H=0.002,DEGREE=2,TAU=1e-3,REF_ROOT="$REF_ROOT",REPEATS=1 \
  ./f9_n1_cn_bwunicluster.sh
```

For the iterative CG-GAMG single-rank CN point, use the same iterative
`RUN_ID` as above:

```bash
cd /pfs/data6/home/ka/ka_ianm/ka_wb9658/domainsplitting/parallel
RUN_ID=<iterative F9 RUN_ID>
REF_ROOT="results/F9_parallel_strong_scaling/$RUN_ID"

sbatch -p cpu_il -N 1 -n 2 --time=04:00:00 \
  --output=../sbatch_F9_CN_n1_iter_gamg_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",RESUME=1,SUBMESH_PROCS=2,H=0.002,DEGREE=2,TAU=1e-3,REF_ROOT="$REF_ROOT",REPEATS=1,GLOBAL_SOLVING_TYPE=iterative,GLOBAL_ITERATIVE_METHOD=cg,GLOBAL_PRECONDITIONER=gamg \
  ./f9_n1_cn_bwunicluster.sh
```
