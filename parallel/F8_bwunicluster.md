# F8 bwUniCluster Parallel Pulse Work-Precision Experiment

This experiment uses travelling-pulse data and a numerically computed reference
solution. It is independent of the historical MMS workflow.

## Configuration

Default sweep:

| Parameter | Value |
|---|---|
| Cluster queue | `cpu_il` |
| Nodes | `1` |
| MPI ranks / subdomains | `64` |
| Pulse data | `ic=pulse`, `rhs=zero` |
| Pulse parameters | `mu=0.5`, `s=0.2`, `b=1.0`, amplitude factor `1.0` |
| CN/DSTLP sweep mesh | `H=0.002` |
| CN/DSTLP FEM degree | `2`, consistent mass |
| LF sweep mesh | `LF_H=H=0.002` |
| LF FEM degree | `LF_DEGREE=1`, mass lumped |
| Overlap | `ell=4` |
| Partitioner | `scotch` |
| Final time | `T=1.0` |
| Tau range | 10 fitted geometric values from `1e-2` to `1e-4` |
| DSTLP prediction rule | `gamma=1`, `minimal_pred_ells=(2,1)` |

Reference:

| Parameter | Value |
|---|---|
| Method | Crank-Nicolson |
| FEM | Degree `2`, consistent mass |
| Reference mesh | `REF_H=0.001` |
| Reference tau | `REF_TAU=2.5e-5` |

The reference is written once per archive, then every successful CN/LF/DSTLP
run is post-processed against it with `parallel/pulse_reference_errors.py`.

## Smoke Test

Use a deliberately cheap reference first:

```bash
cd /path/to/domainsplitting/parallel
sbatch -p dev_cpu_il -N 1 -n 8 --time=00:30:00 \
  --output=../sbatch_F8_smoke_%j.log \
  --export=ALL,PROCS=8,NUM_TAUS=1,TAU_MAX=1e-3,TAU_MIN=1e-3,H=0.005,LF_DEGREE=1,REF_H=0.005,REF_DEGREE=2,REF_TAU=5e-4,REPEATS=1 \
  ./f8_bwunicluster.sh
```

Acceptance checks:

- The archive contains `reference/CN/.../solCN_ref.bp`.
- `metrics_raw.csv` has one row per requested method/tau/repeat.
- The method logs contain `Reference solution file:` and relative error lines.
- DSTLP writes a global `solDSTLP.bp` in addition to rank-local files.

## Main Run

Submit method chunks into the same `RUN_ID`. Start with CN to build the
reference and global meshes:

Use a fresh `RUN_ID` after switching the reference method. Do not resume an
archive that was created with the old LF reference.

```bash
cd /path/to/domainsplitting/parallel
RUN_ID=$(date +%Y%m%dT%H%M%S)_F8_pulse_procs64_full
echo "$RUN_ID"

sbatch -p cpu_il -N 1 -n 64 --time=08:00:00 \
  --output=../sbatch_F8_CN_full_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",METHODS="CN",PROCS=64,NUM_TAUS=10,TAU_MAX=1e-2,TAU_MIN=1e-4,H=0.002,LF_DEGREE=1,REF_H=0.001,REF_DEGREE=2,REF_TAU=2.5e-5,REPEATS=1 \
  ./f8_bwunicluster.sh
```

Then add DSTLP after CN completes:

```bash
RUN_ID=<same RUN_ID>
sbatch -p cpu_il -N 1 -n 64 --time=08:00:00 \
  --output=../sbatch_F8_DSTLP_full_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",RESUME=1,METHODS="DSTLP",PROCS=64,NUM_TAUS=10,TAU_MAX=1e-2,TAU_MIN=1e-4,H=0.002,LF_DEGREE=1,REF_H=0.001,REF_DEGREE=2,REF_TAU=2.5e-5,REPEATS=1 \
  ./f8_bwunicluster.sh
```

Finally add LF. LF is run as the classical explicit mass-lumped P1 baseline on
the experiment mesh size `LF_H=H=0.002`:

```bash
RUN_ID=<same RUN_ID>
sbatch -p cpu_il -N 1 -n 64 --time=08:00:00 \
  --output=../sbatch_F8_LF_full_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",RESUME=1,METHODS="LF",PROCS=64,NUM_TAUS=10,TAU_MAX=1e-2,TAU_MIN=1e-4,H=0.002,LF_DEGREE=1,REF_H=0.001,REF_DEGREE=2,REF_TAU=2.5e-5,REPEATS=1 \
  ./f8_bwunicluster.sh
```

Do not run chunks concurrently with the same `RUN_ID`, because they append to
the same `metrics_raw.csv`.

## Output

The archive is written below:

```text
results/F8_parallel_pulse_work_precision/<RUN_ID>/
```

Important files:

| File or directory | Purpose |
|---|---|
| `manifest.json` | Sweep and reference parameters |
| `metrics_raw.csv` | Timing plus reference-based relative errors |
| `reference/` | Fine CN degree-2 consistent-mass reference solution |
| `logs/` | Method logs plus appended reference-error output |
| `timings/` | External wall-time files |
| `solutions/` | Final method solutions |

## Notes

The reference choice is intentionally explicit and may still need adjustment. If
the pulse reference is too expensive, first increase `REF_TAU`; if the reference
error plateau looks too high, decrease `REF_H` and `REF_TAU`.
