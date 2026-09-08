# Drivers Guide

This file records the commands needed to reproduce paper figures from fresh
driver runs. Run commands from the repository root unless noted otherwise.

## F4: Problem 125, FEM-ML Degree 1

This figure compares Crank-Nicolson, leapfrog, old domain splitting, and DSTLP
for the mass-lumped 1D problem 125.

Driver:

```bash
mkdir -p results/paper_DSTLP
python3 scripts/paper_DSTLP_F4_problem125_femml.py --serial --pilot \
  2>&1 | tee results/paper_DSTLP/f4_problem125_femml_deg1_pilot.log
```

Full run:

```bash
mkdir -p results/paper_DSTLP
python3 scripts/paper_DSTLP_F4_problem125_femml.py --serial \
  2>&1 | tee results/paper_DSTLP/f4_problem125_femml_deg1_full.log
```

Default parameters:

| Parameter | Value |
|---|---|
| Problem | `125` |
| Domain | `DS_1D_2SD_distorted` |
| Space discretization | `FEM_ml`, Lagrange degree `1` |
| `h` | `0.001` |
| `ell` | `8` |
| `T` | `5.0` |
| `tau` range | `1e-4` to `1e-1` |
| Number of full-run tau values | `20` |
| DSTLP `gamma` | `1.0` |
| DSTLP minimal prediction ells | `(2,1)` |

Each run writes a timestamped archive below:

```text
results/paper_DSTLP/f4_problem125_femml_deg1/
```

The archive contains:

```text
manifest.json
raw_cn.csv
raw_leapfrog.csv
raw_old_ds.csv
raw_dstlp.csv
plot.csv
```

To build the TikZ figure, copy the selected full-run `plot.csv` into the stable
TikZ data filename and compile:

```bash
cp results/paper_DSTLP/f4_problem125_femml_deg1/<RUN_DIR>/plot.csv \
  results/tikz_DSTLP_F4_1D/data/problem125_FEMml_deg1_ell8_gamma1_output.csv

cd results/tikz_DSTLP_F4_1D
latexmk -pdf -outdir=out problem125_FEMml_deg1_ell8_gamma1.tex
```

Replace `<RUN_DIR>` with the timestamped archive printed by the driver, for
example `20260730T120000_problem125_FEMml_deg1_ell8_gamma1_full`.

Editable plot file:

```text
results/tikz_DSTLP_F4_1D/problem125_FEMml_deg1_ell8_gamma1.tex
```

The plot includes these active curves:

- Crank-Nicolson error against the exact solution.
- Leapfrog error against the exact solution.
- Old domain splitting error against the exact solution, labelled
  `DS_{\ell}`.
- DSTLP error against the exact solution, labelled `DSTLP_{\ell}`.
- An `O(tau^2)` reference line.

The TikZ helper also contains commented-out lines for:

- `DS_{\ell} - CN`.
- `DSTLP_{\ell} - CN`.

## F5: Problem 125, Standard FEM Degrees 1-4

This figure family uses the same problem and time/domain parameters as F4, but
uses standard FEM without mass lumping. It produces one plot for each FEM degree
`1`, `2`, `3`, and `4`.

Do not use archive
`20260730T112845_problem125_FEM_deg1-2-3-4_ell8_gamma1_full`: its degree plot
CSVs are identical because the first F5 driver version did not propagate
`FEM_DEGREE` into modules that imported the config value by copy. The current
driver updates those module-level values and aborts if two degree outputs are
identical.

Pilot:

```bash
mkdir -p results/paper_DSTLP
python3 scripts/paper_DSTLP_F5_problem125_fem_degrees.py --serial --pilot \
  2>&1 | tee results/paper_DSTLP/f5_problem125_fem_degrees_pilot.log
```

Full run:

```bash
mkdir -p results/paper_DSTLP
python3 scripts/paper_DSTLP_F5_problem125_fem_degrees.py --serial \
  2>&1 | tee results/paper_DSTLP/f5_problem125_fem_degrees_full.log
```

Default parameters:

| Parameter | Value |
|---|---|
| Problem | `125` |
| Domain | `DS_1D_2SD_distorted` |
| Space discretization | `FEM`, Lagrange degrees `1,2,3,4` |
| `h` | `0.001` |
| `ell` | `8` |
| `T` | `5.0` |
| `tau` range | `1e-4` to `1e-1` |
| Number of full-run tau values | `20` |
| DSTLP `gamma` | `1.0` |
| DSTLP minimal prediction ells | `(2,1)` |
| Reference line | `1000*tau^2` |

Each run writes a timestamped archive below:

```text
results/paper_DSTLP/f5_problem125_fem_degrees/
```

The archive contains one subdirectory per degree:

```text
manifest.json
deg1/raw_leapfrog.csv
deg1/raw_dstlp.csv
deg1/plot_deg1.csv
...
deg4/raw_leapfrog.csv
deg4/raw_dstlp.csv
deg4/plot_deg4.csv
```

Before copying data into the TikZ folder, verify that the degree outputs differ:

```bash
sha256sum "$RUN_DIR"/deg*/plot_deg*.csv
```

To build the TikZ figures, copy the selected full-run plot CSVs into the stable
TikZ data filenames and compile:

```bash
RUN_DIR=results/paper_DSTLP/f5_problem125_fem_degrees/<RUN_DIR>

cp "$RUN_DIR/deg1/plot_deg1.csv" \
  results/tikz_DSTLP_F5_1D/data/problem125_FEM_deg1_ell8_gamma1_output.csv
cp "$RUN_DIR/deg2/plot_deg2.csv" \
  results/tikz_DSTLP_F5_1D/data/problem125_FEM_deg2_ell8_gamma1_output.csv
cp "$RUN_DIR/deg3/plot_deg3.csv" \
  results/tikz_DSTLP_F5_1D/data/problem125_FEM_deg3_ell8_gamma1_output.csv
cp "$RUN_DIR/deg4/plot_deg4.csv" \
  results/tikz_DSTLP_F5_1D/data/problem125_FEM_deg4_ell8_gamma1_output.csv

cd results/tikz_DSTLP_F5_1D
latexmk -pdf -outdir=out problem125_FEM_deg1_ell8_gamma1.tex
latexmk -pdf -outdir=out problem125_FEM_deg2_ell8_gamma1.tex
latexmk -pdf -outdir=out problem125_FEM_deg3_ell8_gamma1.tex
latexmk -pdf -outdir=out problem125_FEM_deg4_ell8_gamma1.tex
```

Replace `<RUN_DIR>` with the timestamped archive printed by the driver, for
example `20260730T130000_problem125_FEM_deg1-2-3-4_ell8_gamma1_full`.

Editable plot files:

```text
results/tikz_DSTLP_F5_1D/problem125_FEM_deg1_ell8_gamma1.tex
results/tikz_DSTLP_F5_1D/problem125_FEM_deg2_ell8_gamma1.tex
results/tikz_DSTLP_F5_1D/problem125_FEM_deg3_ell8_gamma1.tex
results/tikz_DSTLP_F5_1D/problem125_FEM_deg4_ell8_gamma1.tex
```

Each F5 plot includes:

- Leapfrog error against the exact solution.
- DSTLP error against the exact solution.
- DSTLP minus Crank-Nicolson.
- Crank-Nicolson error against the exact solution.
- An `O(tau^2)` reference line.

## F6: Problem 125, DSTLP-CN Overlap Sweeps

This figure uses problem `125`, standard FEM degree `2`, the same `h=0.001`,
`T=5.0`, and `tau` range `1e-4` to `1e-1` as F4/F5. It plots only the
`DSTLP - CN` relative `H_0^1 x L^2` difference.

Pilot:

```bash
mkdir -p results/paper_DSTLP
python3 scripts/paper_DSTLP_F6_overlap_sweeps.py --serial --pilot \
  2>&1 | tee results/paper_DSTLP/f6_overlap_sweeps_pilot.log
```

Full run:

```bash
mkdir -p results/paper_DSTLP
python3 scripts/paper_DSTLP_F6_overlap_sweeps.py --serial \
  2>&1 | tee results/paper_DSTLP/f6_overlap_sweeps_full.log
```

The driver creates three sweeps:

| Subplot | Fixed Parameters | Sweep |
|---|---|---|
| 1 | `gamma=1`, `min_pred_ells=(2,1)` | `ell=1,2,4,8,16,20` |
| 2 | `ell=4`, `min_pred_ells=(2,1)` | `gamma=0.5,1.0,1.5,2.0` |
| 3 | `ell=4`, `gamma=1.0` | `min_pred_ells=(1,0),(1,1),(2,1),(2,2),(3,1),(3,2),(3,3)` |

Each run writes a timestamped archive below:

```text
results/paper_DSTLP/f6_problem125_overlap_sweeps/
```

To build the TikZ figure, copy the selected full-run plot CSVs into the stable
TikZ data filenames and compile:

```bash
RUN_DIR=results/paper_DSTLP/f6_problem125_overlap_sweeps/<RUN_DIR>

cp "$RUN_DIR/plot_ell_sweep.csv" \
  results/tikz_DSTLP_F6_1D/data/problem125_FEM_deg2_ell_sweep_output.csv
cp "$RUN_DIR/plot_gamma_sweep.csv" \
  results/tikz_DSTLP_F6_1D/data/problem125_FEM_deg2_gamma_sweep_output.csv
cp "$RUN_DIR/plot_min_pred_sweep.csv" \
  results/tikz_DSTLP_F6_1D/data/problem125_FEM_deg2_min_pred_sweep_output.csv

cd results/tikz_DSTLP_F6_1D
latexmk -pdf -outdir=out problem125_FEM_deg2_overlap_sweeps.tex
```

Editable plot file:

```text
results/tikz_DSTLP_F6_1D/problem125_FEM_deg2_overlap_sweeps.tex
```

## F7: Problem 223, 2D DSTLP Comparison

This figure uses problem `223`, standard FEM degree `2`, `h=0.005`, `T=1.0`,
and the `tau` range `1e-4` to `1e-1` with geometric spacing. It compares
Crank-Nicolson, leapfrog, and DSTLP for overlaps `ell=4` and `ell=8`. The dashed
DSTLP-CN difference curves are drawn in the same colors as the corresponding
DSTLP curves and are intentionally omitted from the legend.

The DSTLP prediction layers use `gamma=1` and `min_pred_ells=(2,1)`:
`inner=max(2, ceil(gamma*tau*c/hmin))` and
`outer=max(1, ceil(gamma*tau*c/hmin))`.

Pilot:

```bash
mkdir -p results/paper_DSTLP
python3 scripts/paper_DSTLP_F7_2D_DSTLP_problem223.py --pilot --batch-size 2 \
  2>&1 | tee results/paper_DSTLP/f7_2d_dstlp_problem223_pilot.log
```

Full run:

```bash
mkdir -p results/paper_DSTLP
python3 scripts/paper_DSTLP_F7_2D_DSTLP_problem223.py --batch-size 2 \
  2>&1 | tee results/paper_DSTLP/f7_2d_dstlp_problem223_full.log
```

Default parameters:

| Parameter | Value |
|---|---|
| Problem | `223` |
| Domain | `DS_2D_4x4SD_cross_SQUARE` |
| Space discretization | `FEM`, Lagrange degree `2` |
| `h` | `0.005` |
| `ell` | `4,8` |
| DSTLP `gamma` | `1.0` |
| DSTLP minimal prediction ells | `(2,1)` |
| `T` | `1.0` |
| `tau` range | `1e-4` to `1e-1` |
| Number of full-run tau values | `20` |
| Reference line | `tau^2` |

Each run writes a timestamped archive below:

```text
results/paper_DSTLP/f7_2d_dstlp_problem223/
```

To build the TikZ figure, copy the selected full-run plot CSV into the stable
TikZ data filename and compile:

```bash
RUN_DIR=results/paper_DSTLP/f7_2d_dstlp_problem223/<RUN_DIR>

cp "$RUN_DIR/plot.csv" \
  results/tikz_DSTLP_F7_2D/data/problem223_FEM_deg2_DSTLP_ell4_ell8_output.csv

cd results/tikz_DSTLP_F7_2D
latexmk -pdf -outdir=out problem223_FEM_deg2_DSTLP_ell4_ell8.tex
latexmk -pdf -outdir=out problem223_FEM_deg2_DSTLP_ell4_ell8_a.tex
latexmk -pdf -outdir=out problem223_FEM_deg2_DSTLP_ell4_ell8_b.tex
```

Editable plot file:

```text
results/tikz_DSTLP_F7_2D/problem223_FEM_deg2_DSTLP_ell4_ell8.tex
results/tikz_DSTLP_F7_2D/problem223_FEM_deg2_DSTLP_ell4_ell8_a.tex
results/tikz_DSTLP_F7_2D/problem223_FEM_deg2_DSTLP_ell4_ell8_b.tex
```

## F8: Parallel Pulse Work-Precision

This figure uses the travelling pulse example. It compares parallel CN, DSTLP
and mass-lumped P1 leapfrog on the same pulse data. Errors are measured against
a finer CN reference solution.

Detailed cluster instructions:

```text
parallel/F8_bwunicluster.md
```

Default parameters:

| Parameter | Value |
|---|---|
| Cluster queue | `cpu_il` |
| MPI ranks / subdomains | `64` |
| Pulse data | `ic=pulse`, `rhs=zero` |
| CN/DSTLP mesh | `H=0.002` |
| CN/DSTLP FEM degree | `2`, consistent mass |
| LF mesh | `LF_H=H=0.002` |
| LF FEM degree | `1`, mass lumped |
| Overlap | `ell=4` |
| Final time | `T=1.0` |
| Tau range | fitted geometric values from `1e-2` to `1e-4` |
| Reference | CN, `REF_H=0.001`, degree `2`, `REF_TAU=2.5e-5` |

Cluster smoke run:

```bash
cd /pfs/data6/home/ka/ka_ianm/ka_wb9658/domainsplitting/parallel
sbatch -p dev_cpu_il -N 1 -n 8 --time=00:30:00 \
  --output=../sbatch_F8_smoke_%j.log \
  --export=ALL,PROCS=8,NUM_TAUS=1,TAU_MAX=1e-3,TAU_MIN=1e-3,H=0.005,LF_DEGREE=1,REF_H=0.005,REF_DEGREE=2,REF_TAU=5e-4,REPEATS=1 \
  ./f8_bwunicluster.sh
```

Full run, split by method into one archive:

```bash
cd /pfs/data6/home/ka/ka_ianm/ka_wb9658/domainsplitting/parallel
RUN_ID=$(date +%Y%m%dT%H%M%S)_F8_pulse_procs64_full
echo "$RUN_ID"

sbatch -p cpu_il -N 1 -n 64 --time=08:00:00 \
  --output=../sbatch_F8_CN_full_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",METHODS="CN",PROCS=64,NUM_TAUS=10,TAU_MAX=1e-2,TAU_MIN=1e-4,H=0.002,LF_DEGREE=1,REF_H=0.001,REF_DEGREE=2,REF_TAU=2.5e-5,REPEATS=1 \
  ./f8_bwunicluster.sh

sbatch -p cpu_il -N 1 -n 64 --time=08:00:00 \
  --output=../sbatch_F8_DSTLP_full_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",RESUME=1,METHODS="DSTLP",PROCS=64,NUM_TAUS=10,TAU_MAX=1e-2,TAU_MIN=1e-4,H=0.002,LF_DEGREE=1,REF_H=0.001,REF_DEGREE=2,REF_TAU=2.5e-5,REPEATS=1 \
  ./f8_bwunicluster.sh

sbatch -p cpu_il -N 1 -n 64 --time=08:00:00 \
  --output=../sbatch_F8_LF_full_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",RESUME=1,METHODS="LF",PROCS=64,NUM_TAUS=10,TAU_MAX=1e-2,TAU_MIN=1e-4,H=0.002,LF_DEGREE=1,REF_H=0.001,REF_DEGREE=2,REF_TAU=2.5e-5,REPEATS=1 \
  ./f8_bwunicluster.sh
```

The historical raw F8 archive used for the paper is intentionally not included
in this reproduction bundle. A full run creates a new archive below
`results/F8_parallel_pulse_work_precision/`; the stable plot CSV used for the
paper is included in the TikZ directory below.

Stable TikZ data and plot files:

```text
results/tikz_DSTLP_F8_parallel/data/f8_parallel_pulse_work_precision.csv
results/tikz_DSTLP_F8_parallel/f8_parallel_pulse_work_precision_time_loop.tex
results/tikz_DSTLP_F8_parallel/f8_parallel_pulse_work_precision_external_wall.tex
results/tikz_DSTLP_F8_parallel/f8_parallel_pulse_timing_table.tex
```

Compile:

```bash
cd results/tikz_DSTLP_F8_parallel
latexmk -pdf -outdir=out f8_parallel_pulse_work_precision_time_loop.tex
latexmk -pdf -outdir=out f8_parallel_pulse_work_precision_external_wall.tex
latexmk -pdf -outdir=out f8_parallel_pulse_timing_table.tex
```

## F9: Parallel Strong Scaling

This figure uses the F8 pulse configuration but fixes the timestep at
`tau=1e-3` and varies the number of MPI ranks/subdomains. It compares parallel
CN and DSTLP. CN is also run with one rank; DSTLP starts at two ranks.

Detailed cluster instructions:

```text
parallel/F9_bwunicluster.md
```

Default parameters:

| Parameter | Value |
|---|---|
| Cluster queue | `cpu_il` |
| Nodes | `1` up to `64` ranks, `2` above `64` ranks |
| Methods | `CN`, `DSTLP` |
| Mesh size | `H=0.002` |
| FEM degree | `2`, consistent mass |
| Time step | `TAU=1e-3` |
| Final time | `T=1.0` |
| Overlap | `ell=4` |
| Rank list | `1,2,4,8,16,24,32,40,52,64,80,100,128` |

Raw F8 archives and their reference solutions are intentionally not included
in this reproduction bundle. Leave `REF_ROOT` and `REF_BP` unset when running
the normal F9 runner. It will generate a new CN reference inside each
rank-count archive. This makes the F9 sweep more expensive, but keeps it
self-contained.

Cluster smoke run:

```bash
cd /pfs/data6/home/ka/ka_ianm/ka_wb9658/domainsplitting/parallel
RUN_ID=$(date +%Y%m%dT%H%M%S)_F9_scaling_smoke
unset REF_ROOT REF_BP

sbatch -p dev_cpu_il -N 1 -n 4 --time=00:30:00 \
  --output=../sbatch_F9_smoke_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",PROCS=4,METHODS="CN DSTLP",H=0.002,DEGREE=2,TAU=1e-3,REPEATS=1 \
  ./f9_bwunicluster.sh
```

Full sweep:

```bash
cd /pfs/data6/home/ka/ka_ianm/ka_wb9658/domainsplitting/parallel
RUN_ID=$(date +%Y%m%dT%H%M%S)_F9_scaling_full
echo "$RUN_ID"
unset REF_ROOT REF_BP

for P in 1 2 4 8 16 24 32 40 52 64 80 100 128; do
  if [ "$P" -eq 1 ]; then METHODS="CN"; else METHODS="CN DSTLP"; fi
  if [ "$P" -le 64 ]; then NODES=1; else NODES=2; fi

  sbatch -p cpu_il -N "$NODES" -n "$P" --time=02:00:00 \
    --output=../sbatch_F9_scaling_${P}_%j.log \
    --export=ALL,RUN_ID="$RUN_ID",PROCS="$P",METHODS="$METHODS",H=0.002,DEGREE=2,TAU=1e-3,REPEATS=1 \
    ./f9_bwunicluster.sh
done
```

After copying the archive back, collect the per-rank metrics into the stable
plot CSV:

```bash
python3 parallel/parallel/f9_collect_scaling_data.py \
  --archive results/paper_DSTLP/f9_parallel_strong_scaling/<RUN_ID> \
  --output results/tikz_DSTLP_F9_parallel/data/f9_parallel_strong_scaling.csv
```

Stable TikZ data and plot files:

```text
results/tikz_DSTLP_F9_parallel/data/f9_parallel_strong_scaling.csv
results/tikz_DSTLP_F9_parallel/data/f9_parallel_strong_scaling_iterative_gamg.csv
results/tikz_DSTLP_F9_parallel/f9_parallel_strong_scaling_time_loop.tex
results/tikz_DSTLP_F9_parallel/f9_parallel_strong_scaling_total_wall.tex
results/tikz_DSTLP_F9_parallel/f9_parallel_strong_scaling_time_loop_compare.tex
results/tikz_DSTLP_F9_parallel/f9_parallel_strong_scaling_total_wall_compare.tex
```

Compile:

```bash
cd results/tikz_DSTLP_F9_parallel
latexmk -pdf -outdir=out f9_parallel_strong_scaling_time_loop.tex
latexmk -pdf -outdir=out f9_parallel_strong_scaling_total_wall.tex
latexmk -pdf -outdir=out f9_parallel_strong_scaling_time_loop_compare.tex
latexmk -pdf -outdir=out f9_parallel_strong_scaling_total_wall_compare.tex
```

For the iterative CG-GAMG comparison, use a separate F9 archive and collect it
into:

```bash
python3 parallel/parallel/f9_collect_scaling_data.py \
  --archive results/paper_DSTLP/f9_parallel_strong_scaling/<ITERATIVE_RUN_ID> \
  --output results/tikz_DSTLP_F9_parallel/data/f9_parallel_strong_scaling_iterative_gamg.csv
```
