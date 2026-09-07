# Domain splitting for localized implicit prediction

This code was used for the numerical experiments in the paper

> Domain splitting for localized implicit prediction
>
> By T. Buchholz and R. Maier

This software is published in accordance with the guidelines for safeguarding
good research practice and serves to reproduce the experiments in the
above-mentioned publication.

## Description

This repository contains the serial and parallel experiment drivers used to
produce the paper figures for a domain-splitting time integrator with localized
implicit prediction. The implementation treats the linear wave equation in
first-order formulation and uses continuous finite elements in space.

The serial drivers for Figures 1–4 are in `scripts/` and use the modules in
`src/`. The cluster drivers for Figures 6 and 7 are in `parallel/` and use the
modules in `parallel/parallel/`. Stable CSV data and the corresponding TikZ
sources are included below `results/`.

Commands in this README are intended to be run from the repository root unless
the command explicitly changes directory. More detailed parameter descriptions
are available in `DriversGuide.md`.

## Installation and setup with Miniforge

We recommend installing conda through
[Miniforge](https://github.com/conda-forge/miniforge). On a Unix-like system,
download and run the installer with

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh"
```

Follow the interactive installation instructions, start a new shell, and
create the environment used by the experiment scripts:

```bash
conda create --name dscg-env -c conda-forge python=3.12
conda activate dscg-env

conda install -c conda-forge \
  "fenics-dolfinx=0.11.*" mpich adios2 adios4dolfinx \
  gmsh python-gmsh scifem pyvista h5py pandas scipy matplotlib \
  meshio sympy tqdm packaging
```

The F6/F7 scripts are configured for the bwUniCluster module system and also
activate `dscg-env`. On another cluster, adapt the `module load` commands to the
available compiler and MPI installation. The MPI implementation in the conda
environment must be compatible with the MPI launcher used for the run.

Compiling the figures additionally requires `latexmk` and a LaTeX installation
containing the `standalone`, `pgfplots`, `booktabs`, and `mathtools` packages.

## Reproduction

The stable CSV files used for the paper figures are already included. Thus, the
TikZ figures can be compiled immediately without rerunning the numerical
experiments. To regenerate the data, use the full-run commands below. Pilot or
smoke commands provide smaller checks of the same workflow.

Raw run archives are intentionally not included. Serial runs write new archives
below `results/paper_DSTLP/`; parallel runs write them below
`parallel/results/`.

### Figure 1: Problem 125 with mass-lumped degree-1 FEM

Run a short pilot or the full experiment:

```bash
mkdir -p results/paper_DSTLP

python3 scripts/paper_DSTLP_F1_problem125_femml.py --serial --pilot \
  2>&1 | tee results/paper_DSTLP/f1_problem125_femml_deg1_pilot.log

python3 scripts/paper_DSTLP_F1_problem125_femml.py --serial \
  2>&1 | tee results/paper_DSTLP/f1_problem125_femml_deg1_full.log
```

The driver prints the timestamped run directory below
`results/paper_DSTLP/f1_problem125_femml_deg1/`. Replace `<RUN_DIR>` below with
that directory name and copy the generated plotting data to its stable
location:

```bash
cp results/paper_DSTLP/f1_problem125_femml_deg1/<RUN_DIR>/plot.csv \
  results/tikz_DSTLP_F1_1D/data/problem125_FEMml_deg1_ell8_gamma1_output.csv
```

The editable TikZ source is
`results/tikz_DSTLP_F1_1D/problem125_FEMml_deg1_ell8_gamma1.tex`. Compile it with

```bash
cd results/tikz_DSTLP_F1_1D
latexmk -pdf -outdir=out problem125_FEMml_deg1_ell8_gamma1.tex
cd ../..
```

### Figure 2: Problem 125 with standard FEM degrees 1–4

Run the pilot or full degree sweep:

```bash
mkdir -p results/paper_DSTLP

python3 scripts/paper_DSTLP_F2_problem125_fem_degrees.py --serial --pilot \
  2>&1 | tee results/paper_DSTLP/f2_problem125_fem_degrees_pilot.log

python3 scripts/paper_DSTLP_F2_problem125_fem_degrees.py --serial \
  2>&1 | tee results/paper_DSTLP/f2_problem125_fem_degrees_full.log
```

Set `<RUN_DIR>` to the timestamped directory printed by the full driver, verify
that the degree outputs differ, and copy the four plotting tables:

```bash
RUN_DIR=results/paper_DSTLP/f2_problem125_fem_degrees/<RUN_DIR>

sha256sum "$RUN_DIR"/deg*/plot_deg*.csv

cp "$RUN_DIR/deg1/plot_deg1.csv" \
  results/tikz_DSTLP_F2_1D/data/problem125_FEM_deg1_ell8_gamma1_output.csv
cp "$RUN_DIR/deg2/plot_deg2.csv" \
  results/tikz_DSTLP_F2_1D/data/problem125_FEM_deg2_ell8_gamma1_output.csv
cp "$RUN_DIR/deg3/plot_deg3.csv" \
  results/tikz_DSTLP_F2_1D/data/problem125_FEM_deg3_ell8_gamma1_output.csv
cp "$RUN_DIR/deg4/plot_deg4.csv" \
  results/tikz_DSTLP_F2_1D/data/problem125_FEM_deg4_ell8_gamma1_output.csv
```

The editable TikZ files are the four
`results/tikz_DSTLP_F2_1D/problem125_FEM_deg*_ell8_gamma1.tex` files. Compile
them with

```bash
cd results/tikz_DSTLP_F2_1D
latexmk -pdf -outdir=out problem125_FEM_deg1_ell8_gamma1.tex
latexmk -pdf -outdir=out problem125_FEM_deg2_ell8_gamma1.tex
latexmk -pdf -outdir=out problem125_FEM_deg3_ell8_gamma1.tex
latexmk -pdf -outdir=out problem125_FEM_deg4_ell8_gamma1.tex
cd ../..
```

### Figure 3: Problem 125 overlap sweeps

Run the pilot or full overlap experiment:

```bash
mkdir -p results/paper_DSTLP

python3 scripts/paper_DSTLP_F3_overlap_sweeps.py --serial --pilot \
  2>&1 | tee results/paper_DSTLP/f3_overlap_sweeps_pilot.log

python3 scripts/paper_DSTLP_F3_overlap_sweeps.py --serial \
  2>&1 | tee results/paper_DSTLP/f3_overlap_sweeps_full.log
```

Copy the three generated sweep tables from the timestamped full-run archive:

```bash
RUN_DIR=results/paper_DSTLP/f3_problem125_overlap_sweeps/<RUN_DIR>

cp "$RUN_DIR/plot_ell_sweep.csv" \
  results/tikz_DSTLP_F3_1D/data/problem125_FEM_deg2_ell_sweep_output.csv
cp "$RUN_DIR/plot_gamma_sweep.csv" \
  results/tikz_DSTLP_F3_1D/data/problem125_FEM_deg2_gamma_sweep_output.csv
cp "$RUN_DIR/plot_min_pred_sweep.csv" \
  results/tikz_DSTLP_F3_1D/data/problem125_FEM_deg2_min_pred_sweep_output.csv
```

The editable TikZ source is
`results/tikz_DSTLP_F3_1D/problem125_FEM_deg2_overlap_sweeps.tex`. Compile it
with

```bash
cd results/tikz_DSTLP_F3_1D
latexmk -pdf -outdir=out problem125_FEM_deg2_overlap_sweeps.tex
cd ../..
```

### Figure 4: Problem 223 in two dimensions

Run the pilot or full two-dimensional experiment:

```bash
mkdir -p results/paper_DSTLP

python3 scripts/paper_DSTLP_F4_2D_DSTLP_problem223.py --pilot --batch-size 2 \
  2>&1 | tee results/paper_DSTLP/f4_2d_dstlp_problem223_pilot.log

python3 scripts/paper_DSTLP_F4_2D_DSTLP_problem223.py --batch-size 2 \
  2>&1 | tee results/paper_DSTLP/f4_2d_dstlp_problem223_full.log
```

Copy the generated table from the timestamped full-run archive:

```bash
RUN_DIR=results/paper_DSTLP/f4_2d_dstlp_problem223/<RUN_DIR>

cp "$RUN_DIR/plot.csv" \
  results/tikz_DSTLP_F4_2D/data/problem223_FEM_deg2_DSTLP_ell4_ell8_output.csv
```

The editable TikZ sources are the three
`results/tikz_DSTLP_F4_2D/problem223_FEM_deg2_DSTLP_ell4_ell8*.tex` files.
Compile them with

```bash
cd results/tikz_DSTLP_F4_2D
latexmk -pdf -outdir=out problem223_FEM_deg2_DSTLP_ell4_ell8.tex
latexmk -pdf -outdir=out problem223_FEM_deg2_DSTLP_ell4_ell8_a.tex
latexmk -pdf -outdir=out problem223_FEM_deg2_DSTLP_ell4_ell8_b.tex
cd ../..
```

### Figure 6: Parallel pulse work-precision

Figure 6 is intended to run through Slurm on bwUniCluster. Detailed cluster
notes are in `parallel/F6_bwunicluster.md`. From the repository root, submit a
small smoke run with

```bash
cd parallel

sbatch -p dev_cpu_il -N 1 -n 8 --time=00:30:00 \
  --output=../sbatch_F6_smoke_%j.log \
  --export=ALL,PROCS=8,NUM_TAUS=1,TAU_MAX=1e-3,TAU_MIN=1e-3,H=0.005,LF_DEGREE=1,REF_H=0.005,REF_DEGREE=2,REF_TAU=5e-4,REPEATS=1 \
  ./f6_bwunicluster.sh

cd ..
```

For the full experiment, use one run identifier for all three methods. Run each
submission below only after the preceding job has completed successfully so
that the later jobs can reuse the reference and archive created by the first
job:

```bash
cd parallel
RUN_ID=$(date +%Y%m%dT%H%M%S)_F6_pulse_procs64_full
echo "$RUN_ID"

sbatch -p cpu_il -N 1 -n 64 --time=08:00:00 \
  --output=../sbatch_F6_CN_full_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",METHODS="CN",PROCS=64,NUM_TAUS=10,TAU_MAX=1e-2,TAU_MIN=1e-4,H=0.002,LF_DEGREE=1,REF_H=0.001,REF_DEGREE=2,REF_TAU=2.5e-5,REPEATS=1 \
  ./f6_bwunicluster.sh

sbatch -p cpu_il -N 1 -n 64 --time=08:00:00 \
  --output=../sbatch_F6_DSTLP_full_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",RESUME=1,METHODS="DSTLP",PROCS=64,NUM_TAUS=10,TAU_MAX=1e-2,TAU_MIN=1e-4,H=0.002,LF_DEGREE=1,REF_H=0.001,REF_DEGREE=2,REF_TAU=2.5e-5,REPEATS=1 \
  ./f6_bwunicluster.sh

sbatch -p cpu_il -N 1 -n 64 --time=08:00:00 \
  --output=../sbatch_F6_LF_full_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",RESUME=1,METHODS="LF",PROCS=64,NUM_TAUS=10,TAU_MAX=1e-2,TAU_MIN=1e-4,H=0.002,LF_DEGREE=1,REF_H=0.001,REF_DEGREE=2,REF_TAU=2.5e-5,REPEATS=1 \
  ./f6_bwunicluster.sh

cd ..
```

The new raw data are written to
`parallel/results/F6_parallel_pulse_work_precision/<RUN_ID>/metrics_raw.csv`.
The TikZ plot uses a wide, one-row-per-timestep table. Postprocess the raw
metrics into the same column layout as the bundled stable CSV and place the
result at

```bash
cp <PROCESSED_F6_CSV> \
  results/tikz_DSTLP_F6_parallel/data/f6_parallel_pulse_work_precision.csv
```

The TikZ sources are in `results/tikz_DSTLP_F6_parallel/`. Compile the two
work-precision plots and the timing table with

```bash
cd results/tikz_DSTLP_F6_parallel
latexmk -pdf -outdir=out f6_parallel_pulse_work_precision_time_loop.tex
latexmk -pdf -outdir=out f6_parallel_pulse_work_precision_external_wall.tex
latexmk -pdf -outdir=out f6_parallel_pulse_timing_table.tex
cd ../..
```

### Figure 7: Parallel strong scaling

Figure 7 is also intended for Slurm on bwUniCluster; see
`parallel/F7_bwunicluster.md` for retry and single-rank instructions. The raw F6
reference archive is not included. Keep `REF_ROOT` and `REF_BP` unset so every
normal F7 rank-count job generates a new CN reference in its own archive.

Run a small smoke test with

```bash
cd parallel
RUN_ID=$(date +%Y%m%dT%H%M%S)_F7_scaling_smoke
unset REF_ROOT REF_BP

sbatch -p dev_cpu_il -N 1 -n 4 --time=00:30:00 \
  --output=../sbatch_F7_smoke_%j.log \
  --export=ALL,RUN_ID="$RUN_ID",PROCS=4,METHODS="CN DSTLP",H=0.002,DEGREE=2,TAU=1e-3,REPEATS=1 \
  ./f7_bwunicluster.sh

cd ..
```

Submit the full rank sweep with

```bash
cd parallel
RUN_ID=$(date +%Y%m%dT%H%M%S)_F7_scaling_full
echo "$RUN_ID"
unset REF_ROOT REF_BP

for P in 1 2 4 8 16 24 32 40 52 64 80 100 128; do
  if [ "$P" -eq 1 ]; then METHODS="CN"; else METHODS="CN DSTLP"; fi
  if [ "$P" -le 64 ]; then NODES=1; else NODES=2; fi

  sbatch -p cpu_il -N "$NODES" -n "$P" --time=02:00:00 \
    --output=../sbatch_F7_scaling_${P}_%j.log \
    --export=ALL,RUN_ID="$RUN_ID",PROCS="$P",METHODS="$METHODS",H=0.002,DEGREE=2,TAU=1e-3,REPEATS=1 \
    ./f7_bwunicluster.sh
done

cd ..
```

The cluster archive is written to
`parallel/results/F7_parallel_strong_scaling/<RUN_ID>/`. After transferring the
archive to the machine used for plotting, place it below
`results/paper_DSTLP/f7_parallel_strong_scaling/` and collect its per-rank data:

```bash
python3 parallel/parallel/f7_collect_scaling_data.py \
  --archive results/paper_DSTLP/f7_parallel_strong_scaling/<RUN_ID> \
  --output results/tikz_DSTLP_F7_parallel/data/f7_parallel_strong_scaling.csv
```

For the iterative CG-GAMG comparison, collect its separate archive into

```bash
python3 parallel/parallel/f7_collect_scaling_data.py \
  --archive results/paper_DSTLP/f7_parallel_strong_scaling/<ITERATIVE_RUN_ID> \
  --output results/tikz_DSTLP_F7_parallel/data/f7_parallel_strong_scaling_iterative_gamg.csv
```

The TikZ sources are in `results/tikz_DSTLP_F7_parallel/`. Compile the scaling
plots with

```bash
cd results/tikz_DSTLP_F7_parallel
latexmk -pdf -outdir=out f7_parallel_strong_scaling_time_loop.tex
latexmk -pdf -outdir=out f7_parallel_strong_scaling_total_wall.tex
latexmk -pdf -outdir=out f7_parallel_strong_scaling_time_loop_compare.tex
latexmk -pdf -outdir=out f7_parallel_strong_scaling_total_wall_compare.tex
cd ../..
```

## Acknowledgments

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research
Foundation) — Project-ID 258734477 — CRC 1173.
