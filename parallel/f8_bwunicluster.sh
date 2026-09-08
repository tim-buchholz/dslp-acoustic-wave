#!/bin/bash
# F8 pulse work-precision experiment for bwUniCluster Ice Lake nodes.

set -o pipefail

module load compiler/gnu/14.2
module load mpi/openmpi/5.0-gnu-14.2
module load devel/miniforge

conda activate dscg-env

set -u

PROCS=${PROCS:-${SLURM_NTASKS:-64}}
RUN_ID=${RUN_ID:-$(date +%Y%m%dT%H%M%S)_F8_pulse_procs${PROCS}}
REPEATS=${REPEATS:-1}
T=${T:-1.0}
H=${H:-0.002}
DEGREE=${DEGREE:-2}
ELL=${ELL:-4}
LF_H=${LF_H:-${H}}
LF_DEGREE=${LF_DEGREE:-1}
LF_ELL=${LF_ELL:-${ELL}}
GAMMA=${GAMMA:-1.0}
WAVE_SPEED=${WAVE_SPEED:-1.0}
MIN_INNER=${MIN_INNER:-2}
MIN_OUTER=${MIN_OUTER:-1}
PULSE_MU=${PULSE_MU:-0.5}
PULSE_S=${PULSE_S:-0.2}
PULSE_B=${PULSE_B:-1.0}
PULSE_AMPLITUDE_FACTOR=${PULSE_AMPLITUDE_FACTOR:-1.0}
REF_H=${REF_H:-0.001}
REF_TAU=${REF_TAU:-2.5e-5}
REF_DEGREE=${REF_DEGREE:-2}
REF_ELL=${REF_ELL:-1}
REF_MESH_REFINEMENT=${REF_MESH_REFINEMENT:-1}
GLOBAL_SOLVING_TYPE=${GLOBAL_SOLVING_TYPE:-direct}
GLOBAL_DIRECT_METHOD=${GLOBAL_DIRECT_METHOD:-cholesky}
GLOBAL_ITERATIVE_METHOD=${GLOBAL_ITERATIVE_METHOD:-cg}
GLOBAL_PRECONDITIONER=${GLOBAL_PRECONDITIONER:-icc}
GLOBAL_MAX_ITERATIONS=${GLOBAL_MAX_ITERATIONS:-1000}
GLOBAL_TOL=${GLOBAL_TOL:-1e-12}
DSTLP_SOLVING_TYPE=${DSTLP_SOLVING_TYPE:-direct}
DSTLP_DIRECT_METHOD=${DSTLP_DIRECT_METHOD:-cholesky}
DSTLP_ITERATIVE_METHOD=${DSTLP_ITERATIVE_METHOD:-cg}
DSTLP_PRECONDITIONER=${DSTLP_PRECONDITIONER:-icc}
DSTLP_MAX_ITERATIONS=${DSTLP_MAX_ITERATIONS:-1000}
DSTLP_TOL=${DSTLP_TOL:-1e-12}
MESH_REFINEMENT=${MESH_REFINEMENT:-1}
PARTITIONER=${PARTITIONER:-scotch}
TAU_MIN=${TAU_MIN:-1e-4}
TAU_MAX=${TAU_MAX:-1e-2}
NUM_TAUS=${NUM_TAUS:-10}
METHODS=${METHODS:-CN LF DSTLP}
RESUME=${RESUME:-0}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-results/F8_parallel_pulse_work_precision}
EXPERIMENT_DIR="${EXPERIMENT_ROOT}/${RUN_ID}"
MESH_ROOT="${EXPERIMENT_DIR}/meshes"
LOG_DIR="${EXPERIMENT_DIR}/logs"
TIMING_DIR="${EXPERIMENT_DIR}/timings"
SOLUTION_DIR="${EXPERIMENT_DIR}/solutions"
REFERENCE_DIR="${EXPERIMENT_DIR}/reference"
METRICS_CSV="${EXPERIMENT_DIR}/metrics_raw.csv"
MANIFEST="${EXPERIMENT_DIR}/manifest.json"

mkdir -p "${EXPERIMENT_DIR}" "${MESH_ROOT}" "${LOG_DIR}" "${TIMING_DIR}" "${SOLUTION_DIR}" "${REFERENCE_DIR}" log
cp F8_bwunicluster.md "${EXPERIMENT_DIR}/F8_bwunicluster.md" 2>/dev/null || true

MPI=(mpirun -np "${PROCS}" --bind-to core --map-by core)
TIME=(python3 parallel/time_command.py -o)

export RUN_ID PROCS REPEATS T H DEGREE LF_H LF_DEGREE LF_ELL ELL GAMMA WAVE_SPEED MIN_INNER MIN_OUTER
export PULSE_MU PULSE_S PULSE_B PULSE_AMPLITUDE_FACTOR REF_H REF_TAU REF_DEGREE REF_ELL
export GLOBAL_SOLVING_TYPE GLOBAL_DIRECT_METHOD GLOBAL_ITERATIVE_METHOD
export GLOBAL_PRECONDITIONER GLOBAL_MAX_ITERATIONS GLOBAL_TOL
export DSTLP_SOLVING_TYPE DSTLP_DIRECT_METHOD DSTLP_ITERATIVE_METHOD
export DSTLP_PRECONDITIONER DSTLP_MAX_ITERATIONS DSTLP_TOL
export MESH_REFINEMENT REF_MESH_REFINEMENT PARTITIONER TAU_MIN TAU_MAX NUM_TAUS MANIFEST METHODS RESUME

tau_values=$(python3 -c 'import numpy as np, os
tau_max=float(os.environ["TAU_MAX"])
tau_min=float(os.environ["TAU_MIN"])
T=float(os.environ["T"])
num=int(os.environ["NUM_TAUS"])
for tau in np.geomspace(tau_max, tau_min, num):
    n_steps=max(1, round(T / tau))
    print(f"{T / n_steps:.16g}")')
if [ -z "${tau_values}" ]; then
    echo "Tau grid generation produced no values; aborting."
    exit 1
fi

write_manifest() {
    python3 -c 'import json, os
data = {
    "run_id": os.environ["RUN_ID"],
    "experiment": "F8_parallel_pulse_work_precision",
    "procs": int(os.environ["PROCS"]),
    "repeats": int(os.environ["REPEATS"]),
    "T": float(os.environ["T"]),
    "h": float(os.environ["H"]),
    "degree": int(os.environ["DEGREE"]),
    "lf_h": float(os.environ["LF_H"]),
    "lf_degree": int(os.environ["LF_DEGREE"]),
    "lf_mass": "lumped",
    "ell": int(os.environ["ELL"]),
    "gamma": float(os.environ["GAMMA"]),
    "wave_speed": float(os.environ["WAVE_SPEED"]),
    "minimal_pred_ells": [int(os.environ["MIN_INNER"]), int(os.environ["MIN_OUTER"])],
    "pulse": {
        "mu": float(os.environ["PULSE_MU"]),
        "s": float(os.environ["PULSE_S"]),
        "b": float(os.environ["PULSE_B"]),
        "amplitude_factor": float(os.environ["PULSE_AMPLITUDE_FACTOR"]),
    },
    "reference": {
        "method": "CN",
        "h": float(os.environ["REF_H"]),
        "tau": float(os.environ["REF_TAU"]),
        "degree": int(os.environ["REF_DEGREE"]),
        "mass": "consistent",
    },
    "global_solver": {
        "solving_type": os.environ["GLOBAL_SOLVING_TYPE"],
        "direct_method": os.environ["GLOBAL_DIRECT_METHOD"],
        "iterative_method": os.environ["GLOBAL_ITERATIVE_METHOD"],
        "preconditioner": os.environ["GLOBAL_PRECONDITIONER"],
        "max_iterations": int(os.environ["GLOBAL_MAX_ITERATIONS"]),
        "tol": float(os.environ["GLOBAL_TOL"]),
    },
    "dstlp_solver": {
        "solving_type": os.environ["DSTLP_SOLVING_TYPE"],
        "direct_method": os.environ["DSTLP_DIRECT_METHOD"],
        "iterative_method": os.environ["DSTLP_ITERATIVE_METHOD"],
        "preconditioner": os.environ["DSTLP_PRECONDITIONER"],
        "max_iterations": int(os.environ["DSTLP_MAX_ITERATIONS"]),
        "tol": float(os.environ["DSTLP_TOL"]),
    },
    "mesh_refinement": float(os.environ["MESH_REFINEMENT"]),
    "partitioner": os.environ["PARTITIONER"],
    "tau_min": float(os.environ["TAU_MIN"]),
    "tau_max": float(os.environ["TAU_MAX"]),
    "num_taus": int(os.environ["NUM_TAUS"]),
    "tau_values": os.environ["TAU_VALUES"].split(),
    "methods": os.environ["METHODS"].split(),
    "resume": os.environ["RESUME"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
}
with open(os.environ["MANIFEST"], "w") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")'
}

export TAU_VALUES="${tau_values}"
write_manifest

copy_log_if_present() {
    src=$1
    dst=$2
    if [ -f "${src}" ]; then
        cp "${src}" "${dst}"
    fi
}

append_metrics() {
    local method=$1
    local tau=$2
    local repeat=$3
    local status=$4
    local row_ell_p1=$5
    local row_ell_p2=$6
    local row_pred_ell=$7
    local run_log=$8
    local submesh_log=$9
    local timing_file=${10}
    local row_h="${H}"
    local row_degree="${DEGREE}"
    local row_ell="${ELL}"
    if [ "${method}" = "LF" ]; then
        row_h="${LF_H}"
        row_degree="${LF_DEGREE}"
        row_ell="${LF_ELL}"
    fi
    python3 parallel/collect_metrics.py \
        --csv "${METRICS_CSV}" \
        --run-id "${RUN_ID}" \
        --method "${method}" \
        --tau "${tau}" \
        --repeat "${repeat}" \
        --status "${status}" \
        --procs "${PROCS}" \
        --h "${row_h}" \
        --degree "${row_degree}" \
        --ell "${row_ell}" \
        --ell-p1 "${row_ell_p1}" \
        --ell-p2 "${row_ell_p2}" \
        --pred-ell "${row_pred_ell}" \
        --log-file "${run_log}" \
        --submeshing-log-file "${submesh_log}" \
        --timing-file "${timing_file}"
}

has_method() {
    local requested=$1
    for method in ${METHODS}; do
        if [ "${method}" = "${requested}" ]; then
            return 0
        fi
    done
    return 1
}

row_is_done() {
    local method=$1
    local tau=$2
    local repeat=$3
    if [ "${RESUME}" != "1" ] || [ ! -f "${METRICS_CSV}" ]; then
        return 1
    fi
    python3 - "$METRICS_CSV" "$method" "$tau" "$repeat" <<'PY'
import csv
import math
import sys

csv_file, method, tau, repeat = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
with open(csv_file, newline="") as handle:
    for row in csv.DictReader(handle):
        try:
            same_tau = math.isclose(float(row["tau"]), tau, rel_tol=1e-12, abs_tol=0.0)
        except (KeyError, ValueError):
            same_tau = False
        if (
            row.get("method") == method
            and row.get("repeat") == repeat
            and same_tau
            and row.get("status") == "ok"
        ):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

run_submesh_generic() {
    local mesh_dir=$1
    local h=$2
    local degree=$3
    local ell=$4
    local pred_ell=$5
    local mesh_refinement=$6
    local time_file=$7
    mkdir -p "${mesh_dir}"
    "${TIME[@]}" "${time_file}" -- \
        "${MPI[@]}" python3 parallel/submeshing.py \
        "${h}" "${degree}" "${ell}" \
        --pred_ell "${pred_ell}" \
        --mesh-refinement "${mesh_refinement}" \
        --partitioner "${PARTITIONER}" \
        --mesh_dir "${mesh_dir}"
}

find_solution_bp() {
    local root=$1
    local filename=$2
    find "${root}" -name "${filename}" -type d -print -quit
}

append_reference_errors() {
    local method=$1
    local solution_root=$2
    local solution_name=$3
    local run_log=$4
    local candidate_degree=$5
    local candidate
    candidate=$(find_solution_bp "${solution_root}" "${solution_name}.bp")
    if [ -z "${candidate}" ]; then
        echo "Reference error failed: could not find ${solution_name}.bp below ${solution_root}" >> "${run_log}"
        return 1
    fi
    echo "=== Reference error calculation for ${method}: ${candidate} ===" >> "${run_log}"
    "${MPI[@]}" python3 parallel/pulse_reference_errors.py \
        --candidate "${candidate}" \
        --reference "${REFERENCE_BP}" \
        --candidate-degree "${candidate_degree}" \
        --reference-degree "${REF_DEGREE}" \
        --time "${T}" \
        >> "${run_log}" 2>&1
}

echo "=== F8 pulse run ${RUN_ID} on ${PROCS} MPI ranks ==="
echo "=== Tau values: ${tau_values} ==="

ref_mesh="${MESH_ROOT}/reference"
ref_sub_time="${TIMING_DIR}/submeshing_reference.time"
ref_sub_log="${LOG_DIR}/submeshing_reference.log"
ref_solution_root="${REFERENCE_DIR}/CN"
if [ "${RESUME}" = "1" ] && [ -f "${ref_mesh}/Omega.hdf5" ] && [ -f "${ref_sub_log}" ]; then
    echo "=== Reusing reference mesh ==="
else
    echo "=== Building reference mesh h=${REF_H}, degree=${REF_DEGREE} ==="
    if ! run_submesh_generic "${ref_mesh}" "${REF_H}" "${REF_DEGREE}" "${REF_ELL}" 1 "${REF_MESH_REFINEMENT}" "${ref_sub_time}" 2>&1 | tee "${LOG_DIR}/submeshing_reference.stdout"; then
        echo "Reference submeshing failed; aborting."
        exit 1
    fi
    copy_log_if_present log/submeshing.log "${ref_sub_log}"
fi

REFERENCE_BP=$(find_solution_bp "${ref_solution_root}" "solCN_ref.bp")
if [ "${RESUME}" = "1" ] && [ -n "${REFERENCE_BP}" ]; then
    echo "=== Reusing reference solution ${REFERENCE_BP} ==="
else
    echo "=== Computing CN degree-${REF_DEGREE} reference tau=${REF_TAU} ==="
    cp "${ref_sub_log}" log/submeshing.log
    ref_time="${TIMING_DIR}/CN_reference.time"
    if ! "${TIME[@]}" "${ref_time}" -- \
        "${MPI[@]}" python3 parallel/CrankNicolson.py "${REF_TAU}" "${T}" \
        --mesh_dir "${ref_mesh}" \
        --no-mass-lump \
        --solvingType "${GLOBAL_SOLVING_TYPE}" \
        --direct_method "${GLOBAL_DIRECT_METHOD}" \
        --iterative_method "${GLOBAL_ITERATIVE_METHOD}" \
        --preconditioner "${GLOBAL_PRECONDITIONER}" \
        --max_iterations "${GLOBAL_MAX_ITERATIONS}" \
        --tol "${GLOBAL_TOL}" \
        --rhs zero \
        --ic pulse \
        --pulse_mu "${PULSE_MU}" \
        --pulse_s "${PULSE_S}" \
        --pulse_b "${PULSE_B}" \
        --pulse_amplitude_factor "${PULSE_AMPLITUDE_FACTOR}" \
        --output_mode last \
        --solution_dir "${ref_solution_root}" \
        --solution_name solCN_ref \
        2>&1 | tee "${LOG_DIR}/CN_reference.stdout"; then
        echo "Reference CN run failed; aborting."
        exit 1
    fi
    copy_log_if_present log/CrankNicolson.log "${LOG_DIR}/CN_reference.log"
    REFERENCE_BP=$(find_solution_bp "${ref_solution_root}" "solCN_ref.bp")
fi
if [ -z "${REFERENCE_BP}" ]; then
    echo "Could not locate reference solution solCN_ref.bp; aborting."
    exit 1
fi
echo "=== Reference solution: ${REFERENCE_BP} ==="

probe_mesh="${MESH_ROOT}/probe"
probe_time="${TIMING_DIR}/probe_submeshing.time"
echo "=== Probe submeshing for hmin ==="
if ! run_submesh_generic "${probe_mesh}" "${H}" "${DEGREE}" "${ELL}" 1 "${MESH_REFINEMENT}" "${probe_time}" 2>&1 | tee "${LOG_DIR}/probe_submeshing.stdout"; then
    echo "Probe submeshing failed; aborting."
    exit 1
fi
copy_log_if_present log/submeshing.log "${LOG_DIR}/probe_submeshing.log"
hmin=$(python3 -c 'import re, sys
text=open(sys.argv[1], errors="replace").read()
matches=re.findall(r"h_min in global mesh:\s*([0-9.eE+-]+)", text)
if not matches:
    raise SystemExit("could not parse h_min from submeshing log")
print(matches[-1])' "${LOG_DIR}/probe_submeshing.log")
echo "=== Parsed hmin=${hmin} ==="

for tau in ${tau_values}; do
    read tau_checked heuristic ell_p1 ell_p2 pred_ell < <(
        python3 parallel/tau_layers.py \
            --tau "${tau}" \
            --hmin "${hmin}" \
            --gamma "${GAMMA}" \
            --c "${WAVE_SPEED}" \
            --min-inner "${MIN_INNER}" \
            --min-outer "${MIN_OUTER}"
    )
    tau_tag=$(python3 -c 'import sys
print(f"{float(sys.argv[1]):.6g}".replace(".", "p").replace("-", "m"))' "${tau}")
    echo "=== tau=${tau}, heuristic=${heuristic}, ell_p1=${ell_p1}, ell_p2=${ell_p2}, pred_ell=${pred_ell} ==="

    dstlp_mesh="${MESH_ROOT}/tau_${tau_tag}/dstlp_pred${pred_ell}"
    dstlp_sub_time="${TIMING_DIR}/submeshing_DSTLP_tau_${tau_tag}.time"
    dstlp_sub_log="${LOG_DIR}/submeshing_DSTLP_tau_${tau_tag}.log"
    dstlp_mesh_ready=0
    if has_method DSTLP; then
        if [ "${RESUME}" = "1" ] && [ -f "${dstlp_mesh}/Omega.hdf5" ] && [ -f "${dstlp_sub_log}" ]; then
            echo "=== Reusing DSTLP submesh for tau=${tau} ==="
            dstlp_mesh_ready=1
        elif run_submesh_generic "${dstlp_mesh}" "${H}" "${DEGREE}" "${ELL}" "${pred_ell}" "${MESH_REFINEMENT}" "${dstlp_sub_time}" 2>&1 | tee "${LOG_DIR}/submeshing_DSTLP_tau_${tau_tag}.stdout"; then
            copy_log_if_present log/submeshing.log "${dstlp_sub_log}"
            dstlp_mesh_ready=1
        else
            copy_log_if_present log/submeshing.log "${dstlp_sub_log}"
            for rep in $(seq 1 "${REPEATS}"); do
                append_metrics DSTLP "${tau}" "${rep}" "submeshing_failed" "${ell_p1}" "${ell_p2}" "${pred_ell}" "${LOG_DIR}/DSTLP_tau_${tau_tag}_rep${rep}.log" "${dstlp_sub_log}" "${TIMING_DIR}/DSTLP_tau_${tau_tag}_rep${rep}.time"
            done
        fi
    fi

    cn_mesh="${MESH_ROOT}/tau_${tau_tag}/cn_global"
    cn_sub_time="${TIMING_DIR}/submeshing_CN_tau_${tau_tag}.time"
    cn_sub_log="${LOG_DIR}/submeshing_CN_tau_${tau_tag}.log"
    cn_mesh_ready=0
    if has_method CN; then
        if [ "${RESUME}" = "1" ] && [ -f "${cn_mesh}/Omega.hdf5" ] && [ -f "${cn_sub_log}" ]; then
            echo "=== Reusing CN submesh for tau=${tau} ==="
            cn_mesh_ready=1
        elif run_submesh_generic "${cn_mesh}" "${H}" "${DEGREE}" "${ELL}" 1 "${MESH_REFINEMENT}" "${cn_sub_time}" 2>&1 | tee "${LOG_DIR}/submeshing_CN_tau_${tau_tag}.stdout"; then
            copy_log_if_present log/submeshing.log "${cn_sub_log}"
            cn_mesh_ready=1
        else
            copy_log_if_present log/submeshing.log "${cn_sub_log}"
            for rep in $(seq 1 "${REPEATS}"); do
                append_metrics CN "${tau}" "${rep}" "submeshing_failed" -1 -1 -1 "${LOG_DIR}/CN_tau_${tau_tag}_rep${rep}.log" "${cn_sub_log}" "${TIMING_DIR}/CN_tau_${tau_tag}_rep${rep}.time"
            done
        fi
    fi

    lf_mesh="${MESH_ROOT}/tau_${tau_tag}/lf_global"
    lf_sub_time="${TIMING_DIR}/submeshing_LF_tau_${tau_tag}.time"
    lf_sub_log="${LOG_DIR}/submeshing_LF_tau_${tau_tag}.log"
    lf_mesh_ready=0
    if has_method LF; then
        if [ "${RESUME}" = "1" ] && [ -f "${lf_mesh}/Omega.hdf5" ] && [ -f "${lf_sub_log}" ]; then
            echo "=== Reusing LF submesh for tau=${tau} ==="
            lf_mesh_ready=1
        elif run_submesh_generic "${lf_mesh}" "${LF_H}" "${LF_DEGREE}" "${LF_ELL}" 1 "${MESH_REFINEMENT}" "${lf_sub_time}" 2>&1 | tee "${LOG_DIR}/submeshing_LF_tau_${tau_tag}.stdout"; then
            copy_log_if_present log/submeshing.log "${lf_sub_log}"
            lf_mesh_ready=1
        else
            copy_log_if_present log/submeshing.log "${lf_sub_log}"
            for rep in $(seq 1 "${REPEATS}"); do
                append_metrics LF "${tau}" "${rep}" "submeshing_failed" -1 -1 -1 "${LOG_DIR}/LF_tau_${tau_tag}_rep${rep}.log" "${lf_sub_log}" "${TIMING_DIR}/LF_tau_${tau_tag}_rep${rep}.time"
            done
        fi
    fi

    for rep in $(seq 1 "${REPEATS}"); do
        if has_method CN && row_is_done CN "${tau}" "${rep}"; then
            echo "=== Skipping completed CN tau=${tau} rep=${rep} ==="
        elif has_method CN && [ "${cn_mesh_ready}" = "1" ]; then
            cn_time="${TIMING_DIR}/CN_tau_${tau_tag}_rep${rep}.time"
            cn_log="${LOG_DIR}/CN_tau_${tau_tag}_rep${rep}.log"
            cn_solution_root="${SOLUTION_DIR}/CN/tau_${tau_tag}/rep_${rep}"
            cp "${cn_sub_log}" log/submeshing.log
            if "${TIME[@]}" "${cn_time}" -- \
                "${MPI[@]}" python3 parallel/CrankNicolson.py "${tau}" "${T}" \
                --mesh_dir "${cn_mesh}" \
                --no-mass-lump \
                --solvingType "${GLOBAL_SOLVING_TYPE}" \
                --direct_method "${GLOBAL_DIRECT_METHOD}" \
                --iterative_method "${GLOBAL_ITERATIVE_METHOD}" \
                --preconditioner "${GLOBAL_PRECONDITIONER}" \
                --max_iterations "${GLOBAL_MAX_ITERATIONS}" \
                --tol "${GLOBAL_TOL}" \
                --rhs zero \
                --ic pulse \
                --pulse_mu "${PULSE_MU}" \
                --pulse_s "${PULSE_S}" \
                --pulse_b "${PULSE_B}" \
                --pulse_amplitude_factor "${PULSE_AMPLITUDE_FACTOR}" \
                --output_mode last \
                --solution_dir "${cn_solution_root}" \
                2>&1 | tee "${LOG_DIR}/CN_tau_${tau_tag}_rep${rep}.stdout"; then
                status=ok
            else
                status=failed
            fi
            copy_log_if_present log/CrankNicolson.log "${cn_log}"
            if [ "${status}" = "ok" ] && ! append_reference_errors CN "${cn_solution_root}" solCN "${cn_log}" "${DEGREE}"; then
                status=reference_error_failed
            fi
            append_metrics CN "${tau}" "${rep}" "${status}" -1 -1 -1 "${cn_log}" "${cn_sub_log}" "${cn_time}"
        fi

        if has_method LF && row_is_done LF "${tau}" "${rep}"; then
            echo "=== Skipping completed LF tau=${tau} rep=${rep} ==="
        elif has_method LF && [ "${lf_mesh_ready}" = "1" ]; then
            lf_time="${TIMING_DIR}/LF_tau_${tau_tag}_rep${rep}.time"
            lf_log="${LOG_DIR}/LF_tau_${tau_tag}_rep${rep}.log"
            lf_solution_root="${SOLUTION_DIR}/LF/tau_${tau_tag}/rep_${rep}"
            cp "${lf_sub_log}" log/submeshing.log
            if "${TIME[@]}" "${lf_time}" -- \
                "${MPI[@]}" python3 parallel/Leapfrog.py "${tau}" "${T}" \
                --mesh_dir "${lf_mesh}" \
                --mass-lump \
                --solvingType "${GLOBAL_SOLVING_TYPE}" \
                --direct_method "${GLOBAL_DIRECT_METHOD}" \
                --iterative_method "${GLOBAL_ITERATIVE_METHOD}" \
                --preconditioner "${GLOBAL_PRECONDITIONER}" \
                --max_iterations "${GLOBAL_MAX_ITERATIONS}" \
                --tol "${GLOBAL_TOL}" \
                --rhs zero \
                --ic pulse \
                --pulse_mu "${PULSE_MU}" \
                --pulse_s "${PULSE_S}" \
                --pulse_b "${PULSE_B}" \
                --pulse_amplitude_factor "${PULSE_AMPLITUDE_FACTOR}" \
                --output_mode last \
                --solution_dir "${lf_solution_root}" \
                2>&1 | tee "${LOG_DIR}/LF_tau_${tau_tag}_rep${rep}.stdout"; then
                status=ok
            else
                status=failed
            fi
            copy_log_if_present log/Leapfrog.log "${lf_log}"
            if [ "${status}" = "ok" ] && ! append_reference_errors LF "${lf_solution_root}" solLF "${lf_log}" "${LF_DEGREE}"; then
                status=reference_error_failed
            fi
            append_metrics LF "${tau}" "${rep}" "${status}" -1 -1 -1 "${lf_log}" "${lf_sub_log}" "${lf_time}"
        fi

        if has_method DSTLP && row_is_done DSTLP "${tau}" "${rep}"; then
            echo "=== Skipping completed DSTLP tau=${tau} rep=${rep} ==="
        elif has_method DSTLP && [ "${dstlp_mesh_ready}" = "1" ]; then
            dstlp_time="${TIMING_DIR}/DSTLP_tau_${tau_tag}_rep${rep}.time"
            dstlp_log="${LOG_DIR}/DSTLP_tau_${tau_tag}_rep${rep}.log"
            dstlp_solution_root="${SOLUTION_DIR}/DSTLP/tau_${tau_tag}/rep_${rep}"
            cp "${dstlp_sub_log}" log/submeshing.log
            if "${TIME[@]}" "${dstlp_time}" -- \
                "${MPI[@]}" python3 parallel/DSTLP.py "${tau}" "${T}" \
                --ell_p1 "${ell_p1}" \
                --ell_p2 "${ell_p2}" \
                --mesh_dir "${dstlp_mesh}" \
                --solvingType "${DSTLP_SOLVING_TYPE}" \
                --direct_method "${DSTLP_DIRECT_METHOD}" \
                --iterative_method "${DSTLP_ITERATIVE_METHOD}" \
                --preconditioner "${DSTLP_PRECONDITIONER}" \
                --max_iterations "${DSTLP_MAX_ITERATIONS}" \
                --tol "${DSTLP_TOL}" \
                --rhs zero \
                --ic pulse \
                --pulse_mu "${PULSE_MU}" \
                --pulse_s "${PULSE_S}" \
                --pulse_b "${PULSE_B}" \
                --pulse_amplitude_factor "${PULSE_AMPLITUDE_FACTOR}" \
                --output_mode last \
                --solution_dir "${dstlp_solution_root}" \
                --use_alloc_free_comms \
                --write_global_solution \
                2>&1 | tee "${LOG_DIR}/DSTLP_tau_${tau_tag}_rep${rep}.stdout"; then
                status=ok
            else
                status=failed
            fi
            copy_log_if_present log/DSTLP.log "${dstlp_log}"
            if [ "${status}" = "ok" ] && ! append_reference_errors DSTLP "${dstlp_solution_root}" solDSTLP "${dstlp_log}" "${DEGREE}"; then
                status=reference_error_failed
            fi
            append_metrics DSTLP "${tau}" "${rep}" "${status}" "${ell_p1}" "${ell_p2}" "${pred_ell}" "${dstlp_log}" "${dstlp_sub_log}" "${dstlp_time}"
        fi
    done
done

echo "=== F8 finished: ${EXPERIMENT_DIR} ==="
echo "=== Metrics: ${METRICS_CSV} ==="
