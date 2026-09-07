#!/bin/bash
# F7 strong-scaling experiment for bwUniCluster Ice Lake nodes.

set -o pipefail

module load compiler/gnu/14.2
module load mpi/openmpi/5.0-gnu-14.2
module load devel/miniforge

conda activate dscg-env

set -u

SCRIPT_ROOT=$(pwd -P)
PROCS=${PROCS:-${SLURM_NTASKS:-64}}
RUN_ID=${RUN_ID:-$(date +%Y%m%dT%H%M%S)_F7_scaling}
REPEATS=${REPEATS:-1}
T=${T:-1.0}
TAU=${TAU:-1e-3}
H=${H:-0.002}
DEGREE=${DEGREE:-2}
ELL=${ELL:-4}
GAMMA=${GAMMA:-1.0}
WAVE_SPEED=${WAVE_SPEED:-1.0}
MIN_INNER=${MIN_INNER:-2}
MIN_OUTER=${MIN_OUTER:-1}
PULSE_MU=${PULSE_MU:-0.5}
PULSE_S=${PULSE_S:-0.2}
PULSE_B=${PULSE_B:-1.0}
PULSE_AMPLITUDE_FACTOR=${PULSE_AMPLITUDE_FACTOR:-1.0}
REF_BP=${REF_BP:-}
REF_ROOT=${REF_ROOT:-}
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
METHODS=${METHODS:-CN DSTLP}
RESUME=${RESUME:-0}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-results/F7_parallel_strong_scaling}
case "${EXPERIMENT_ROOT}" in
    /*) ;;
    *) EXPERIMENT_ROOT="${SCRIPT_ROOT}/${EXPERIMENT_ROOT}" ;;
esac
if [ -n "${REF_ROOT}" ]; then
    case "${REF_ROOT}" in
        /*) ;;
        *) REF_ROOT="${SCRIPT_ROOT}/${REF_ROOT}" ;;
    esac
fi
if [ -n "${REF_BP}" ]; then
    case "${REF_BP}" in
        /*) ;;
        *) REF_BP="${SCRIPT_ROOT}/${REF_BP}" ;;
    esac
fi
EXPERIMENT_DIR="${EXPERIMENT_ROOT}/${RUN_ID}"
RANK_TAG="procs${PROCS}"
MESH_ROOT="${EXPERIMENT_DIR}/meshes/${RANK_TAG}"
LOG_DIR="${EXPERIMENT_DIR}/logs/${RANK_TAG}"
TIMING_DIR="${EXPERIMENT_DIR}/timings/${RANK_TAG}"
SOLUTION_DIR="${EXPERIMENT_DIR}/solutions/${RANK_TAG}"
REFERENCE_DIR="${EXPERIMENT_DIR}/reference/${RANK_TAG}"
METRICS_CSV="${EXPERIMENT_DIR}/metrics_raw_${RANK_TAG}.csv"
MANIFEST="${EXPERIMENT_DIR}/manifest_${RANK_TAG}.json"
JOB_WORK_DIR="${EXPERIMENT_DIR}/work/${RANK_TAG}"

mkdir -p "${EXPERIMENT_DIR}" "${MESH_ROOT}" "${LOG_DIR}" "${TIMING_DIR}" "${SOLUTION_DIR}" "${REFERENCE_DIR}" "${JOB_WORK_DIR}"
cp "${SCRIPT_ROOT}/F7_bwunicluster.md" "${EXPERIMENT_DIR}/F7_bwunicluster.md" 2>/dev/null || true
ln -sfn "${SCRIPT_ROOT}/parallel" "${JOB_WORK_DIR}/parallel"
cd "${JOB_WORK_DIR}"
mkdir -p log

MPI=(mpirun -np "${PROCS}" --bind-to core --map-by core)
TIME=(python3 parallel/f5_time_command.py -o)

export RUN_ID PROCS REPEATS T TAU H DEGREE ELL GAMMA WAVE_SPEED MIN_INNER MIN_OUTER
export PULSE_MU PULSE_S PULSE_B PULSE_AMPLITUDE_FACTOR REF_BP REF_ROOT REF_H REF_TAU REF_DEGREE REF_ELL
export GLOBAL_SOLVING_TYPE GLOBAL_DIRECT_METHOD GLOBAL_ITERATIVE_METHOD
export GLOBAL_PRECONDITIONER GLOBAL_MAX_ITERATIONS GLOBAL_TOL
export DSTLP_SOLVING_TYPE DSTLP_DIRECT_METHOD DSTLP_ITERATIVE_METHOD
export DSTLP_PRECONDITIONER DSTLP_MAX_ITERATIONS DSTLP_TOL
export MESH_REFINEMENT REF_MESH_REFINEMENT PARTITIONER MANIFEST METHODS RESUME

TAU_FIT=$(python3 -c 'import os
T=float(os.environ["T"])
tau=float(os.environ["TAU"])
n_steps=max(1, round(T / tau))
print(f"{T / n_steps:.16g}")')
TAU="${TAU_FIT}"
export TAU

write_manifest() {
    python3 -c 'import json, os
data = {
    "run_id": os.environ["RUN_ID"],
    "experiment": "F7_parallel_strong_scaling",
    "procs": int(os.environ["PROCS"]),
    "repeats": int(os.environ["REPEATS"]),
    "T": float(os.environ["T"]),
    "tau": float(os.environ["TAU"]),
    "h": float(os.environ["H"]),
    "degree": int(os.environ["DEGREE"]),
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
        "external_bp": os.environ.get("REF_BP", ""),
        "external_root": os.environ.get("REF_ROOT", ""),
        "fallback_method": "CN",
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
    "methods": os.environ["METHODS"].split(),
    "resume": os.environ["RESUME"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
}
with open(os.environ["MANIFEST"], "w") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")'
}

copy_log_if_present() {
    src=$1
    dst=$2
    if [ -f "${src}" ]; then
        cp "${src}" "${dst}"
    fi
}

append_metrics() {
    local method=$1
    local repeat=$2
    local status=$3
    local row_ell_p1=$4
    local row_ell_p2=$5
    local row_pred_ell=$6
    local run_log=$7
    local submesh_log=$8
    local timing_file=$9
    python3 parallel/f5_collect_metrics.py \
        --csv "${METRICS_CSV}" \
        --run-id "${RUN_ID}" \
        --method "${method}" \
        --tau "${TAU}" \
        --repeat "${repeat}" \
        --status "${status}" \
        --procs "${PROCS}" \
        --h "${H}" \
        --degree "${DEGREE}" \
        --ell "${ELL}" \
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
    local repeat=$2
    if [ "${RESUME}" != "1" ] || [ ! -f "${METRICS_CSV}" ]; then
        return 1
    fi
    python3 - "$METRICS_CSV" "$method" "$repeat" <<'PY'
import csv
import sys

csv_file, method, repeat = sys.argv[1], sys.argv[2], sys.argv[3]
with open(csv_file, newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("method") == method and row.get("repeat") == repeat and row.get("status") == "ok":
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
    local candidate
    candidate=$(find_solution_bp "${solution_root}" "${solution_name}.bp")
    if [ -z "${candidate}" ]; then
        echo "Reference error failed: could not find ${solution_name}.bp below ${solution_root}" >> "${run_log}"
        return 1
    fi
    echo "=== Reference error calculation for ${method}: ${candidate} ===" >> "${run_log}"
    "${MPI[@]}" python3 parallel/f6_reference_errors.py \
        --candidate "${candidate}" \
        --reference "${REFERENCE_BP}" \
        --candidate-degree "${DEGREE}" \
        --reference-degree "${REF_DEGREE}" \
        --time "${T}" \
        >> "${run_log}" 2>&1
}

write_manifest

echo "=== F7 strong-scaling run ${RUN_ID}, ${PROCS} MPI ranks, tau=${TAU} ==="
echo "=== Methods: ${METHODS} ==="

if [ -n "${REF_BP}" ]; then
    REFERENCE_BP="${REF_BP}"
    echo "=== Using external reference ${REFERENCE_BP} ==="
elif [ -n "${REF_ROOT}" ]; then
    REFERENCE_BP=$(find_solution_bp "${REF_ROOT}" "solCN_ref.bp")
    if [ -n "${REFERENCE_BP}" ]; then
        echo "=== Using external reference found below ${REF_ROOT}: ${REFERENCE_BP} ==="
    else
        echo "Could not find solCN_ref.bp below REF_ROOT=${REF_ROOT}; aborting."
        exit 1
    fi
else
    ref_mesh="${MESH_ROOT}/reference"
    ref_sub_time="${TIMING_DIR}/submeshing_reference.time"
    ref_sub_log="${LOG_DIR}/submeshing_reference.log"
    ref_solution_root="${REFERENCE_DIR}/CN"
    if [ "${RESUME}" = "1" ] && [ -f "${ref_mesh}/Omega.hdf5" ] && [ -f "${ref_sub_log}" ]; then
        echo "=== Reusing reference mesh for ${PROCS} ranks ==="
    else
        echo "=== Building fallback reference mesh h=${REF_H}, degree=${REF_DEGREE} ==="
        if ! run_submesh_generic "${ref_mesh}" "${REF_H}" "${REF_DEGREE}" "${REF_ELL}" 1 "${REF_MESH_REFINEMENT}" "${ref_sub_time}" 2>&1 | tee "${LOG_DIR}/submeshing_reference.stdout"; then
            echo "Reference submeshing failed; aborting."
            exit 1
        fi
        copy_log_if_present log/submeshing.log "${ref_sub_log}"
    fi

    REFERENCE_BP=$(find_solution_bp "${ref_solution_root}" "solCN_ref.bp")
    if [ "${RESUME}" = "1" ] && [ -n "${REFERENCE_BP}" ]; then
        echo "=== Reusing fallback reference solution ${REFERENCE_BP} ==="
    else
        echo "=== Computing fallback CN reference tau=${REF_TAU} ==="
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
            echo "Fallback reference CN run failed; aborting."
            exit 1
        fi
        copy_log_if_present log/CrankNicolson.log "${LOG_DIR}/CN_reference.log"
        REFERENCE_BP=$(find_solution_bp "${ref_solution_root}" "solCN_ref.bp")
    fi
fi

if [ -z "${REFERENCE_BP}" ]; then
    echo "Could not locate reference solution; aborting."
    exit 1
fi

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
read tau_checked heuristic ell_p1 ell_p2 pred_ell < <(
    python3 parallel/f5_tau_layers.py \
        --tau "${TAU}" \
        --hmin "${hmin}" \
        --gamma "${GAMMA}" \
        --c "${WAVE_SPEED}" \
        --min-inner "${MIN_INNER}" \
        --min-outer "${MIN_OUTER}"
)
echo "=== hmin=${hmin}, heuristic=${heuristic}, ell_p1=${ell_p1}, ell_p2=${ell_p2}, pred_ell=${pred_ell} ==="

cn_mesh="${MESH_ROOT}/cn_global"
cn_sub_time="${TIMING_DIR}/submeshing_CN.time"
cn_sub_log="${LOG_DIR}/submeshing_CN.log"
cn_mesh_ready=0
if has_method CN; then
    if [ "${RESUME}" = "1" ] && [ -f "${cn_mesh}/Omega.hdf5" ] && [ -f "${cn_sub_log}" ]; then
        echo "=== Reusing CN mesh for ${PROCS} ranks ==="
        cn_mesh_ready=1
    elif run_submesh_generic "${cn_mesh}" "${H}" "${DEGREE}" "${ELL}" 1 "${MESH_REFINEMENT}" "${cn_sub_time}" 2>&1 | tee "${LOG_DIR}/submeshing_CN.stdout"; then
        copy_log_if_present log/submeshing.log "${cn_sub_log}"
        cn_mesh_ready=1
    else
        copy_log_if_present log/submeshing.log "${cn_sub_log}"
        for rep in $(seq 1 "${REPEATS}"); do
            append_metrics CN "${rep}" "submeshing_failed" -1 -1 -1 "${LOG_DIR}/CN_rep${rep}.log" "${cn_sub_log}" "${TIMING_DIR}/CN_rep${rep}.time"
        done
    fi
fi

dstlp_mesh="${MESH_ROOT}/dstlp_pred${pred_ell}"
dstlp_sub_time="${TIMING_DIR}/submeshing_DSTLP.time"
dstlp_sub_log="${LOG_DIR}/submeshing_DSTLP.log"
dstlp_mesh_ready=0
if has_method DSTLP; then
    if [ "${RESUME}" = "1" ] && [ -f "${dstlp_mesh}/Omega.hdf5" ] && [ -f "${dstlp_sub_log}" ]; then
        echo "=== Reusing DSTLP mesh for ${PROCS} ranks ==="
        dstlp_mesh_ready=1
    elif run_submesh_generic "${dstlp_mesh}" "${H}" "${DEGREE}" "${ELL}" "${pred_ell}" "${MESH_REFINEMENT}" "${dstlp_sub_time}" 2>&1 | tee "${LOG_DIR}/submeshing_DSTLP.stdout"; then
        copy_log_if_present log/submeshing.log "${dstlp_sub_log}"
        dstlp_mesh_ready=1
    else
        copy_log_if_present log/submeshing.log "${dstlp_sub_log}"
        for rep in $(seq 1 "${REPEATS}"); do
            append_metrics DSTLP "${rep}" "submeshing_failed" "${ell_p1}" "${ell_p2}" "${pred_ell}" "${LOG_DIR}/DSTLP_rep${rep}.log" "${dstlp_sub_log}" "${TIMING_DIR}/DSTLP_rep${rep}.time"
        done
    fi
fi

for rep in $(seq 1 "${REPEATS}"); do
    if has_method CN && row_is_done CN "${rep}"; then
        echo "=== Skipping completed CN procs=${PROCS} rep=${rep} ==="
    elif has_method CN && [ "${cn_mesh_ready}" = "1" ]; then
        cn_time="${TIMING_DIR}/CN_rep${rep}.time"
        cn_log="${LOG_DIR}/CN_rep${rep}.log"
        cn_solution_root="${SOLUTION_DIR}/CN/rep_${rep}"
        cp "${cn_sub_log}" log/submeshing.log
        if "${TIME[@]}" "${cn_time}" -- \
            "${MPI[@]}" python3 parallel/CrankNicolson.py "${TAU}" "${T}" \
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
            2>&1 | tee "${LOG_DIR}/CN_rep${rep}.stdout"; then
            status=ok
        else
            status=failed
        fi
        copy_log_if_present log/CrankNicolson.log "${cn_log}"
        if [ "${status}" = "ok" ] && ! append_reference_errors CN "${cn_solution_root}" solCN "${cn_log}"; then
            status=reference_error_failed
        fi
        append_metrics CN "${rep}" "${status}" -1 -1 -1 "${cn_log}" "${cn_sub_log}" "${cn_time}"
    fi

    if has_method DSTLP && row_is_done DSTLP "${rep}"; then
        echo "=== Skipping completed DSTLP procs=${PROCS} rep=${rep} ==="
    elif has_method DSTLP && [ "${dstlp_mesh_ready}" = "1" ]; then
        dstlp_time="${TIMING_DIR}/DSTLP_rep${rep}.time"
        dstlp_log="${LOG_DIR}/DSTLP_rep${rep}.log"
        dstlp_solution_root="${SOLUTION_DIR}/DSTLP/rep_${rep}"
        cp "${dstlp_sub_log}" log/submeshing.log
        if "${TIME[@]}" "${dstlp_time}" -- \
            "${MPI[@]}" python3 parallel/DSTLP.py "${TAU}" "${T}" \
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
            2>&1 | tee "${LOG_DIR}/DSTLP_rep${rep}.stdout"; then
            status=ok
        else
            status=failed
        fi
        copy_log_if_present log/DSTLP.log "${dstlp_log}"
        if [ "${status}" = "ok" ] && ! append_reference_errors DSTLP "${dstlp_solution_root}" solDSTLP "${dstlp_log}"; then
            status=reference_error_failed
        fi
        append_metrics DSTLP "${rep}" "${status}" "${ell_p1}" "${ell_p2}" "${pred_ell}" "${dstlp_log}" "${dstlp_sub_log}" "${dstlp_time}"
    fi
done

echo "=== F7 finished: ${EXPERIMENT_DIR} (${RANK_TAG}) ==="
echo "=== Metrics: ${METRICS_CSV} ==="
