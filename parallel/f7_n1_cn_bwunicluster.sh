#!/bin/bash
# F7 one-rank CN workaround: submesh with 2 MPI ranks, run CN with 1 rank.

set -o pipefail

module load compiler/gnu/14.2
module load mpi/openmpi/5.0-gnu-14.2
module load devel/miniforge

conda activate dscg-env

set -u

SCRIPT_ROOT=$(pwd -P)
PROCS=1
SUBMESH_PROCS=${SUBMESH_PROCS:-2}
RUN_ID=${RUN_ID:-$(date +%Y%m%dT%H%M%S)_F7_scaling}
REPEATS=${REPEATS:-1}
T=${T:-1.0}
TAU=${TAU:-1e-3}
H=${H:-0.002}
DEGREE=${DEGREE:-2}
ELL=${ELL:-4}
PULSE_MU=${PULSE_MU:-0.5}
PULSE_S=${PULSE_S:-0.2}
PULSE_B=${PULSE_B:-1.0}
PULSE_AMPLITUDE_FACTOR=${PULSE_AMPLITUDE_FACTOR:-1.0}
REF_BP=${REF_BP:-}
REF_ROOT=${REF_ROOT:-}
REF_DEGREE=${REF_DEGREE:-2}
GLOBAL_SOLVING_TYPE=${GLOBAL_SOLVING_TYPE:-direct}
GLOBAL_DIRECT_METHOD=${GLOBAL_DIRECT_METHOD:-cholesky}
GLOBAL_ITERATIVE_METHOD=${GLOBAL_ITERATIVE_METHOD:-cg}
GLOBAL_PRECONDITIONER=${GLOBAL_PRECONDITIONER:-icc}
GLOBAL_MAX_ITERATIONS=${GLOBAL_MAX_ITERATIONS:-1000}
GLOBAL_TOL=${GLOBAL_TOL:-1e-12}
MESH_REFINEMENT=${MESH_REFINEMENT:-1}
PARTITIONER=${PARTITIONER:-scotch}
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
RANK_TAG="procs1"
MESH_ROOT="${EXPERIMENT_DIR}/meshes/${RANK_TAG}"
LOG_DIR="${EXPERIMENT_DIR}/logs/${RANK_TAG}"
TIMING_DIR="${EXPERIMENT_DIR}/timings/${RANK_TAG}"
SOLUTION_DIR="${EXPERIMENT_DIR}/solutions/${RANK_TAG}"
METRICS_CSV="${EXPERIMENT_DIR}/metrics_raw_${RANK_TAG}.csv"
MANIFEST="${EXPERIMENT_DIR}/manifest_${RANK_TAG}_cn_n1.json"
JOB_WORK_DIR="${EXPERIMENT_DIR}/work/${RANK_TAG}_cn_n1"

mkdir -p "${EXPERIMENT_DIR}" "${MESH_ROOT}" "${LOG_DIR}" "${TIMING_DIR}" "${SOLUTION_DIR}" "${JOB_WORK_DIR}"
cp "${SCRIPT_ROOT}/F7_bwunicluster.md" "${EXPERIMENT_DIR}/F7_bwunicluster.md" 2>/dev/null || true
ln -sfn "${SCRIPT_ROOT}/parallel" "${JOB_WORK_DIR}/parallel"
cd "${JOB_WORK_DIR}"
mkdir -p log

MPI_RUN=(mpirun -np "${PROCS}" --bind-to core --map-by core)
MPI_SUBMESH=(mpirun -np "${SUBMESH_PROCS}" --bind-to core --map-by core)
TIME=(python3 parallel/f5_time_command.py -o)

TAU_FIT=$(python3 -c 'import os
T=float(os.environ.get("T", "1.0"))
tau=float(os.environ.get("TAU", "1e-3"))
n_steps=max(1, round(T / tau))
print(f"{T / n_steps:.16g}")')
TAU="${TAU_FIT}"

find_solution_bp() {
    local root=$1
    local filename=$2
    find "${root}" -name "${filename}" -type d -print -quit
}

copy_log_if_present() {
    local src=$1
    local dst=$2
    if [ -f "${src}" ]; then
        cp "${src}" "${dst}"
    fi
}

row_is_done() {
    local repeat=$1
    if [ "${RESUME}" != "1" ] || [ ! -f "${METRICS_CSV}" ]; then
        return 1
    fi
    python3 - "$METRICS_CSV" "$repeat" <<'PY'
import csv
import sys

csv_file, repeat = sys.argv[1], sys.argv[2]
with open(csv_file, newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("method") == "CN" and row.get("repeat") == repeat and row.get("status") == "ok":
            raise SystemExit(0)
raise SystemExit(1)
PY
}

append_metrics() {
    local repeat=$1
    local status=$2
    local run_log=$3
    local submesh_log=$4
    local timing_file=$5
    python3 parallel/f5_collect_metrics.py \
        --csv "${METRICS_CSV}" \
        --run-id "${RUN_ID}" \
        --method CN \
        --tau "${TAU}" \
        --repeat "${repeat}" \
        --status "${status}" \
        --procs "${PROCS}" \
        --h "${H}" \
        --degree "${DEGREE}" \
        --ell "${ELL}" \
        --ell-p1 -1 \
        --ell-p2 -1 \
        --pred-ell -1 \
        --log-file "${run_log}" \
        --submeshing-log-file "${submesh_log}" \
        --timing-file "${timing_file}"
}

append_reference_errors() {
    local solution_root=$1
    local run_log=$2
    local candidate
    candidate=$(find_solution_bp "${solution_root}" "solCN.bp")
    if [ -z "${candidate}" ]; then
        echo "Reference error failed: could not find solCN.bp below ${solution_root}" >> "${run_log}"
        return 1
    fi
    echo "=== Reference error calculation for CN: ${candidate} ===" >> "${run_log}"
    "${MPI_RUN[@]}" python3 parallel/f6_reference_errors.py \
        --candidate "${candidate}" \
        --reference "${REFERENCE_BP}" \
        --candidate-degree "${DEGREE}" \
        --reference-degree "${REF_DEGREE}" \
        --time "${T}" \
        >> "${run_log}" 2>&1
}

write_manifest() {
    python3 -c 'import json, os
data = {
    "run_id": os.environ["RUN_ID"],
    "experiment": "F7_parallel_strong_scaling_CN_n1_workaround",
    "procs": 1,
    "submesh_procs": int(os.environ["SUBMESH_PROCS"]),
    "repeats": int(os.environ["REPEATS"]),
    "T": float(os.environ["T"]),
    "tau": float(os.environ["TAU"]),
    "h": float(os.environ["H"]),
    "degree": int(os.environ["DEGREE"]),
    "ell": int(os.environ["ELL"]),
    "reference": {
        "external_bp": os.environ.get("REF_BP", ""),
        "external_root": os.environ.get("REF_ROOT", ""),
        "degree": int(os.environ["REF_DEGREE"]),
    },
    "solver": {
        "solving_type": os.environ["GLOBAL_SOLVING_TYPE"],
        "direct_method": os.environ["GLOBAL_DIRECT_METHOD"],
        "iterative_method": os.environ["GLOBAL_ITERATIVE_METHOD"],
        "preconditioner": os.environ["GLOBAL_PRECONDITIONER"],
        "max_iterations": int(os.environ["GLOBAL_MAX_ITERATIONS"]),
        "tol": float(os.environ["GLOBAL_TOL"]),
    },
    "partitioner": os.environ["PARTITIONER"],
    "mesh_refinement": float(os.environ["MESH_REFINEMENT"]),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
}
with open(os.environ["MANIFEST"], "w") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")'
}

export RUN_ID SUBMESH_PROCS REPEATS T TAU H DEGREE ELL REF_BP REF_ROOT REF_DEGREE
export GLOBAL_SOLVING_TYPE GLOBAL_DIRECT_METHOD GLOBAL_ITERATIVE_METHOD
export GLOBAL_PRECONDITIONER GLOBAL_MAX_ITERATIONS GLOBAL_TOL
export PARTITIONER MESH_REFINEMENT MANIFEST
write_manifest

if [ -n "${REF_BP}" ]; then
    REFERENCE_BP="${REF_BP}"
elif [ -n "${REF_ROOT}" ]; then
    REFERENCE_BP=$(find_solution_bp "${REF_ROOT}" "solCN_ref.bp")
else
    echo "Set REF_ROOT or REF_BP for the F6 reference solution; aborting."
    exit 1
fi
if [ -z "${REFERENCE_BP}" ]; then
    echo "Could not locate reference solution; aborting."
    exit 1
fi

echo "=== F7 CN n=1 workaround, RUN_ID=${RUN_ID}, submesh ranks=${SUBMESH_PROCS}, CN ranks=1 ==="
echo "=== Reference solution: ${REFERENCE_BP} ==="

cn_mesh="${MESH_ROOT}/cn_global_from_${SUBMESH_PROCS}"
cn_sub_time="${TIMING_DIR}/submeshing_CN_from_${SUBMESH_PROCS}.time"
cn_sub_log="${LOG_DIR}/submeshing_CN_from_${SUBMESH_PROCS}.log"
cn_mesh_ready=0

if [ "${RESUME}" = "1" ] && [ -f "${cn_mesh}/Omega.hdf5" ] && [ -f "${cn_sub_log}" ]; then
    echo "=== Reusing CN mesh created with ${SUBMESH_PROCS} ranks ==="
    cn_mesh_ready=1
elif "${TIME[@]}" "${cn_sub_time}" -- \
    "${MPI_SUBMESH[@]}" python3 parallel/submeshing.py \
    "${H}" "${DEGREE}" "${ELL}" \
    --pred_ell 1 \
    --mesh-refinement "${MESH_REFINEMENT}" \
    --partitioner "${PARTITIONER}" \
    --mesh_dir "${cn_mesh}" \
    2>&1 | tee "${LOG_DIR}/submeshing_CN_from_${SUBMESH_PROCS}.stdout"; then
    copy_log_if_present log/submeshing.log "${cn_sub_log}"
    cn_mesh_ready=1
else
    copy_log_if_present log/submeshing.log "${cn_sub_log}"
    for rep in $(seq 1 "${REPEATS}"); do
        append_metrics "${rep}" "submeshing_failed" "${LOG_DIR}/CN_rep${rep}.log" "${cn_sub_log}" "${TIMING_DIR}/CN_rep${rep}.time"
    done
fi

for rep in $(seq 1 "${REPEATS}"); do
    if row_is_done "${rep}"; then
        echo "=== Skipping completed CN procs=1 rep=${rep} ==="
        continue
    fi
    if [ "${cn_mesh_ready}" != "1" ]; then
        continue
    fi

    cn_time="${TIMING_DIR}/CN_rep${rep}.time"
    cn_log="${LOG_DIR}/CN_rep${rep}.log"
    cn_solution_root="${SOLUTION_DIR}/CN/rep_${rep}"
    cp "${cn_sub_log}" log/submeshing.log
    if "${TIME[@]}" "${cn_time}" -- \
        "${MPI_RUN[@]}" python3 parallel/CrankNicolson.py "${TAU}" "${T}" \
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
    if [ "${status}" = "ok" ] && ! append_reference_errors "${cn_solution_root}" "${cn_log}"; then
        status=reference_error_failed
    fi
    append_metrics "${rep}" "${status}" "${cn_log}" "${cn_sub_log}" "${cn_time}"
done

echo "=== F7 CN n=1 workaround finished: ${EXPERIMENT_DIR} ==="
echo "=== Metrics: ${METRICS_CSV} ==="
