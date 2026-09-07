"""Run problem 125 with standard FEM degrees 1-4 for the DSTLP paper."""

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run problem-125 standard-FEM DSTLP comparisons for degrees 1-4."
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run 5 tau values instead of the full 20-value series.",
    )
    parser.add_argument("--problem", type=int, default=125)
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--h", type=float, default=0.001)
    parser.add_argument("--ell", type=int, default=8)
    parser.add_argument("--tau-max", type=float, default=0.1)
    parser.add_argument("--tau-min", type=float, default=0.0001)
    parser.add_argument("--num-taus", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min-pred-ell-p1", type=int, default=2)
    parser.add_argument("--min-pred-ell-p2", type=int, default=1)
    parser.add_argument("--reference-scale", type=float, default=1000.0)
    parser.add_argument(
        "--output-root",
        default="results/paper_DSTLP/f2_problem125_fem_degrees",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Disable process-level parallelism for easier debugging.",
    )
    return parser.parse_args()


ARGS = parse_args()
if ARGS.tau_min <= 0 or ARGS.tau_max <= 0:
    raise ValueError("--tau-min and --tau-max must be positive")
if ARGS.tau_min >= ARGS.tau_max:
    raise ValueError("--tau-min must be smaller than --tau-max")
if ARGS.num_taus < 2:
    raise ValueError("--num-taus must be at least 2")
if ARGS.min_pred_ell_p1 < 1 or ARGS.min_pred_ell_p2 < 0:
    raise ValueError("minimal prediction ell values must be at least (1, 0)")
if any(degree < 1 for degree in ARGS.degrees):
    raise ValueError("--degrees must contain positive integers")

sys.path.extend([".", "..", "../src", "src"])

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import Config

import pandas as pd

import TimeIntegration
from DomainSplittingTestLocPred import DSTLP
from Experiments import (
    SpaceDiscretization,
    SplittedGrids1D,
    time_convergence,
    time_convergence_compare_no_save,
)
from Norms import combine_norms


def set_fem_degree(degree: int) -> None:
    """Propagate degree changes to modules that import Config values by copy."""
    Config.FEM_DEGREE = degree
    TimeIntegration.FEM_DEGREE = degree
    SplittedGrids1D.FEM_DEGREE = degree


class PaperDSTLP(DSTLP):
    def __init__(self, T, tau, Omega, SpaceDiscretizationClass, problem):
        super().__init__(
            T,
            tau,
            Omega,
            SpaceDiscretizationClass,
            problem,
            minimal_pred_ells=(ARGS.min_pred_ell_p1, ARGS.min_pred_ell_p2),
            gamma=ARGS.gamma,
        )

    def __repr__(self) -> str:
        return super().__repr__().replace(
            "TimeIntegrator=PaperDSTLP", "TimeIntegrator=DSTLP"
        )


def git_revision(path: str = ".") -> str:
    result = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_dirty(path: str = ".") -> bool | None:
    result = subprocess.run(
        ["git", "-C", path, "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def h1l2(data: pd.DataFrame, prefix: str = ""):
    rel_col = prefix + "relative H1L2 error"
    if rel_col in data.columns:
        return data[rel_col].to_numpy()

    rel_h1 = prefix + "relative H1 error q"
    rel_l2p = prefix + "relative L2 error p"
    if rel_h1 in data.columns and rel_l2p in data.columns:
        return combine_norms(data[rel_h1].to_numpy(), data[rel_l2p].to_numpy())

    return combine_norms(
        data[prefix + "H1 error q"].to_numpy(),
        data[prefix + "L2 error p"].to_numpy(),
    )


def sort_by_tau(csv_file: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_file)
    data.sort_values("tau", ascending=False, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def run_leapfrog(parallelize: bool, num_taus: int) -> str:
    return time_convergence(
        problem_no=ARGS.problem,
        SDclass=SpaceDiscretization.FEM,
        TIclass=TimeIntegration.leapfrog,
        SGclass=SplittedGrids1D.DS_1D_2SD_distorted,
        h=ARGS.h,
        ell=ARGS.ell,
        T=5.0,
        tau_max=ARGS.tau_max,
        tau_min=ARGS.tau_min,
        num_taus=num_taus,
        parallelize_series=parallelize,
    )


def run_dstlp_against_cn(parallelize: bool, num_taus: int) -> str:
    return time_convergence_compare_no_save(
        problem_no=ARGS.problem,
        SDclass=SpaceDiscretization.FEM,
        SGclass=SplittedGrids1D.DS_1D_2SD_distorted,
        DSclass=PaperDSTLP,
        h=ARGS.h,
        ell=ARGS.ell,
        T=5.0,
        tau_max=ARGS.tau_max,
        tau_min=ARGS.tau_min,
        num_taus=num_taus,
        parallelize_series=parallelize,
    )


def processed_results(leapfrog_csv: Path, dstlp_csv: Path, output_csv: Path) -> None:
    leapfrog = sort_by_tau(leapfrog_csv)
    dstlp = sort_by_tau(dstlp_csv)

    output = pd.DataFrame({"tau": dstlp["tau"]})
    output["CN H1L2 relative error"] = h1l2(dstlp, "Reference_")
    output["leapfrog H1L2 relative error"] = h1l2(leapfrog)
    output["DSTLP H1L2 relative error"] = h1l2(dstlp)
    output["DSTLP - CN H1L2 relative error"] = h1l2(dstlp, "Difference_")
    output["O(tau^2)"] = ARGS.reference_scale * output["tau"] ** 2
    output.to_csv(output_csv, index=False)


def archive_csv(source: str, destination: Path) -> None:
    shutil.copy2(source, destination)


def assert_distinct_degree_outputs(outputs: dict[str, dict[str, str]]) -> None:
    seen: dict[str, str] = {}
    for degree, files in outputs.items():
        with open(files["plot"], "rb") as csv_file:
            import hashlib

            digest = hashlib.sha256(csv_file.read()).hexdigest()
        if digest in seen:
            raise RuntimeError(
                "F2 degree outputs are identical for "
                f"degrees {seen[digest]} and {degree}; check FEM_DEGREE propagation."
            )
        seen[digest] = degree


def main() -> None:
    mode = "pilot" if ARGS.pilot else "full"
    num_taus = 5 if ARGS.pilot else ARGS.num_taus
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    degrees_tag = "-".join(str(degree) for degree in ARGS.degrees)
    run_dir = (
        Path(ARGS.output_root)
        / (
            f"{timestamp}_problem{ARGS.problem}_FEM_deg{degrees_tag}"
            f"_ell{ARGS.ell}_gamma{ARGS.gamma:g}_{mode}"
        ).replace(".", "p")
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "date": timestamp,
        "status": "running",
        "git_commit": git_revision(),
        "git_dirty": git_dirty(),
        "machine": socket.gethostname(),
        "python": sys.executable,
        "command": " ".join(sys.argv),
        "experiment": "F2",
        "problem": ARGS.problem,
        "domain": "DS_1D_2SD_distorted",
        "degrees": ARGS.degrees,
        "mass_type": "consistent",
        "h": ARGS.h,
        "ell": ARGS.ell,
        "tau_min": ARGS.tau_min,
        "tau_max": ARGS.tau_max,
        "num_taus": num_taus,
        "T": 5.0,
        "dstlp_minimal_pred_ells": [
            ARGS.min_pred_ell_p1,
            ARGS.min_pred_ell_p2,
        ],
        "dstlp_gamma": ARGS.gamma,
        "reference_scale": ARGS.reference_scale,
        "parallelize_series": not ARGS.serial,
        "curves": [
            "CrankNicolson",
            "leapfrog",
            "DSTLP",
            "DSTLP - CrankNicolson",
            "reference_scale * tau^2",
        ],
        "degree_outputs": {},
        "thread_limits": {
            variable: os.environ[variable]
            for variable in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    manifest_file = run_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

    started = time.perf_counter()
    try:
        for degree in ARGS.degrees:
            print(f"=== Running F2 degree {degree} ===")
            set_fem_degree(degree)
            degree_dir = run_dir / f"deg{degree}"
            degree_dir.mkdir()

            leapfrog_csv = run_leapfrog(not ARGS.serial, num_taus)
            archive_csv(leapfrog_csv, degree_dir / "raw_leapfrog.csv")

            dstlp_csv = run_dstlp_against_cn(not ARGS.serial, num_taus)
            archive_csv(dstlp_csv, degree_dir / "raw_dstlp.csv")

            processed_results(
                degree_dir / "raw_leapfrog.csv",
                degree_dir / "raw_dstlp.csv",
                degree_dir / f"plot_deg{degree}.csv",
            )
            manifest["degree_outputs"][str(degree)] = {
                "raw_leapfrog": str(degree_dir / "raw_leapfrog.csv"),
                "raw_dstlp": str(degree_dir / "raw_dstlp.csv"),
                "plot": str(degree_dir / f"plot_deg{degree}.csv"),
            }
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

        assert_distinct_degree_outputs(manifest["degree_outputs"])
    except BaseException:
        manifest["status"] = "failed"
        manifest["elapsed_s"] = time.perf_counter() - started
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        raise

    manifest["status"] = "complete"
    manifest["elapsed_s"] = time.perf_counter() - started
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Archived F2 experiment in {run_dir}")


if __name__ == "__main__":
    main()
