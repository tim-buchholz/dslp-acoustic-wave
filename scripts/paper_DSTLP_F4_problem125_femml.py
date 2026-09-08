"""Run the first 1D paper figure for problem 125 with mass lumping."""

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
        description="Run problem-125 FEM-ML degree-1 CN/lf/DS/DSTLP comparison."
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run 5 tau values instead of the full 20-value series.",
    )
    parser.add_argument("--problem", type=int, default=125)
    parser.add_argument("--h", type=float, default=0.001)
    parser.add_argument("--ell", type=int, default=8)
    parser.add_argument("--tau-max", type=float, default=0.1)
    parser.add_argument("--tau-min", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min-pred-ell-p1", type=int, default=2)
    parser.add_argument("--min-pred-ell-p2", type=int, default=1)
    parser.add_argument(
        "--output-root",
        default="results/paper_DSTLP/f4_problem125_femml_deg1",
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
if ARGS.min_pred_ell_p1 < 1 or ARGS.min_pred_ell_p2 < 0:
    raise ValueError("minimal prediction ell values must be at least (1, 0)")

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

Config.FEM_DEGREE = 1

import pandas as pd

import TimeIntegration
from DomainSplitting import DomainSplitting as OldDomainSplitting
from DomainSplittingTestLocPred import DSTLP
from Experiments import (
    SpaceDiscretization,
    SplittedGrids1D,
    time_convergence,
    time_convergence_compare_no_save,
)
from Norms import combine_norms


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


def processed_results(
    cn_csv: Path,
    leapfrog_csv: Path,
    old_ds_csv: Path,
    dstlp_csv: Path,
    output_csv: Path,
) -> None:
    cn = sort_by_tau(cn_csv)
    leapfrog = sort_by_tau(leapfrog_csv)
    old_ds = sort_by_tau(old_ds_csv)
    dstlp = sort_by_tau(dstlp_csv)

    output = pd.DataFrame({"tau": cn["tau"]})
    output["CN H1L2 relative error"] = h1l2(cn)
    output["leapfrog H1L2 relative error"] = h1l2(leapfrog)
    output["DS H1L2 relative error"] = h1l2(old_ds)
    output["DS - CN H1L2 relative error"] = h1l2(old_ds, "Difference_")
    output["DSTLP H1L2 relative error"] = h1l2(dstlp)
    output["DSTLP - CN H1L2 relative error"] = h1l2(dstlp, "Difference_")
    output["O(tau^2)"] = output["tau"] ** 2
    output["old DS CFL tau estimate"] = ARGS.h
    output.to_csv(output_csv, index=False)


def run_exact_series(ti_class, parallelize: bool) -> str:
    return time_convergence(
        problem_no=ARGS.problem,
        SDclass=SpaceDiscretization.FEM_ml,
        TIclass=ti_class,
        SGclass=SplittedGrids1D.DS_1D_2SD_distorted,
        h=ARGS.h,
        ell=ARGS.ell,
        T=5.0,
        tau_max=ARGS.tau_max,
        tau_min=ARGS.tau_min,
        num_taus=5 if ARGS.pilot else 20,
        parallelize_series=parallelize,
    )


def run_against_cn(ds_class, parallelize: bool) -> str:
    return time_convergence_compare_no_save(
        problem_no=ARGS.problem,
        SDclass=SpaceDiscretization.FEM_ml,
        SGclass=SplittedGrids1D.DS_1D_2SD_distorted,
        DSclass=ds_class,
        h=ARGS.h,
        ell=ARGS.ell,
        T=5.0,
        tau_max=ARGS.tau_max,
        tau_min=ARGS.tau_min,
        num_taus=5 if ARGS.pilot else 20,
        parallelize_series=parallelize,
    )


def archive_csv(source: str, destination: Path) -> None:
    shutil.copy2(source, destination)


def main() -> None:
    mode = "pilot" if ARGS.pilot else "full"
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    run_dir = (
        Path(ARGS.output_root)
        / f"{timestamp}_problem{ARGS.problem}_FEMml_deg1_ell{ARGS.ell}_gamma{ARGS.gamma:g}_{mode}".replace(
            ".", "p"
        )
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
        "experiment": "F4",
        "problem": ARGS.problem,
        "domain": "DS_1D_2SD_distorted",
        "degree": 1,
        "mass_type": "mass_lumped",
        "h": ARGS.h,
        "ell": ARGS.ell,
        "tau_min": ARGS.tau_min,
        "tau_max": ARGS.tau_max,
        "num_taus": 5 if ARGS.pilot else 20,
        "T": 5.0,
        "old_ds_cfl_tau_estimate": ARGS.h,
        "dstlp_minimal_pred_ells": [
            ARGS.min_pred_ell_p1,
            ARGS.min_pred_ell_p2,
        ],
        "dstlp_gamma": ARGS.gamma,
        "parallelize_series": not ARGS.serial,
        "curves": [
            "CrankNicolson",
            "leapfrog",
            "DomainSplitting",
            "DomainSplitting - CrankNicolson",
            "DSTLP",
            "DSTLP - CrankNicolson",
            "O(tau^2)",
        ],
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
        cn_csv = run_exact_series(
            TimeIntegration.CrankNicolson, parallelize=not ARGS.serial
        )
        archive_csv(cn_csv, run_dir / "raw_cn.csv")

        leapfrog_csv = run_exact_series(
            TimeIntegration.leapfrog, parallelize=not ARGS.serial
        )
        archive_csv(leapfrog_csv, run_dir / "raw_leapfrog.csv")

        old_ds_csv = run_against_cn(
            OldDomainSplitting, parallelize=not ARGS.serial
        )
        archive_csv(old_ds_csv, run_dir / "raw_old_ds.csv")

        dstlp_csv = run_against_cn(PaperDSTLP, parallelize=not ARGS.serial)
        archive_csv(dstlp_csv, run_dir / "raw_dstlp.csv")

        processed_results(
            run_dir / "raw_cn.csv",
            run_dir / "raw_leapfrog.csv",
            run_dir / "raw_old_ds.csv",
            run_dir / "raw_dstlp.csv",
            run_dir / "plot.csv",
        )
    except BaseException:
        manifest["status"] = "failed"
        manifest["elapsed_s"] = time.perf_counter() - started
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        raise

    manifest["status"] = "complete"
    manifest["elapsed_s"] = time.perf_counter() - started
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Archived F4 experiment in {run_dir}")


if __name__ == "__main__":
    main()
