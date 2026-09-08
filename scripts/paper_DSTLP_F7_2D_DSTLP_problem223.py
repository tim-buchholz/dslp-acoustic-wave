"""Run problem 223 2D DSTLP/CN/leapfrog comparison for the DSTLP paper."""

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
        description="Run problem-223 2D FEM degree-2 CN/lf/DSTLP comparison."
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run 3 tau values instead of the full configured series.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--problem", type=int, default=223)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--h", type=float, default=0.005)
    parser.add_argument("--ells", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min-pred-ell-p1", type=int, default=2)
    parser.add_argument("--min-pred-ell-p2", type=int, default=1)
    parser.add_argument("--tau-min", type=float, default=0.0001)
    parser.add_argument("--tau-max", type=float, default=0.1)
    parser.add_argument("--num-taus", type=int, default=20)
    parser.add_argument("--reference-scale", type=float, default=100.0)
    parser.add_argument(
        "--output-root",
        default="results/paper_DSTLP/f7_2d_dstlp_problem223",
    )
    return parser.parse_args()


ARGS = parse_args()
if ARGS.batch_size < 1:
    raise ValueError("--batch-size must be at least 1")
if ARGS.tau_min <= 0 or ARGS.tau_max <= 0:
    raise ValueError("--tau-min and --tau-max must be positive")
if ARGS.tau_min >= ARGS.tau_max:
    raise ValueError("--tau-min must be smaller than --tau-max")
if ARGS.num_taus < 2:
    raise ValueError("--num-taus must be at least 2")
if ARGS.degree < 1:
    raise ValueError("--degree must be positive")
if any(ell < 1 for ell in ARGS.ells):
    raise ValueError("--ells must contain positive integers")
if ARGS.gamma <= 0:
    raise ValueError("--gamma must be positive")
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

Config.FEM_DEGREE = ARGS.degree
Config.DEFAULT_SOLVING_TYP = Config.DIRECT

import pandas as pd

import TimeIntegration
from DomainSplittingTestLocPred import DSTLP
from Experiments import (
    SpaceDiscretization,
    SplittedGrids2D,
    time_convergence_batchwise,
)
from Norms import combine_norms

TimeIntegration.FEM_DEGREE = ARGS.degree
SplittedGrids2D.FEM_DEGREE = ARGS.degree


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


def run_exact_curve(TIclass, ell: int, num_taus: int) -> str:
    return time_convergence_batchwise(
        problem_no=ARGS.problem,
        SDclass=SpaceDiscretization.FEM,
        TIclass=TIclass,
        SGclass=SplittedGrids2D.DS_2D_16SD_cross_SQUARE,
        h=ARGS.h,
        ell=ell,
        T=1.0,
        tau_max=ARGS.tau_max,
        tau_min=ARGS.tau_min,
        num_taus=num_taus,
        batchsize=ARGS.batch_size,
        sidewise_againstCN=False,
    )


def run_dstlp_against_cn(ell: int, num_taus: int) -> str:
    return time_convergence_batchwise(
        problem_no=ARGS.problem,
        SDclass=SpaceDiscretization.FEM,
        TIclass=PaperDSTLP,
        SGclass=SplittedGrids2D.DS_2D_16SD_cross_SQUARE,
        h=ARGS.h,
        ell=ell,
        T=1.0,
        tau_max=ARGS.tau_max,
        tau_min=ARGS.tau_min,
        num_taus=num_taus,
        batchsize=ARGS.batch_size,
        sidewise_againstCN=True,
    )


def archive_csv(source: str, destination: Path) -> None:
    shutil.copy2(source, destination)


def processed_results(run_dir: Path, cn_csv: Path, leapfrog_csv: Path) -> None:
    cn = sort_by_tau(cn_csv)
    leapfrog = sort_by_tau(leapfrog_csv)
    output = pd.DataFrame({"tau": cn["tau"]})
    output["CN H1L2 relative error"] = h1l2(cn)
    output["leapfrog H1L2 relative error"] = h1l2(leapfrog)

    for ell in ARGS.ells:
        dstlp = sort_by_tau(run_dir / f"ell{ell}" / "raw_dstlp_vs_cn.csv")
        output[f"DSTLP ell={ell} H1L2 relative error"] = h1l2(dstlp)
        output[f"DSTLP ell={ell} - CN H1L2 relative error"] = h1l2(
            dstlp, "Difference_"
        )

    output["O(tau^2)"] = ARGS.reference_scale * output["tau"] ** 2
    output.to_csv(run_dir / "plot.csv", index=False)


def main() -> None:
    mode = "pilot" if ARGS.pilot else "full"
    num_taus = 3 if ARGS.pilot else ARGS.num_taus
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    ells_tag = "-".join(str(ell) for ell in ARGS.ells)
    run_dir = (
        Path(ARGS.output_root)
        / (
            f"{timestamp}_problem{ARGS.problem}_FEM_deg{ARGS.degree}"
            f"_h{ARGS.h:g}_ell{ells_tag}_gamma{ARGS.gamma:g}_{mode}"
        ).replace(".", "p")
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    reference_ell = max(ARGS.ells)
    manifest = {
        "date": timestamp,
        "status": "running",
        "git_commit": git_revision(),
        "git_dirty": git_dirty(),
        "parallel_commit": git_revision("parallel"),
        "machine": socket.gethostname(),
        "python": sys.executable,
        "command": " ".join(sys.argv),
        "experiment": "F7",
        "problem": ARGS.problem,
        "domain": "DS_2D_4x4SD_cross_SQUARE",
        "degree": ARGS.degree,
        "mass_type": "consistent",
        "h": ARGS.h,
        "tau_min": ARGS.tau_min,
        "tau_max": ARGS.tau_max,
        "num_taus": num_taus,
        "T": 1.0,
        "ells": ARGS.ells,
        "dstlp_gamma": ARGS.gamma,
        "dstlp_minimal_pred_ells": [
            ARGS.min_pred_ell_p1,
            ARGS.min_pred_ell_p2,
        ],
        "dstlp_prediction_layers": "inner=max(min_pred_ell_p1, ceil(gamma*tau*c/hmin)); outer=max(min_pred_ell_p2, ceil(gamma*tau*c/hmin))",
        "reference_ell_for_cn_lf": reference_ell,
        "batch_size": ARGS.batch_size,
        "solver": Config.DEFAULT_SOLVING_TYP,
        "reference_scale": ARGS.reference_scale,
        "curves": [
            "CrankNicolson",
            "leapfrog",
            "DSTLP ell=4",
            "DSTLP ell=4 - CrankNicolson",
            "DSTLP ell=8",
            "DSTLP ell=8 - CrankNicolson",
            "reference_scale * tau^2",
        ],
        "outputs": {},
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
        cn_csv = run_exact_curve(TimeIntegration.CrankNicolson, reference_ell, num_taus)
        archived_cn = run_dir / "raw_cn.csv"
        archive_csv(cn_csv, archived_cn)
        manifest["outputs"]["raw_cn"] = str(archived_cn)
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

        leapfrog_csv = run_exact_curve(TimeIntegration.leapfrog, reference_ell, num_taus)
        archived_lf = run_dir / "raw_leapfrog.csv"
        archive_csv(leapfrog_csv, archived_lf)
        manifest["outputs"]["raw_leapfrog"] = str(archived_lf)
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

        for ell in ARGS.ells:
            print(f"=== Running F7 DSTLP ell={ell} against CN ===")
            ell_dir = run_dir / f"ell{ell}"
            ell_dir.mkdir()
            dstlp_csv = run_dstlp_against_cn(ell, num_taus)
            archived_dstlp = ell_dir / "raw_dstlp_vs_cn.csv"
            archive_csv(dstlp_csv, archived_dstlp)
            manifest["outputs"][f"ell{ell}/raw_dstlp_vs_cn"] = str(archived_dstlp)
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

        processed_results(run_dir, archived_cn, archived_lf)
    except BaseException:
        manifest["status"] = "failed"
        manifest["elapsed_s"] = time.perf_counter() - started
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        raise

    manifest["status"] = "complete"
    manifest["elapsed_s"] = time.perf_counter() - started
    manifest["outputs"]["plot"] = str(run_dir / "plot.csv")
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Archived F7 experiment in {run_dir}")


if __name__ == "__main__":
    main()
