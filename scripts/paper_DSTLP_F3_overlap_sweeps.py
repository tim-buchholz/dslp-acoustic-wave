"""Run problem 125 DSTLP-CN overlap sweeps for the DSTLP paper."""

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
        description="Run problem-125 FEM degree-2 DSTLP-CN overlap parameter sweeps."
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run 5 tau values instead of the full 20-value series.",
    )
    parser.add_argument("--problem", type=int, default=125)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--h", type=float, default=0.001)
    parser.add_argument("--tau-max", type=float, default=0.1)
    parser.add_argument("--tau-min", type=float, default=0.0001)
    parser.add_argument("--num-taus", type=int, default=20)
    parser.add_argument("--ell-sweep", type=int, nargs="+", default=[1, 2, 4, 8, 16, 20])
    parser.add_argument("--gamma-sweep", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument(
        "--min-pred-sweep",
        nargs="+",
        default=["1,0", "1,1", "2,1", "2,2", "3,1", "3,2", "3,3"],
        help="Minimal prediction ell pairs as p1,p2.",
    )
    parser.add_argument(
        "--output-root",
        default="results/paper_DSTLP/f3_problem125_overlap_sweeps",
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
if ARGS.degree < 1:
    raise ValueError("--degree must be positive")
if any(ell < 1 for ell in ARGS.ell_sweep):
    raise ValueError("--ell-sweep must contain positive integers")
if any(gamma <= 0 for gamma in ARGS.gamma_sweep):
    raise ValueError("--gamma-sweep must contain positive values")


def parse_min_pred_pairs(values: list[str]) -> list[tuple[int, int]]:
    pairs = []
    for value in values:
        try:
            p1_raw, p2_raw = value.split(",", maxsplit=1)
            pair = (int(p1_raw), int(p2_raw))
        except ValueError as exc:
            raise ValueError(
                f"Invalid min-pred pair {value!r}; expected p1,p2"
            ) from exc
        if pair[0] < 1 or pair[1] < 0:
            raise ValueError("minimal prediction ell values must be at least (1, 0)")
        pairs.append(pair)
    return pairs


MIN_PRED_SWEEP = parse_min_pred_pairs(ARGS.min_pred_sweep)
CURRENT_GAMMA = 1.0
CURRENT_MIN_PRED_ELLS = (2, 1)

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

import pandas as pd

import TimeIntegration
from DomainSplittingTestLocPred import DSTLP
from Experiments import (
    SpaceDiscretization,
    SplittedGrids1D,
    time_convergence_compare_no_save,
)
from Norms import combine_norms

TimeIntegration.FEM_DEGREE = ARGS.degree
SplittedGrids1D.FEM_DEGREE = ARGS.degree


class PaperDSTLP(DSTLP):
    def __init__(self, T, tau, Omega, SpaceDiscretizationClass, problem):
        super().__init__(
            T,
            tau,
            Omega,
            SpaceDiscretizationClass,
            problem,
            minimal_pred_ells=CURRENT_MIN_PRED_ELLS,
            gamma=CURRENT_GAMMA,
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


def run_dstlp_against_cn(
    *,
    ell: int,
    gamma: float,
    minimal_pred_ells: tuple[int, int],
    num_taus: int,
    parallelize: bool,
) -> str:
    global CURRENT_GAMMA, CURRENT_MIN_PRED_ELLS
    CURRENT_GAMMA = gamma
    CURRENT_MIN_PRED_ELLS = minimal_pred_ells
    return time_convergence_compare_no_save(
        problem_no=ARGS.problem,
        SDclass=SpaceDiscretization.FEM,
        SGclass=SplittedGrids1D.DS_1D_2SD_distorted,
        DSclass=PaperDSTLP,
        h=ARGS.h,
        ell=ell,
        T=5.0,
        tau_max=ARGS.tau_max,
        tau_min=ARGS.tau_min,
        num_taus=num_taus,
        parallelize_series=parallelize,
    )


def archive_variant(
    *,
    run_dir: Path,
    sweep: str,
    tag: str,
    ell: int,
    gamma: float,
    minimal_pred_ells: tuple[int, int],
    num_taus: int,
    parallelize: bool,
) -> tuple[Path, pd.DataFrame]:
    variant_dir = run_dir / sweep / tag
    variant_dir.mkdir(parents=True)
    raw_csv = run_dstlp_against_cn(
        ell=ell,
        gamma=gamma,
        minimal_pred_ells=minimal_pred_ells,
        num_taus=num_taus,
        parallelize=parallelize,
    )
    archived_raw = variant_dir / "raw_dstlp_vs_cn.csv"
    shutil.copy2(raw_csv, archived_raw)

    data = sort_by_tau(archived_raw)
    processed = pd.DataFrame({"tau": data["tau"]})
    processed["DSTLP - CN H1L2 relative error"] = h1l2(data, "Difference_")
    processed["ell"] = ell
    processed["gamma"] = gamma
    processed["min_pred_ell_p1"] = minimal_pred_ells[0]
    processed["min_pred_ell_p2"] = minimal_pred_ells[1]
    processed_file = variant_dir / "plot.csv"
    processed.to_csv(processed_file, index=False)
    return processed_file, processed


def add_curve(wide: pd.DataFrame | None, data: pd.DataFrame, label: str) -> pd.DataFrame:
    curve = data[["tau", "DSTLP - CN H1L2 relative error"]].rename(
        columns={"DSTLP - CN H1L2 relative error": label}
    )
    if wide is None:
        return curve
    return wide.merge(curve, on="tau", how="outer")


def write_wide_csv(curves: list[tuple[str, pd.DataFrame]], output_csv: Path) -> None:
    wide = None
    for label, data in curves:
        wide = add_curve(wide, data, label)
    assert wide is not None
    wide.sort_values("tau", ascending=False, inplace=True)
    wide.to_csv(output_csv, index=False)


def main() -> None:
    mode = "pilot" if ARGS.pilot else "full"
    num_taus = 5 if ARGS.pilot else ARGS.num_taus
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    run_dir = (
        Path(ARGS.output_root)
        / (
            f"{timestamp}_problem{ARGS.problem}_FEM_deg{ARGS.degree}"
            f"_overlap_sweeps_{mode}"
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
        "experiment": "F3",
        "problem": ARGS.problem,
        "domain": "DS_1D_2SD_distorted",
        "degree": ARGS.degree,
        "mass_type": "consistent",
        "h": ARGS.h,
        "tau_min": ARGS.tau_min,
        "tau_max": ARGS.tau_max,
        "num_taus": num_taus,
        "T": 5.0,
        "ell_sweep": ARGS.ell_sweep,
        "gamma_sweep": ARGS.gamma_sweep,
        "min_pred_sweep": [list(pair) for pair in MIN_PRED_SWEEP],
        "curves": ["DSTLP - CrankNicolson"],
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
    parallelize = not ARGS.serial
    try:
        ell_curves = []
        for ell in ARGS.ell_sweep:
            print(f"=== F3 ell sweep: ell={ell}, gamma=1, min_pred=(2,1) ===")
            plot_file, data = archive_variant(
                run_dir=run_dir,
                sweep="ell_sweep",
                tag=f"ell{ell}",
                ell=ell,
                gamma=1.0,
                minimal_pred_ells=(2, 1),
                num_taus=num_taus,
                parallelize=parallelize,
            )
            label = f"ell={ell}"
            ell_curves.append((label, data))
            manifest["outputs"][f"ell_sweep/{label}"] = str(plot_file)
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        write_wide_csv(ell_curves, run_dir / "plot_ell_sweep.csv")

        gamma_curves = []
        for gamma in ARGS.gamma_sweep:
            print(f"=== F3 gamma sweep: ell=4, gamma={gamma:g}, min_pred=(2,1) ===")
            tag = f"gamma{gamma:g}".replace(".", "p")
            plot_file, data = archive_variant(
                run_dir=run_dir,
                sweep="gamma_sweep",
                tag=tag,
                ell=4,
                gamma=gamma,
                minimal_pred_ells=(2, 1),
                num_taus=num_taus,
                parallelize=parallelize,
            )
            label = f"gamma={gamma:g}"
            gamma_curves.append((label, data))
            manifest["outputs"][f"gamma_sweep/{label}"] = str(plot_file)
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        write_wide_csv(gamma_curves, run_dir / "plot_gamma_sweep.csv")

        min_pred_curves = []
        for pair in MIN_PRED_SWEEP:
            print(f"=== F3 min-pred sweep: ell=4, gamma=1, min_pred={pair} ===")
            tag = f"minpred{pair[0]}-{pair[1]}"
            plot_file, data = archive_variant(
                run_dir=run_dir,
                sweep="min_pred_sweep",
                tag=tag,
                ell=4,
                gamma=1.0,
                minimal_pred_ells=pair,
                num_taus=num_taus,
                parallelize=parallelize,
            )
            label = f"minpred={pair[0]}-{pair[1]}"
            min_pred_curves.append((label, data))
            manifest["outputs"][f"min_pred_sweep/{label}"] = str(plot_file)
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        write_wide_csv(min_pred_curves, run_dir / "plot_min_pred_sweep.csv")
    except BaseException:
        manifest["status"] = "failed"
        manifest["elapsed_s"] = time.perf_counter() - started
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        raise

    manifest["status"] = "complete"
    manifest["elapsed_s"] = time.perf_counter() - started
    manifest["outputs"]["plot_ell_sweep"] = str(run_dir / "plot_ell_sweep.csv")
    manifest["outputs"]["plot_gamma_sweep"] = str(run_dir / "plot_gamma_sweep.csv")
    manifest["outputs"]["plot_min_pred_sweep"] = str(run_dir / "plot_min_pred_sweep.csv")
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Archived F3 experiment in {run_dir}")


if __name__ == "__main__":
    main()
