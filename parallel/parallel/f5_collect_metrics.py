#!/usr/bin/env python3
"""Append one F5 work-precision metrics row parsed from run logs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


FIELDS = [
    "run_id",
    "method",
    "tau",
    "repeat",
    "status",
    "procs",
    "h",
    "degree",
    "ell",
    "ell_p1",
    "ell_p2",
    "pred_ell",
    "num_steps",
    "submeshing_total_s",
    "global_meshing_s",
    "solver_setup_s",
    "time_loop_s",
    "wall_time_per_step_s",
    "external_wall_s",
    "max_rss_kb",
    "rel_l2_u",
    "rel_h1_u",
    "rel_l2_v",
    "rel_xh",
    "log_file",
    "submeshing_log_file",
    "timing_file",
]


PATTERNS = {
    "num_steps": re.compile(r">>> time steps:\s*([0-9]+)\s*<<<"),
    "solver_setup_s": re.compile(r"System solver setup time:\s*([0-9.eE+-]+)"),
    "time_loop_s": re.compile(r"Time loop finished after time:\s*([0-9.eE+-]+)"),
    "wall_time_per_step_s": re.compile(
        r">>> wall time per time step:\s*([0-9.eE+-]+)\s*seconds\s*<<<"
    ),
    "rel_l2_u": re.compile(r"rel L2 error of un on Omega:\s*([0-9.eE+-]+)"),
    "rel_h1_u": re.compile(r"rel H1 error of un on Omega:\s*([0-9.eE+-]+)"),
    "rel_l2_v": re.compile(r"rel L2 error of vn on Omega:\s*([0-9.eE+-]+)"),
    "rel_xh": re.compile(r"rel Xh error:\s*([0-9.eE+-]+)"),
}

SUBMESH_PATTERNS = {
    "submeshing_total_s": re.compile(
        r"Total time of submeshing script\s*([0-9.eE+-]+)\s*seconds"
    ),
    "global_meshing_s": re.compile(
        r"Total time for global meshing and partitioning:\s*([0-9.eE+-]+)"
    ),
}

TIME_PATTERNS = {
    "external_wall_s": re.compile(r"wall_s=([0-9.eE+-]+)"),
    "max_rss_kb": re.compile(r"max_rss_kb=([0-9.eE+-]+)"),
}


def read_text(path: Path) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(errors="replace")


def parse_patterns(text: str, patterns: dict[str, re.Pattern[str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, pattern in patterns.items():
        matches = pattern.findall(text)
        if matches:
            values[key] = str(matches[-1])
    return values


def append_row(csv_file: Path, row: dict[str, str]) -> None:
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_file.exists()
    with csv_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True, choices=["CN", "LF", "DSTLP"])
    parser.add_argument("--tau", required=True, type=float)
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--status", required=True)
    parser.add_argument("--procs", required=True, type=int)
    parser.add_argument("--h", required=True, type=float)
    parser.add_argument("--degree", required=True, type=int)
    parser.add_argument("--ell", required=True, type=int)
    parser.add_argument("--ell-p1", type=int, default=-1)
    parser.add_argument("--ell-p2", type=int, default=-1)
    parser.add_argument("--pred-ell", type=int, default=-1)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--submeshing-log-file", required=True, type=Path)
    parser.add_argument("--timing-file", required=True, type=Path)
    args = parser.parse_args()

    row = {
        "run_id": args.run_id,
        "method": args.method,
        "tau": f"{args.tau:.16g}",
        "repeat": str(args.repeat),
        "status": args.status,
        "procs": str(args.procs),
        "h": f"{args.h:.16g}",
        "degree": str(args.degree),
        "ell": str(args.ell),
        "ell_p1": "" if args.ell_p1 < 0 else str(args.ell_p1),
        "ell_p2": "" if args.ell_p2 < 0 else str(args.ell_p2),
        "pred_ell": "" if args.pred_ell < 0 else str(args.pred_ell),
        "log_file": str(args.log_file),
        "submeshing_log_file": str(args.submeshing_log_file),
        "timing_file": str(args.timing_file),
    }
    row.update(parse_patterns(read_text(args.log_file), PATTERNS))
    row.update(parse_patterns(read_text(args.submeshing_log_file), SUBMESH_PATTERNS))
    row.update(parse_patterns(read_text(args.timing_file), TIME_PATTERNS))

    if not row.get("wall_time_per_step_s") and row.get("time_loop_s") and row.get("num_steps"):
        time_loop = float(row["time_loop_s"])
        num_steps = float(row["num_steps"])
        if math.isfinite(time_loop) and num_steps > 0:
            row["wall_time_per_step_s"] = f"{time_loop / num_steps:.16g}"

    append_row(args.csv, row)


if __name__ == "__main__":
    main()
