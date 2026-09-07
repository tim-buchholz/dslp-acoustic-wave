#!/usr/bin/env python3
"""Collect F7 per-rank metrics into one pgfplots-friendly CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "procs",
    "cn_time_loop_s",
    "cn_external_wall_s",
    "cn_accounted_wall_s",
    "cn_wall_step_s",
    "cn_setup_s",
    "cn_meshing_s",
    "cn_rel_xh",
    "dstlp_time_loop_s",
    "dstlp_external_wall_s",
    "dstlp_accounted_wall_s",
    "dstlp_wall_step_s",
    "dstlp_setup_s",
    "dstlp_meshing_s",
    "dstlp_rel_xh",
    "dstlp_ell_p1",
    "dstlp_ell_p2",
    "dstlp_pred_ell",
]


def read_rows(archive: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_file in sorted(archive.glob("metrics_raw_procs*.csv")):
        with csv_file.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.16g}"


def method_prefix(method: str) -> str:
    if method == "CN":
        return "cn"
    if method == "DSTLP":
        return "dstlp"
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    by_procs: dict[int, dict[str, dict[str, str]]] = {}
    for row in read_rows(args.archive):
        if row.get("status") != "ok":
            continue
        method = row.get("method", "")
        if method not in {"CN", "DSTLP"}:
            continue
        procs = int(float(row["procs"]))
        by_procs.setdefault(procs, {})[method] = row

    output_rows: list[dict[str, str]] = []
    for procs in sorted(by_procs):
        out = {field: "" for field in FIELDS}
        out["procs"] = str(procs)
        for method, row in by_procs[procs].items():
            prefix = method_prefix(method)
            meshing_key = "submeshing_total_s" if method == "DSTLP" else "global_meshing_s"
            meshing = as_float(row, meshing_key)
            setup = as_float(row, "solver_setup_s")
            time_loop = as_float(row, "time_loop_s")
            accounted = None
            if meshing is not None and setup is not None and time_loop is not None:
                accounted = meshing + setup + time_loop
            out[f"{prefix}_time_loop_s"] = fmt(time_loop)
            out[f"{prefix}_external_wall_s"] = fmt(as_float(row, "external_wall_s"))
            out[f"{prefix}_accounted_wall_s"] = fmt(accounted)
            out[f"{prefix}_wall_step_s"] = fmt(as_float(row, "wall_time_per_step_s"))
            out[f"{prefix}_setup_s"] = fmt(setup)
            out[f"{prefix}_meshing_s"] = fmt(meshing)
            out[f"{prefix}_rel_xh"] = fmt(as_float(row, "rel_xh"))
            if method == "DSTLP":
                out["dstlp_ell_p1"] = row.get("ell_p1", "")
                out["dstlp_ell_p2"] = row.get("ell_p2", "")
                out["dstlp_pred_ell"] = row.get("pred_ell", "")
        output_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
