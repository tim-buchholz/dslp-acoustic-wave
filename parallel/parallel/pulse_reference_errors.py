#!/usr/bin/env python3
"""Compute pulse-run errors against an ADIOS reference solution."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import adios4dolfinx
import dolfinx as dfx
from mpi4py import MPI

from Norms import error_norm_ref, norm_H10, norm_L2


def read_function(filename: Path, degree: int, name: str, time: float, comm: MPI.Comm):
    mesh = adios4dolfinx.read_mesh(filename, comm)
    V = dfx.fem.functionspace(mesh, ("Lagrange", degree))
    function = dfx.fem.Function(V)
    adios4dolfinx.read_function(filename, function, time=time, name=name)
    return function


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate-degree", required=True, type=int)
    parser.add_argument("--reference-degree", required=True, type=int)
    parser.add_argument("--time", required=True, type=float)
    parser.add_argument("--degree-raise", type=int, default=1)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    uh = read_function(args.candidate, args.candidate_degree, "u", args.time, comm)
    vh = read_function(args.candidate, args.candidate_degree, "v", args.time, comm)
    u_ref = read_function(args.reference, args.reference_degree, "u", args.time, comm)
    v_ref = read_function(args.reference, args.reference_degree, "v", args.time, comm)

    error_l2_u_abs = error_norm_ref(uh, u_ref, "L2", degree_raise=args.degree_raise)
    error_h1_u_abs = error_norm_ref(uh, u_ref, "H10", degree_raise=args.degree_raise)
    error_l2_v_abs = error_norm_ref(vh, v_ref, "L2", degree_raise=args.degree_raise)

    norm_l2_u_ref = norm_L2(u_ref.function_space, u_ref)
    norm_h1_u_ref = norm_H10(u_ref.function_space, u_ref)
    norm_l2_v_ref = norm_L2(v_ref.function_space, v_ref)

    rel_l2_u = error_l2_u_abs / norm_l2_u_ref
    rel_h1_u = error_h1_u_abs / norm_h1_u_ref
    rel_l2_v = error_l2_v_abs / norm_l2_v_ref
    error_xh_abs = math.sqrt(error_h1_u_abs**2 + error_l2_v_abs**2)
    norm_xh_ref = math.sqrt(norm_h1_u_ref**2 + norm_l2_v_ref**2)
    rel_xh = error_xh_abs / norm_xh_ref

    if comm.rank == 0:
        print(f"Reference solution file: {args.reference}")
        print(f"Candidate solution file: {args.candidate}")
        print(f"L2 norm of un reference:{norm_l2_u_ref}")
        print(f"H1 norm of un reference:{norm_h1_u_ref}")
        print(f"L2 norm of vn reference:{norm_l2_v_ref}")
        print(f"Xh norm of reference:{norm_xh_ref}")
        print(f"rel L2 error of un on Omega: {rel_l2_u}")
        print(f"rel H1 error of un on Omega: {rel_h1_u}")
        print(f"rel L2 error of vn on Omega: {rel_l2_v}")
        print(f"rel Xh error: {rel_xh}")


if __name__ == "__main__":
    main()
