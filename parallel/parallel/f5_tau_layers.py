#!/usr/bin/env python3
"""Print tau, heuristic layer count, inner layers, outer layers, total layers."""

from __future__ import annotations

import argparse
import math


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--hmin", type=float, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--min-inner", type=int, default=2)
    parser.add_argument("--min-outer", type=int, default=1)
    args = parser.parse_args()

    heuristic = math.ceil(args.gamma * args.tau * args.c / args.hmin)
    inner = max(args.min_inner, heuristic)
    outer = max(args.min_outer, heuristic)
    print(f"{args.tau:.16g} {heuristic:d} {inner:d} {outer:d} {inner + outer:d}")


if __name__ == "__main__":
    main()
