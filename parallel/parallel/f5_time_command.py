#!/usr/bin/env python3
"""Run a command and write wall time plus max RSS in /usr/bin/time-like format."""

from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command")

    start = time.monotonic()
    completed = subprocess.run(command)
    wall = time.monotonic() - start
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss = usage.ru_maxrss
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"wall_s={wall:.6f}\nmax_rss_kb={max_rss}\n")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
