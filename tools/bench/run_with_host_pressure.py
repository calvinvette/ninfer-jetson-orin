#!/usr/bin/env python3
"""Run one command while recording—and bounding—Linux host-memory pressure.

This is intentionally external to the Engine: it protects capacity experiments
whose process can otherwise consume the host's physically unified DRAM before
the benchmark has an opportunity to report an error.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields = value.split()
        if fields:
            values[key] = int(fields[0]) * 1024
    return values


def positive_gib(text: str) -> int:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return int(value * (1 << 30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSON telemetry output path")
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--min-available-gib", type=positive_gib, default=2 * (1 << 30),
                        help="terminate when MemAvailable falls below this value (default: 2)")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="command to run; place -- before the command")
    args = parser.parse_args()
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")

    started = time.time()
    process = subprocess.Popen(args.command, start_new_session=True)
    samples: list[dict[str, int | float]] = []
    pressure_abort = False
    try:
        while process.poll() is None:
            current = meminfo()
            samples.append({
                "time_seconds": time.time() - started,
                "mem_available_bytes": current.get("MemAvailable", 0),
                "mem_free_bytes": current.get("MemFree", 0),
                "swap_free_bytes": current.get("SwapFree", 0),
            })
            if samples[-1]["mem_available_bytes"] < args.min_available_gib:
                pressure_abort = True
                os.killpg(process.pid, signal.SIGTERM)
                break
            time.sleep(args.interval_ms / 1000)
        return_code = process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    final = meminfo()
    report = {
        "command": args.command,
        "started_unix_seconds": started,
        "duration_seconds": time.time() - started,
        "return_code": return_code,
        "pressure_abort": pressure_abort,
        "min_available_bytes": args.min_available_gib,
        "final_mem_available_bytes": final.get("MemAvailable", 0),
        "final_swap_free_bytes": final.get("SwapFree", 0),
        "samples": samples,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    if pressure_abort:
        print("host-memory floor reached; benchmark was terminated", file=sys.stderr)
        return 124
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
