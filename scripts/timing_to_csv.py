#!/usr/bin/env python3
"""
timing_to_csv.py -- collect the per-model reporter.py reports of one run into a
CSV: one row per model, and per variant the min / median / max time in ms.

    model,label,mlir_llvm_median_ms,mlir_llvm_min_ms,mlir_llvm_max_ms,...

Usage:
    python scripts/timing_to_csv.py                       # newest run under --root
    python scripts/timing_to_csv.py --run-id 20260904_055133
    python scripts/timing_to_csv.py 01_MLP/01_MLP_*.json -o mlp.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REFERENCE_VARIANT = "mlir_llvm"

NS_PER_MS = 1e6

STATS = ("median", "min", "max")

# Short names for the `label` column.
SHORT_NAMES = {
    "GoogleNetInceptionModule": "Inception",
    "SqueezeNetFireModule": "FireModule",
}

RUN_ID_RE = re.compile(r"_(\d{8}_\d{6})\.json$")


def discover_reports(root, run_id=None):
    """All model reports for one run: the given run_id, or the newest one."""
    candidates = {}
    for path in sorted(root.glob("*/*_*.json")):
        match = RUN_ID_RE.search(path.name)
        if match:
            candidates.setdefault(match.group(1), []).append(path)
    if not candidates:
        raise SystemExit(f"no timing reports found under {root}")

    if run_id is None:
        run_id = max(candidates)
    elif run_id not in candidates:
        raise SystemExit(
            f"run '{run_id}' not found (available: {', '.join(sorted(candidates))})"
        )
    return run_id, candidates[run_id]


def load_report(path):
    """Return (model_name, {variant: {"median": ms, "min": ms, "max": ms}})."""
    with open(path, "r") as f:
        data = json.load(f)

    model_name = data.get("model_name") or data.get("model_dir") or Path(path).stem
    timing_summary = data.get("timing_summary")
    if not timing_summary:
        raise ValueError(f"{path}: no 'timing_summary' field found")

    timings = {}
    for variant, stats in timing_summary.items():
        if stats.get("median_ns") is None:
            continue
        timings[variant] = {
            stat: stats[f"{stat}_ns"] / NS_PER_MS
            for stat in STATS
            if stats.get(f"{stat}_ns") is not None
        }

    if REFERENCE_VARIANT not in timings:
        print(
            f"Warning: {path}: reference variant '{REFERENCE_VARIANT}' missing, "
            "it cannot be normalized to the reference",
            file=sys.stderr,
        )

    return model_name, timings


def collect_variants(rows):
    """Every variant seen, reference first, then alphabetically."""
    seen = set()
    for _, timings in rows:
        seen.update(timings)
    return sorted(seen, key=lambda v: (v != REFERENCE_VARIANT, v))


def write_csv(rows, variants, output_path, precision=3):
    columns = [f"{v}_{stat}_ms" for v in variants for stat in STATS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "label"] + columns)
        for model_name, timings in rows:
            values = []
            for variant in variants:
                stats = timings.get(variant, {})
                for stat in STATS:
                    value = stats.get(stat)
                    # "nan" marks a missing value
                    values.append("nan" if value is None else f"{value:.{precision}f}")
            writer.writerow(
                [model_name, SHORT_NAMES.get(model_name, model_name)] + values
            )
    return columns


def main():
    parser = argparse.ArgumentParser(
        description="Write a CSV of min/median/max runtimes (in ms) per model "
        "and variant from reporter.py's reports."
    )
    parser.add_argument(
        "reports",
        nargs="*",
        help="Timing report JSON files (default: newest run under --root)",
    )
    parser.add_argument(
        "--root",
        default=".",
        type=Path,
        help="Directory holding the model directories (default: .)",
    )
    parser.add_argument(
        "--run-id",
        help="Use this run id (e.g. 20260904_055133) instead of the newest one",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path("timing.csv"),
        type=Path,
        help="Output CSV path (default: timing.csv)",
    )
    args = parser.parse_args()

    if args.reports:
        paths = [Path(p) for p in args.reports]
        run_id = None
    else:
        run_id, paths = discover_reports(args.root, args.run_id)

    rows = []
    for path in paths:
        try:
            rows.append(load_report(path))
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            sys.exit(1)

    variants = collect_variants(rows)
    write_csv(rows, variants, args.output)

    print(f"Wrote {args.output} ({len(rows)} models, run {run_id or 'from arguments'})")
    print(f"Variants: {', '.join(variants)}")


if __name__ == "__main__":
    main()
