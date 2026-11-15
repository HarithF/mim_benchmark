#!/usr/bin/env python3
"""
geomean.py -- the geometric means of one report run.

Run time (timing.csv): per model, the median of each variant divided by the
median of the reference variant; then the geometric mean of those ratios over
every model that has a median for the reference and for every variant compared
with it (the rule the paper's figure applies). Groups:

  CPU  mim_llvm, mim_mlir_llvm  over  mlir_llvm
  GPU  mim_nvptx                over  iree_cuda

A group is reported when timing.csv has its reference and at least one of its
variants, so TARGET=cpu, gpu and both all work.

Compile time (compile_times.csv): per path, the geometric mean in seconds of
each column, over the models that have that column for every path of the group.

    python3 scripts/geomean.py work/reports/<run_id>
    python3 scripts/geomean.py work/reports/<run_id> --models models.txt

prints the headline numbers and writes the per-model ratios and all compile-time
columns to <run_id>/summary.md (-o to change). --models restricts both parts to
the models a file lists (KernelBench stems, one per line, as in models.txt).
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

GROUPS = (
    ("CPU", "mlir_llvm", ("mim_llvm", "mim_mlir_llvm")),
    ("GPU", "iree_cuda", ("mim_nvptx",)),
)
COMPILE_COLUMNS = ("trace_s", "frontend_s", "to_llvm_ir_s", "opt_s", "llc_s",
                   "link_s", "total_s", "end_to_end_s")
# The compile-time column printed to the console; the first one with data wins
# (iree_cuda has no LLVM IR step, so the GPU group falls back to total_s).
HEADLINE = ("to_llvm_ir_s", "total_s")


def model_key(name):
    """`MLP` (timing.csv), `01_MLP` (compile_times.csv) and `1_MLP`
    (models.txt) all name one model."""
    return re.sub(r"^\d+_", "", name.strip())


def positive(s):
    """The value as a float, or None when it is missing, nan or not positive."""
    try:
        x = float(s)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def n_models(n):
    return f"{n} model" + "s" * (n != 1)


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def read_rows(path, keep):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if keep is None or model_key(r["model"]) in keep]
        return reader.fieldnames or [], rows


def runtime(path, keep, md, console):
    fields, rows = read_rows(path, keep)
    for group, ref, variants in GROUPS:
        variants = [v for v in variants if f"{v}_median_ms" in fields]
        if f"{ref}_median_ms" not in fields or not variants:
            continue
        used, skipped = [], []
        for r in rows:
            meds = {v: positive(r[f"{v}_median_ms"]) for v in [ref] + variants}
            if all(meds.values()):
                used.append((r["model"], {v: meds[v] / meds[ref] for v in variants}))
            else:
                skipped.append(r["model"])
        if not used:
            console.append(f"{group} run time over {ref}: no model has every variant")
            continue
        gm = {v: geomean([q[v] for _, q in used]) for v in variants}
        md.append(f"## {group} run time relative to `{ref}` ({n_models(len(used))})\n")
        md.append("| model | " + " | ".join(f"`{v}`" for v in variants) + " |")
        md.append("|---|" + "---:|" * len(variants))
        for model, q in used:
            md.append(f"| {model} | " + " | ".join(f"{q[v]:.2f}" for v in variants) + " |")
        md.append("| **geometric mean** | "
                  + " | ".join(f"**{gm[v]:.2f}**" for v in variants) + " |")
        md.append("| faster than the reference | "
                  + " | ".join(f"{sum(q[v] < 1 for _, q in used)} of {len(used)}"
                               for v in variants) + " |\n")
        if skipped:
            md.append(f"Left out, a median is missing: {', '.join(skipped)}\n")
        console.append(f"{group} run time over {ref} ({n_models(len(used))}): "
                       + ", ".join(f"{v} {gm[v]:.2f}x" for v in variants))


def compile_time(path, keep, md, console):
    _, rows = read_rows(path, keep)
    data = {}                                  # model -> path -> row
    for r in rows:
        data.setdefault(model_key(r["model"]), {})[r["path"]] = r
    present = {p for paths in data.values() for p in paths}
    for group, ref, variants in GROUPS:
        paths = [p for p in variants + (ref,) if p in present]
        if not paths:
            continue
        gms = {}                               # column -> (models, {path: seconds})
        for col in COMPILE_COLUMNS:
            models = [m for m, by_path in data.items()
                      if all(p in by_path and positive(by_path[p].get(col)) for p in paths)]
            if models:
                gms[col] = (len(models), {p: geomean([positive(data[m][p][col]) for m in models])
                                          for p in paths})
        if not gms:
            continue
        md.append(f"## {group} compile time, geometric mean in seconds\n")
        md.append("| column | models | " + " | ".join(f"`{p}`" for p in paths) + " |")
        md.append("|---|---:|" + "---:|" * len(paths))
        for col, (n, gm) in gms.items():
            md.append(f"| `{col}` | {n} | " + " | ".join(f"{gm[p]:.2f}" for p in paths) + " |")
        md.append("\nColumns as in `compile_times.md`; each is averaged over the models that"
                  " have it for every path.\n")
        col = next((c for c in HEADLINE if c in gms), None)
        if col:
            n, gm = gms[col]
            console.append(f"{group} compile time, {col} ({n_models(n)}): "
                           + ", ".join(f"{p} {gm[p]:.2f} s" for p in paths))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path,
                    help="a report run: work/reports/<run_id> (timing.csv, compile_times.csv)")
    ap.add_argument("--models", type=Path,
                    help="only the models this file lists (KernelBench stems, as in models.txt)")
    ap.add_argument("-o", "--output", type=Path,
                    help="where to write the summary (default: <run_dir>/summary.md)")
    args = ap.parse_args()

    keep = None
    if args.models:
        keep = {model_key(l) for l in args.models.read_text().splitlines()
                if l.strip() and not l.lstrip().startswith("#")}

    timing = args.run_dir / "timing.csv"
    compile_csv = args.run_dir / "compile_times.csv"
    md = [f"# Geometric means: {args.run_dir.name}\n"]
    if keep is not None:
        md.append(f"Restricted to the {n_models(len(keep))} of `{args.models}`.\n")
    console = []
    found = False
    for path, fn in ((timing, runtime), (compile_csv, compile_time)):
        if path.is_file():
            found = True
            fn(path, keep, md, console)
        else:
            console.append(f"{path.name} not found")
    if not found:
        sys.exit(f"geomean.py: neither timing.csv nor compile_times.csv in {args.run_dir}")

    if keep is not None:
        seen = set()
        for path in (timing, compile_csv):
            if path.is_file():
                seen |= {model_key(r["model"]) for r in read_rows(path, keep)[1]}
        if keep - seen:
            md.append(f"Listed but not in this run: {', '.join(sorted(keep - seen))}\n")

    out = args.output or args.run_dir / "summary.md"
    out.write_text("\n".join(md))
    print(f"  geometric means: {out}")
    for line in console:
        print(f"    {line}")


if __name__ == "__main__":
    main()
