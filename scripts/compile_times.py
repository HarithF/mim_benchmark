#!/usr/bin/env python3
"""
compile_times.py -- one table of compile times per model, path and step.

Columns (seconds; empty when a step does not apply or was not recorded):

  trace_s          tracing the PyTorch model into an FX graph: Dynamo for the
                   MimIR paths, torch.export for torch-mlir. Not part of
                   to_llvm_ir_s; end_to_end_s includes it.
  frontend_s       graph -> LLVM IR (mim_llvm: MimIR decompose + translate +
                   optimize) or -> MLIR (mim_mlir_llvm: MimIR's MLIR emit;
                   mlir_llvm: torch-mlir's decompositions, import and lowering
                   to linalg)
  mlir_opt_s       lower.sh: the mlir-opt pipeline (MLIR paths)
  libm_s           lower.sh: math ops -> libm calls, when needed
  mlir_translate_s lower.sh: MLIR -> LLVM IR
  to_llvm_ir_s     frontend + lowering: PyTorch graph -> LLVM IR
  opt_s, llc_s     compile_all.sh: opt -O3, llc
  link_s           compile_all.sh: clang compiling the driver and linking
  total_s          to_llvm_ir_s + opt_s + llc_s + link_s
  end_to_end_s     trace_s + total_s

Sources: <mimir_out>/<k>/*_manifest.json (frontend), <mimir_out>/results/<k>.json
(Dynamo tracing: dynamo_trace, mlir_dynamo_trace), <torch_out>/<k>.timing.json
(export_mlir.py), <models>/<NN_Model>/<variant>/*_{lower,compile}_timing.json.

    python3 scripts/compile_times.py --mimir-out work/mimir --torch-out work/torch_mlir \\
        --models work/models --out work/reports/compile_times
writes compile_times.csv and compile_times.md.
"""

import argparse
import csv
import glob
import json
import os

PATHS = ("mim_llvm", "mim_mlir_llvm", "mlir_llvm")
COLUMNS = ("trace_s", "frontend_s", "mlir_opt_s", "libm_s", "mlir_translate_s",
           "to_llvm_ir_s", "opt_s", "llc_s", "link_s", "total_s", "end_to_end_s")


def load(pattern):
    """The JSON file matching `pattern`, or {} if there is none."""
    hits = sorted(glob.glob(pattern))
    if not hits:
        return {}
    with open(hits[0]) as f:
        return json.load(f)


def manifest_timings(mimir_dir, mlir):
    suffix = "_mlir_manifest.json" if mlir else "_manifest.json"
    for path in sorted(glob.glob(os.path.join(mimir_dir, "*" + suffix))):
        if not mlir and path.endswith("_mlir_manifest.json"):
            continue
        with open(path) as f:
            return json.load(f).get("timings") or {}
    return {}


def runner_timings(mimir_out, stem):
    """Timings the frontend runner recorded for one model (results/<k>.json)."""
    r = load(os.path.join(mimir_out, "results", f"{stem}.json"))
    r = r.get("results", r) if isinstance(r, dict) else r
    return (r[0].get("timings") or {}) if r else {}


def add(*values):
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def rows_for(model_dir, mimir_out, torch_out):
    name = os.path.basename(model_dir)                        # 01_MLP
    stem = f"{int(name.split('_', 1)[0])}_{name.split('_', 1)[1]}"  # 1_MLP
    mimir_dir = os.path.join(mimir_out, stem)
    traced = runner_timings(mimir_out, stem)
    rows = []
    for path in PATHS:
        vdir = os.path.join(model_dir, path)
        if not os.path.isdir(vdir):
            continue
        r = dict.fromkeys(COLUMNS)
        if path == "mim_llvm":
            t = manifest_timings(mimir_dir, mlir=False)
            if t:
                r["frontend_s"] = add(t.get("decompose"), t.get("translate"), t.get("optimize"))
            r["trace_s"] = traced.get("dynamo_trace")
        elif path == "mim_mlir_llvm":
            t = manifest_timings(mimir_dir, mlir=True)
            if t:
                r["frontend_s"] = add(t.get("decompose"), t.get("mlir"))
            r["trace_s"] = traced.get("mlir_dynamo_trace")
        else:
            t = load(os.path.join(torch_out, f"{stem}.timing.json"))
            r["trace_s"] = t.get("torch_export_s")
            r["frontend_s"] = t.get("to_linalg_s")
        lower = load(os.path.join(vdir, "*_lower_timing.json"))
        r["mlir_opt_s"] = lower.get("mlir_opt_s")
        r["libm_s"] = lower.get("libm_s")
        r["mlir_translate_s"] = lower.get("mlir_translate_s")
        r["to_llvm_ir_s"] = add(r["frontend_s"], r["mlir_opt_s"], r["libm_s"], r["mlir_translate_s"])
        comp = load(os.path.join(vdir, "*_compile_timing.json"))
        r["opt_s"], r["llc_s"], r["link_s"] = comp.get("opt_s"), comp.get("llc_s"), comp.get("link_s")
        r["total_s"] = add(r["to_llvm_ir_s"], r["opt_s"], r["llc_s"], r["link_s"])
        r["end_to_end_s"] = add(r["trace_s"], r["total_s"])
        rows.append((name, path, r))
    return rows


def fmt(v):
    return "" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimir-out", required=True)
    ap.add_argument("--torch-out", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True, help="output prefix (.csv and .md are appended)")
    args = ap.parse_args()

    rows = []
    for model_dir in sorted(glob.glob(os.path.join(args.models, "[0-9]*_*"))):
        if os.path.isdir(model_dir):
            rows.extend(rows_for(model_dir, args.mimir_out, args.torch_out))
    if not rows:
        raise SystemExit(f"no models under {args.models}")

    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("model", "path") + COLUMNS)
        for name, path, r in rows:
            w.writerow([name, path] + [fmt(r[c]) for c in COLUMNS])

    with open(args.out + ".md", "w") as f:
        f.write("| model | path | " + " | ".join(COLUMNS) + " |\n")
        f.write("|---|---|" + "---:|" * len(COLUMNS) + "\n")
        for name, path, r in rows:
            f.write(f"| {name} | {path} | " + " | ".join(fmt(r[c]) for c in COLUMNS) + " |\n")
        f.write("\nSums over all models:\n\n| path | " + " | ".join(COLUMNS) + " |\n")
        f.write("|---|" + "---:|" * len(COLUMNS) + "\n")
        for path in PATHS:
            sel = [r for _, p, r in rows if p == path]
            if sel:
                f.write(f"| {path} | " + " | ".join(fmt(add(*(r[c] for r in sel))) for c in COLUMNS) + " |\n")
    print(f"wrote {args.out}.csv and {args.out}.md ({len(rows)} rows)")


if __name__ == "__main__":
    main()
