#!/usr/bin/env bash
# KernelBench level-3 benchmark: MimIR (LLVM and MLIR paths) vs torch-mlir.
#
#   ./run.sh [all|export|build|report] [--only "1_MLP 44_MiniGPTBlock"] [--models FILE] [--force]
#
#   export   pipeline/export.sh   MimIR frontend -> .ll + .mlir, torch-mlir -> .mlir
#   build    pipeline/build.sh    import, data, lower, drivers, compile, references
#   report   pipeline/report.sh   run all variants, check against PyTorch, timings
#   all      the three in order (default)
#
# Sub-steps can be run directly, e.g. `pipeline/build.sh lower`. Settings are in
# config.sh and can be overridden from the environment. See README.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

stages=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)   export ONLY="$2"; shift 2 ;;
    --models) export MODELS_FILE="$(realpath "$2")"; shift 2 ;;
    --force)  export FORCE=1; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    all|export|build|report) stages+=("$1"); shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ ${#stages[@]} -gt 0 ]] || stages=(all)

for s in "${stages[@]}"; do
  case "$s" in
    export) pipeline/export.sh all ;;
    build)  pipeline/build.sh all ;;
    report) pipeline/report.sh ;;
    all)    pipeline/export.sh all; pipeline/build.sh all; pipeline/report.sh ;;
  esac
done
