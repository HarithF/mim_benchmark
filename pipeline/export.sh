#!/usr/bin/env bash
# Stage 1: compile every model with the MimIR frontend (LLVM IR + MLIR) and
# export it with torch-mlir (linalg-on-tensors MLIR).
#
#   pipeline/export.sh [mimir|torch|all]      (default: all)
#
# Already-exported models are skipped; FORCE=1 redoes them.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

what="${1:-all}"
load_models
mkdir -p "$MIMIR_OUT/results" "$TORCH_OUT" "$LOGS/export_mimir" "$LOGS/export_torch" "$WORK/no_fixtures"

# --- MimIR: frontend sweep, one model per runner invocation -------------------
# --size-divisor 1 plus an EMPTY --fixtures dir = the shapes written in the
# KernelBench sources (the default overlay shrinks 26 level-3 models).
# --emit-mlir keeps <k>/mimir_graph_<hash>.ll (LLVM path, also run and
# compared with eager) and <k>/mimir_graph_<hash>.mlir (MLIR path, verified
# with mlir-opt) side by side; results/<k>.json records both outcomes.
export_mimir() {
  log "MimIR export: ${#MODELS[@]} model(s) -> $MIMIR_OUT"
  local k res
  for k in "${MODELS[@]}"; do
    res="$MIMIR_OUT/results/$k.json"
    if [[ -f "$res" && "${FORCE:-0}" != 1 ]]; then
      echo "  $k: done earlier ($(status_of "$res"))"
      continue
    fi
    rm -rf "${MIMIR_OUT:?}/$k" "$res"      # a stale graph would make the import ambiguous
    echo -n "  $k: "
    ( cd "$FRONTEND" && MIMIR_CACHE_DIR="$WORK/jit_cache" uv run --no-sync python scripts/run_kernelbench_mimir.py \
        --suite level3 --kernel "level3/$k.py" \
        --size-divisor 1 --fixtures "$WORK/no_fixtures" \
        --timeout "$MIMIR_TIMEOUT" --max-memory-gb "$MIMIR_MAX_MEMORY_GB" --max-fp-iters "$MIMIR_MAX_FP_ITERS" \
        --emit-mlir "$MIMIR_OUT" --min-pass-rate 0 --results-json "$res" \
    ) > "$LOGS/export_mimir/$k.log" 2>&1 || true
    if [[ -f "$res" ]]; then status_of "$res"; else echo "no result (see $LOGS/export_mimir/$k.log)"; fi
  done
}

status_of() {
  "$PYTHON" - "$1" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
r = (r["results"] if isinstance(r, dict) else r)[0]
first = lambda s: (s or "").strip().splitlines()[-1][:100] if (s or "").strip() else ""
print(f"llvm {r['status']} [{r['phase']}]  mlir {r.get('mlir_status') or '-'}"
      + (f"  -- {first(r.get('detail'))}" if r["status"] != "PASS" else ""))
EOF
}

# --- torch-mlir: scripts/export_mlir.py, one model per invocation --------------
# Floating-point buffers are passed as arguments (not frozen), and each module
# gets a <k>.params.txt naming every argument -- the drivers bind by it.
export_torch() {
  "$TORCH_MLIR_PY" -c "import torch_mlir" 2>/dev/null \
    || die "TORCH_MLIR_PY=$TORCH_MLIR_PY cannot import torch_mlir; point it at a Python that can"
  log "torch-mlir export: ${#MODELS[@]} model(s) -> $TORCH_OUT"
  local k
  for k in "${MODELS[@]}"; do
    if [[ -f "$TORCH_OUT/$k.mlir" && "${FORCE:-0}" != 1 ]]; then
      echo "  $k: done earlier"
      continue
    fi
    rm -f "$TORCH_OUT/$k.mlir" "$TORCH_OUT/$k.params.txt"
    echo -n "  $k: "
    if "$TORCH_MLIR_PY" "$SCRIPTS/export_mlir.py" --repo-dir "$KERNELBENCH_DIR" --out-dir "$TORCH_OUT" \
         --pattern "$k.py" --timeout "$TORCH_EXPORT_TIMEOUT" > "$LOGS/export_torch/$k.log" 2>&1 \
       && [[ -f "$TORCH_OUT/$k.mlir" ]]; then
      echo "ok"
    else
      echo "FAILED (see $LOGS/export_torch/$k.log)"
    fi
  done
}

case "$what" in
  mimir) export_mimir ;;
  torch) export_torch ;;
  all)   export_mimir; export_torch ;;
  *)     die "usage: $0 [mimir|torch|all]" ;;
esac
