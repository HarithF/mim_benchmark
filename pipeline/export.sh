#!/usr/bin/env bash
# Stage 1: export every model.
#   TARGET=cpu: MimIR frontend (LLVM IR + MLIR) and torch-mlir (linalg MLIR).
#   TARGET=gpu: MimIR frontend for NVPTX and IREE for CUDA.
#   TARGET=both: all four.
#
#   pipeline/export.sh [mimir|torch|mimir_gpu|iree|all]      (default: all)
#
# Already-exported models are skipped; FORCE=1 redoes them.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

what="${1:-all}"
load_models
mkdir -p "$WORK/no_fixtures"

# --- MimIR: frontend sweep, one model per runner invocation -------------------
# --size-divisor 1 plus an EMPTY --fixtures dir = the shapes written in the
# KernelBench sources (the default overlay shrinks 26 level-3 models).
# --emit-mlir keeps <k>/mimir_graph_<hash>.ll (LLVM path, also run and
# compared with eager) and <k>/mimir_graph_<hash>.mlir (MLIR path, verified
# with mlir-opt) side by side; results/<k>.json records both outcomes.
export_mimir() {
  log "MimIR export: ${#MODELS[@]} model(s) -> $MIMIR_OUT"
  mkdir -p "$MIMIR_OUT/results" "$LOGS/export_mimir"
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
  mkdir -p "$TORCH_OUT" "$LOGS/export_torch"
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

# --- MimIR on the GPU: the same runner with target=nvptx -------------------------
# The backend reads MIMIR_TARGET/MIMIR_GPU_SM; the runner gives each case its own
# MIMIR_DEBUG_DIR subdirectory, which keeps <k>/mimir_graph_<hash>.ll (host code,
# fatbin embedded) and _dev.ll. The runner also runs it on the GPU and compares
# with eager PyTorch. No address-space limit: cuInit reserves huge virtual ranges.
export_mimir_gpu() {
  resolve_gpu_sm
  log "MimIR GPU export (nvptx, sm_$GPU_SM): ${#MODELS[@]} model(s) -> $MIMIR_GPU_OUT"
  mkdir -p "$MIMIR_GPU_OUT/results" "$LOGS/export_mimir_gpu"
  local k res
  for k in "${MODELS[@]}"; do
    res="$MIMIR_GPU_OUT/results/$k.json"
    if [[ -f "$res" && "${FORCE:-0}" != 1 ]]; then
      echo "  $k: done earlier ($(status_of "$res"))"
      continue
    fi
    rm -rf "${MIMIR_GPU_OUT:?}/$k" "$res"
    echo -n "  $k: "
    ( cd "$FRONTEND" && MIMIR_TARGET=nvptx MIMIR_GPU_SM="$GPU_SM" MIMIR_DEBUG_DIR="$MIMIR_GPU_OUT" \
        MIMIR_CACHE_DIR="$WORK/jit_cache" uv run --no-sync python scripts/run_kernelbench_mimir.py \
        --suite level3 --kernel "level3/$k.py" \
        --size-divisor 1 --fixtures "$WORK/no_fixtures" \
        --timeout "$MIMIR_TIMEOUT" --max-memory-gb 0 --max-fp-iters "$MIMIR_MAX_FP_ITERS" \
        --min-pass-rate 0 --results-json "$res" \
    ) > "$LOGS/export_mimir_gpu/$k.log" 2>&1 || true
    if [[ -f "$res" ]]; then status_of "$res"; else echo "no result (see $LOGS/export_mimir_gpu/$k.log)"; fi
  done
}

# --- IREE baseline (GPU): iree-turbine export + iree-compile for CUDA ------------
# Parameters and buffers are externalized by name; the driver binds them from
# weights.bin (scripts/iree_driver.py).
export_iree() {
  resolve_gpu_sm
  "$IREE_PY" -c "import iree.turbine, iree.runtime" 2>/dev/null \
    || die "IREE_PY=$IREE_PY cannot import iree.turbine / iree.runtime"
  log "IREE export (cuda, sm_$GPU_SM): ${#MODELS[@]} model(s) -> $IREE_OUT"
  mkdir -p "$LOGS/export_iree"
  local k
  for k in "${MODELS[@]}"; do
    if [[ -f "$IREE_OUT/$k/$k.iree.json" && "${FORCE:-0}" != 1 ]]; then
      echo "  $k: done earlier"
      continue
    fi
    rm -rf "${IREE_OUT:?}/$k"
    echo -n "  $k: "
    if timeout "$IREE_TIMEOUT" "$IREE_PY" "$SCRIPTS/export_iree.py" "$KERNELBENCH_DIR/$k.py" "$IREE_OUT/$k" \
         --target cuda --sm "$GPU_SM" > "$LOGS/export_iree/$k.log" 2>&1; then
      echo "ok"
    else
      echo "FAILED (see $LOGS/export_iree/$k.log)"
    fi
  done
}

case "$what" in
  mimir) export_mimir ;;
  mimir_gpu) export_mimir_gpu ;;
  iree)  export_iree ;;
  torch) export_torch ;;
  all)
    if want_cpu; then export_mimir; export_torch; fi
    if want_gpu; then export_mimir_gpu; export_iree; fi ;;
  *)     die "usage: $0 [mimir|torch|mimir_gpu|iree|all]" ;;
esac
