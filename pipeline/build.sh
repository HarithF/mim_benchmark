#!/usr/bin/env bash
# Stage 2: import the exported models into the benchmark tree, generate data
# and drivers, lower the MLIR variants and compile everything to binaries.
#
#   pipeline/build.sh [import|data|lower|drivers|compile|refs|all]   (default: all)
#
# The steps are idempotent: data is regenerated only when the model's mim .ll
# is newer than its manifest, an MLIR variant is re-lowered only when its
# source .mlir is newer than its .ll. FORCE=1 redoes everything.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

what="${1:-all}"
load_models
mkdir -p "$MODELS_DIR" "$LOGS/lower"

DIRS=()   # benchmark-tree dirs of the selected models that exist after import
collect_dirs() {
  DIRS=()
  local k d
  for k in "${MODELS[@]}"; do
    d="$MODELS_DIR/$(model_dir_name "$k")"
    [[ -d "$d" ]] && DIRS+=("$d")
  done
}

# --- import ----------------------------------------------------------------------
# TARGET=cpu: a model is imported only if BOTH CPU frontends produced it (and,
# unless ALLOW_UNVERIFIED=1, the MimIR run passed on both of its paths), so every
# model has all three CPU variants. TARGET=gpu: mim_nvptx, then iree_cuda for the
# models that have it. MimIR's .ll (mim_llvm, else mim_nvptx) is the shape and
# weight-layout source for data, drivers and the reporter.
step_import() {
  log "import -> $MODELS_DIR"
  if want_cpu; then import_cpu; fi
  if want_gpu; then import_mimir_gpu; import_iree; fi
  # The PyTorch source, for the reference output and for forward()'s input order.
  local k d
  for k in "${MODELS[@]}"; do
    d="$MODELS_DIR/$(model_dir_name "$k")"
    [[ -d "$d/mim_llvm" || -d "$d/mim_nvptx" ]] || continue
    rm -rf "$d/torch_ref" && mkdir -p "$d/torch_ref" && cp "$KERNELBENCH_DIR/$k.py" "$d/torch_ref/"
  done
}

import_cpu() {
  local stage="$WORK/stage" k res why n_ok=0
  rm -rf "$stage" && mkdir -p "$stage/mimir" "$stage/torch"
  for k in "${MODELS[@]}"; do
    why=""
    res="$MIMIR_OUT/results/$k.json"
    if [[ ! -f "$res" ]]; then
      why="no MimIR export"
    elif [[ "$ALLOW_UNVERIFIED" != 1 ]] && ! "$PYTHON" - "$res" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1])); r = (r["results"] if isinstance(r, dict) else r)[0]
sys.exit(0 if r["status"] == "PASS" and r.get("mlir_status") == "PASS" else 1)
EOF
    then
      why="MimIR run did not pass on both paths (ALLOW_UNVERIFIED=1 to import anyway)"
    elif [[ $(find "$MIMIR_OUT/$k" -maxdepth 1 -name 'mimir_graph_*.ll' 2>/dev/null | wc -l) -ne 1 \
         || $(find "$MIMIR_OUT/$k" -maxdepth 1 -name 'mimir_graph_*.mlir' 2>/dev/null | wc -l) -ne 1 ]]; then
      why="expected one mimir_graph_*.ll and one .mlir in $MIMIR_OUT/$k"
    elif [[ ! -f "$TORCH_OUT/$k.mlir" || ! -f "$TORCH_OUT/$k.params.txt" ]]; then
      why="no torch-mlir export"
    fi
    if [[ -n "$why" ]]; then
      echo "  skip $k: $why"
      continue
    fi
    ln -s "$MIMIR_OUT/$k" "$stage/mimir/$k"
    ln -s "$TORCH_OUT/$k.mlir" "$TORCH_OUT/$k.params.txt" "$stage/torch/"
    n_ok=$((n_ok + 1))
  done
  echo "  $n_ok of ${#MODELS[@]} model(s) importable (CPU)"
  [[ $n_ok -gt 0 ]] || return 0
  "$SCRIPTS/import_mimir.sh" "$stage/mimir" "$MODELS_DIR" | tail -1
  "$SCRIPTS/import_torch_mlir.sh" "$stage/torch" "$MODELS_DIR" | tail -1
}

# <k>/mimir_graph_<hash>.ll (host code with the device fatbin embedded)
# -> NN_<Model>/mim_nvptx/<Model>_mim_nvptx.ll, symbol renamed to @<Model> as
# import_mimir.sh does. Its signature equals mim_llvm's. _dev.ll is not needed.
import_mimir_gpu() {
  local k d model res sym why n_ok=0 lls
  for k in "${MODELS[@]}"; do
    d="$MODELS_DIR/$(model_dir_name "$k")"
    res="$MIMIR_GPU_OUT/results/$k.json"
    mapfile -t lls < <(find "$MIMIR_GPU_OUT/$k" -maxdepth 1 -name 'mimir_graph_*.ll' ! -name '*_dev.ll' 2>/dev/null)
    why=""
    if [[ ! -f "$res" ]]; then
      why="no MimIR GPU export"
    elif [[ "$ALLOW_UNVERIFIED" != 1 ]] && ! "$PYTHON" - "$res" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1])); r = (r["results"] if isinstance(r, dict) else r)[0]
sys.exit(0 if r["status"] == "PASS" else 1)
EOF
    then
      why="MimIR GPU run did not pass (ALLOW_UNVERIFIED=1 to import anyway)"
    elif [[ ${#lls[@]} -ne 1 ]]; then
      why="expected one host mimir_graph_*.ll in $MIMIR_GPU_OUT/$k, found ${#lls[@]}"
    elif ! grep -q '^@.fatbin = private constant' "${lls[0]}"; then
      why="no embedded fatbin in $(basename "${lls[0]}") (ptxas/fatbinary missing at export?)"
    fi
    if [[ -n "$why" ]]; then
      echo "  skip $k (mim_nvptx): $why"
      continue
    fi
    model="${k#*_}"
    mkdir -p "$d/mim_nvptx"
    sym="$(basename "${lls[0]}" .ll)"
    sed "s/@${sym}\b/@${model}/g" "${lls[0]}" > "$d/mim_nvptx/${model}_mim_nvptx.ll.tmp"
    if cmp -s "$d/mim_nvptx/${model}_mim_nvptx.ll.tmp" "$d/mim_nvptx/${model}_mim_nvptx.ll"; then
      rm -f "$d/mim_nvptx/${model}_mim_nvptx.ll.tmp"
    else
      mv -f "$d/mim_nvptx/${model}_mim_nvptx.ll.tmp" "$d/mim_nvptx/${model}_mim_nvptx.ll"
    fi
    n_ok=$((n_ok + 1))
  done
  echo "  $n_ok model(s) with a mim_nvptx variant"
}

# $IREE_OUT/<k>/{<k>_cuda.vmfb,<k>.iree.json} -> NN_<Model>/iree_cuda/. Needs a
# MimIR variant in the same model dir (its .ll defines weights.bin's layout).
# The executable bin/iree_cuda is a wrapper written by the compile step.
import_iree() {
  local k d n_ok=0
  for k in "${MODELS[@]}"; do
    d="$MODELS_DIR/$(model_dir_name "$k")"
    if [[ ! -f "$IREE_OUT/$k/$k.iree.json" || ! -f "$IREE_OUT/$k/${k}_cuda.vmfb" ]]; then
      echo "  skip $k (iree_cuda): no IREE export"
      continue
    fi
    if [[ ! -d "$d/mim_llvm" && ! -d "$d/mim_nvptx" ]]; then
      echo "  skip $k (iree_cuda): no MimIR variant to take the weight layout from"
      continue
    fi
    rm -rf "$d/iree_cuda" && mkdir -p "$d/iree_cuda"
    cp "$IREE_OUT/$k/${k}_cuda.vmfb" "$d/iree_cuda/"
    cp "$IREE_OUT/$k/$k.iree.json" "$d/iree_cuda/${k#*_}.iree.json"
    n_ok=$((n_ok + 1))
  done
  echo "  $n_ok model(s) with an iree_cuda variant"
}

# bin/iree_cuda: runs scripts/iree_driver.py with IREE_PY, same command line as the C drivers.
write_iree_wrappers() {
  local d meta
  for d in "${DIRS[@]}"; do
    meta=$(find "$d/iree_cuda" -maxdepth 1 -name '*.iree.json' 2>/dev/null | head -1 || true)
    [[ -n "$meta" ]] || continue
    mkdir -p "$d/bin"
    printf '#!/usr/bin/env bash\nexec %q %q %q "$@"\n' "$IREE_PY" "$SCRIPTS/iree_driver.py" "$meta" > "$d/bin/iree_cuda"
    chmod +x "$d/bin/iree_cuda"
  done
}

# --- data ------------------------------------------------------------------------
step_data() {
  log "data (gen_data.py, seed $DATA_SEED)"
  collect_dirs
  local d m ll force
  for d in "${DIRS[@]}"; do
    ll=$(find "$d/mim_llvm" "$d/mim_nvptx" -maxdepth 1 -name '*.ll' 2>/dev/null | head -1 || true)
    m=$(find "$d" -maxdepth 1 -name '*_params.json' | head -1)
    force=()
    if [[ "${FORCE:-0}" == 1 || -z "$m" || ( -n "$ll" && "$ll" -nt "$m" ) ]]; then force=(--force); fi
    if [[ ${#force[@]} -eq 0 ]]; then echo "  $(basename "$d"): up to date"; continue; fi
    "$PYTHON" "$SCRIPTS/gen_data.py" "$d" --seed "$DATA_SEED" "${force[@]}" \
      | grep -E '^\[|input +:|NOTE|ERROR|Error|wrote .*weights' || true
  done
}

# --- lower -----------------------------------------------------------------------
# lower.sh on every MLIR variant dir (mlir_llvm = torch-mlir, mim_mlir_llvm =
# MimIR's MLIR), LOWER_JOBS at a time. A dir whose .ll is newer than its source
# .mlir is up to date. The old .ll is removed first, so a failed or timed-out
# lowering can never leave a stale one behind for the compile step.
lower_one() {
  local dir="$1" log="$LOGS/lower/$(basename "$(dirname "$1")")_$(basename "$1").log" start=$SECONDS
  rm -f "$dir"/*.ll "$dir"/*_lowered.mlir
  if timeout "$LOWER_TIMEOUT" "$SCRIPTS/lower.sh" "$dir" > "$log" 2>&1 && ls "$dir"/*.ll > /dev/null 2>&1; then
    echo "  ok      $(basename "$(dirname "$dir")")/$(basename "$dir")  ($((SECONDS - start)) s)"
  else
    echo "  FAILED  $(basename "$(dirname "$dir")")/$(basename "$dir")  ($((SECONDS - start)) s, see $log)"
  fi
}
export -f lower_one
export SCRIPTS LOGS LOWER_TIMEOUT

step_lower() {
  log "lower (lower.sh, $LOWER_JOBS parallel job(s), timeout ${LOWER_TIMEOUT}s per dir; 0 = none)"
  collect_dirs
  local d v src ll todo=()
  for d in "${DIRS[@]}"; do
    for v in mlir_llvm mim_mlir_llvm; do
      src=$(find "$d/$v" -maxdepth 1 -name '*.mlir' ! -name '*_lowered.mlir' 2>/dev/null | head -1 || true)
      [[ -n "$src" ]] || continue
      ll=$(find "$d/$v" -maxdepth 1 -name '*.ll' | head -1)
      if [[ "${FORCE:-0}" != 1 && -n "$ll" && "$ll" -nt "$src" ]]; then continue; fi
      todo+=("$d/$v")
    done
  done
  echo "  ${#todo[@]} variant dir(s) to lower"
  [[ ${#todo[@]} -gt 0 ]] || return 0
  printf '%s\n' "${todo[@]}" | xargs -P "$LOWER_JOBS" -I{} bash -c 'lower_one "$1"' _ {}
}

# --- drivers ---------------------------------------------------------------------
step_drivers() {
  log "drivers (gen_drivers.py)"
  collect_dirs
  local d
  for d in "${DIRS[@]}"; do
    "$PYTHON" "$SCRIPTS/gen_drivers.py" "$d" --force | grep -E '^\[|weights:|ERROR|WARNING|SKIP' || true
  done
}

# --- compile ---------------------------------------------------------------------
# Every variant that has a .ll and a driver. bin/ is cleared first so that a
# variant which no longer builds cannot leave an old binary to be timed.
step_compile() {
  log "compile (compile_all.sh, $OPT_LEVEL)"
  collect_dirs
  local d names=()
  for d in "${DIRS[@]}"; do rm -rf "$d/bin"; names+=("$(basename "$d")"); done
  COMPILE_ONLY="${names[*]}" "$SCRIPTS/compile_all.sh" "$MODELS_DIR" "$OPT_LEVEL" "$LLVM_LIB_DIR" \
    > "$LOGS/compile.log" 2>&1 || true
  grep -E '^!!|==> Done|failed' "$LOGS/compile.log" || true
  local n_missing=0 v variants
  mapfile -t variants < <(target_variants)
  if want_gpu; then write_iree_wrappers; fi
  for d in "${DIRS[@]}"; do
    for v in "${variants[@]}"; do
      [[ -x "$d/bin/$v" ]] || { echo "  missing binary: $(basename "$d")/bin/$v"; n_missing=$((n_missing + 1)); }
    done
  done
  echo "  ${#DIRS[@]} model(s), $n_missing missing binar(y/ies)"
}

# --- refs ------------------------------------------------------------------------
# Cache each model's PyTorch output (<Name>_ref.bin), one subprocess per model.
step_refs() {
  log "references (produce_refs.py)"
  collect_dirs
  local d
  for d in "${DIRS[@]}"; do
    "$PYTHON" "$SCRIPTS/produce_refs.py" "$d" 2>&1 | grep -vE '^warn: cpu' | tail -1 || true
  done
}

case "$what" in
  import)  step_import ;;
  data)    step_data ;;
  lower)   step_lower ;;
  drivers) step_drivers ;;
  compile) step_compile ;;
  refs)    step_refs ;;
  all)     step_import; step_data; step_lower; step_drivers; step_compile; step_refs ;;
  *)       die "usage: $0 [import|data|lower|drivers|compile|refs|all]" ;;
esac
