# Sourced by run.sh and the stage scripts.
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$BENCH_ROOT/scripts"
# shellcheck source=../config.sh
source "$BENCH_ROOT/config.sh"
export LLVM_LIB_DIR CUDA_HOME

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

case "$TARGET" in
  cpu|gpu|both) ;;
  *) die "TARGET must be cpu, gpu or both (got '$TARGET')" ;;
esac
want_cpu() { [[ "$TARGET" != gpu ]]; }
want_gpu() { [[ "$TARGET" != cpu ]]; }

# The variants (bin/ executables) this target builds.
target_variants() {
  if want_cpu; then printf '%s\n' mim_llvm mim_mlir_llvm mlir_llvm; fi
  if want_gpu; then printf '%s\n' mim_nvptx iree_cuda; fi
}

# MODELS: the KernelBench stems to process (models.txt, optionally narrowed by
# ONLY="1_MLP 44_MiniGPTBlock").
load_models() {
  [[ -f "$MODELS_FILE" ]] || die "no model list at $MODELS_FILE"
  mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "$MODELS_FILE" | sed 's/[[:space:]]//g')
  if [[ -n "${ONLY:-}" ]]; then
    local keep=() m o
    for m in "${MODELS[@]}"; do
      for o in $ONLY; do [[ "$m" == "$o" ]] && keep+=("$m"); done
    done
    MODELS=("${keep[@]}")
  fi
  [[ ${#MODELS[@]} -gt 0 ]] || die "no models selected"
}

# GPU_SM, e.g. 120: from the environment, else from nvidia-smi (first GPU). MimIR
# could ask nvidia-smi itself, but IREE needs it explicitly and both must agree.
resolve_gpu_sm() {
  [[ -n "$GPU_SM" ]] && return 0
  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
  [[ "$cap" =~ ^[0-9]+$ ]] || die "GPU_SM is unset and nvidia-smi cannot tell the compute capability; set GPU_SM (e.g. 86)"
  GPU_SM="$cap"
}

# 44_MiniGPTBlock -> 44_MiniGPTBlock, 1_MLP -> 01_MLP (the benchmark tree's directory name)
model_dir_name() { printf '%02d_%s' "$((10#${1%%_*}))" "${1#*_}"; }
