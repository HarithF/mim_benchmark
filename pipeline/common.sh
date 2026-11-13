# Sourced by run.sh and the stage scripts.
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$BENCH_ROOT/scripts"
# shellcheck source=../config.sh
source "$BENCH_ROOT/config.sh"
export LLVM_LIB_DIR

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

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

# 44_MiniGPTBlock -> 44_MiniGPTBlock, 1_MLP -> 01_MLP (the benchmark tree's directory name)
model_dir_name() { printf '%02d_%s' "$((10#${1%%_*}))" "${1#*_}"; }
