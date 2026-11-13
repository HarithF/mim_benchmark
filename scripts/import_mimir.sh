#!/usr/bin/env bash
#
# import_mimir.sh
#
# Imports freshly generated mimir graphs from a source tree into the
# per-model benchmark directories, renaming both the files and the graph
# function symbol inside them, e.g.:
#
#   <SRC>/1_MLP/mimir_graph_ae1358530cc26de2.ll   -> 01_MLP/mim_llvm/MLP_mim.ll
#   <SRC>/1_MLP/mimir_graph_ae1358530cc26de2.mlir -> 01_MLP/mim_mlir_llvm/MLP_mim_mlir.mlir
#
# Logic:
#   - model name   = source dir name with the leading "N_" prefix stripped
#   - target dir   = model number zero-padded to 2 digits + "_" + model name
#   - the "@mimir_graph_<hash>" function symbol is rewritten to "@<model name>"
#     (basic block labels carrying the same hash are left untouched, matching
#     the existing files)
#   - a model is imported only if BOTH its .ll and .mlir are present;
#     otherwise it is skipped entirely and no target directory is touched
#   - missing target directories (and their mim_llvm / mim_mlir_llvm
#     subdirectories) are created
#
# Usage:
#   ./import_mimir.sh SRC_DIR DEST_ROOT [--dry-run]
#
# Both SRC_DIR (the tree of generated mimir_graph_* files) and DEST_ROOT (the
# benchmark root holding the NN_<Model> directories) are required.
#
# Examples:
#   ./import_mimir.sh ../myfrong/mimir-frontend/mlir_regen.stale/mlir .
#   ./import_mimir.sh ../myfrong/mimir-frontend/mlir_regen.stale/mlir . --dry-run

set -euo pipefail
shopt -s nullglob

usage() {
  echo "Usage: $(basename "${BASH_SOURCE[0]}") SRC_DIR DEST_ROOT [--dry-run]" >&2
  echo >&2
  echo "  SRC_DIR    tree of generated <N>_<Model>/mimir_graph_*.{ll,mlir} files" >&2
  echo "  DEST_ROOT  benchmark root holding the NN_<Model> directories" >&2
}

DRY_RUN=0

positional=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
    *) positional+=("$arg") ;;
  esac
done

if [[ ${#positional[@]} -lt 2 ]]; then
  echo "SRC_DIR and DEST_ROOT are required." >&2
  usage
  exit 1
fi
if [[ ${#positional[@]} -gt 2 ]]; then
  echo "Too many arguments: ${positional[*]}" >&2
  usage
  exit 1
fi

SRC_DIR="${positional[0]}"
DEST_ROOT="${positional[1]}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source directory not found: $SRC_DIR" >&2
  exit 1
fi
if [[ ! -d "$DEST_ROOT" ]]; then
  echo "Destination root not found: $DEST_ROOT" >&2
  exit 1
fi

SRC_DIR="$(cd "$SRC_DIR" && pwd)"
DEST_ROOT="$(cd "$DEST_ROOT" && pwd)"

echo "Source:      $SRC_DIR"
echo "Destination: $DEST_ROOT"
[[ $DRY_RUN -eq 1 ]] && echo "(dry run)"
echo

n_copied=0
n_created=0
n_skipped=0

# Copies $1 to $2, rewriting the "@<symbol>" function name to "@<model>".
install_renamed() {
  local src="$1" dst="$2" sym="$3" model="$4"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    $(basename "$src") -> ${dst#$DEST_ROOT/}  (@$sym -> @$model)"
    return
  fi

  mkdir -p "$(dirname "$dst")"
  # Replace only on a content change, so mtimes stay meaningful for the
  # build pipeline's "is the .ll newer than its source?" checks.
  sed "s/@${sym}\b/@${model}/g" "$src" > "$dst.tmp"
  if cmp -s "$dst.tmp" "$dst"; then rm -f "$dst.tmp"; else mv -f "$dst.tmp" "$dst"; fi
  echo "    $(basename "$src") -> ${dst#$DEST_ROOT/}  (@$sym -> @$model)"
}

for src in "$SRC_DIR"/*/; do
  src="${src%/}"
  base="$(basename "$src")"

  # Only directories that look like "<number>_<ModelName>".
  if [[ ! "$base" =~ ^([0-9]+)_(.+)$ ]]; then
    continue
  fi
  num="${BASH_REMATCH[1]}"
  model="${BASH_REMATCH[2]}"
  dest="$DEST_ROOT/$(printf '%02d' "$num")_${model}"

  echo "$base -> $(basename "$dest")"

  # There should be exactly one mimir_graph_<hash> per model directory; the
  # hash is the function symbol we need to rewrite.
  syms=()
  for f in "$src"/mimir_graph_*.ll "$src"/mimir_graph_*.mlir; do
    s="$(basename "$f")"
    s="${s%.*}"
    [[ " ${syms[*]-} " == *" $s "* ]] || syms+=("$s")
  done

  if [[ ${#syms[@]} -eq 0 ]]; then
    echo "    no mimir_graph_* .ll/.mlir found, skipping" >&2
    n_skipped=$((n_skipped + 1))
    continue
  fi
  if [[ ${#syms[@]} -gt 1 ]]; then
    echo "    multiple graphs found (${syms[*]}), skipping" >&2
    n_skipped=$((n_skipped + 1))
    continue
  fi
  sym="${syms[0]}"

  # Both files must be present: a model is imported as a whole or not at all,
  # so a half-populated target directory is never left behind.
  missing=()
  [[ -f "$src/$sym.ll" ]]   || missing+=("$sym.ll")
  [[ -f "$src/$sym.mlir" ]] || missing+=("$sym.mlir")

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "    missing ${missing[*]}, skipping model" >&2
    n_skipped=$((n_skipped + 1))
    continue
  fi

  if [[ ! -d "$dest" ]]; then
    echo "    creating $(basename "$dest")"
    [[ $DRY_RUN -eq 0 ]] && mkdir -p "$dest/mim_llvm" "$dest/mim_mlir_llvm"
    n_created=$((n_created + 1))
  fi

  install_renamed "$src/$sym.ll" "$dest/mim_llvm/${model}_mim.ll" "$sym" "$model"
  install_renamed "$src/$sym.mlir" "$dest/mim_mlir_llvm/${model}_mim_mlir.mlir" "$sym" "$model"
  n_copied=$((n_copied + 1))
done

echo
echo "models imported: $n_copied  dirs created: $n_created  models skipped: $n_skipped"
