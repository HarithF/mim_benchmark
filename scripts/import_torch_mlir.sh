#!/usr/bin/env bash
#
# import_torch_mlir.sh
#
# Imports freshly generated torch-mlir graphs from a flat source directory
# into the per-model benchmark directories, renaming both the file and the
# graph function symbol inside it, e.g.:
#
#   <SRC>/1_MLP.mlir -> 01_MLP/mlir_llvm/MLP_mlir.mlir   (@m_1_MLP -> @MLP)
#
# Logic:
#   - model name   = source file name with the leading "N_" prefix and the
#                    ".mlir" extension stripped
#   - target dir   = model number zero-padded to 2 digits + "_" + model name
#   - the function symbol is read from the file's own "func.func @<sym>" line
#     (torch-mlir emits "m_<N>_<Model>") and rewritten to "@<model name>"
#   - a model is skipped entirely, leaving its target directory untouched, if
#     the symbol cannot be determined or the file holds more than one function
#   - missing target directories (and their mlir_llvm subdirectory) are created
#
# Usage:
#   ./import_torch_mlir.sh SRC_DIR DEST_ROOT [--dry-run]
#
# Both SRC_DIR (the flat directory of generated <N>_<Model>.mlir files) and
# DEST_ROOT (the benchmark root holding the NN_<Model> directories) are
# required.
#
# Examples:
#   ./import_torch_mlir.sh ../torch/mlir_full .
#   ./import_torch_mlir.sh ../torch/mlir_full . --dry-run

set -euo pipefail
shopt -s nullglob

usage() {
  echo "Usage: $(basename "${BASH_SOURCE[0]}") SRC_DIR DEST_ROOT [--dry-run]" >&2
  echo >&2
  echo "  SRC_DIR    flat directory of generated <N>_<Model>.mlir files" >&2
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
n_no_manifest=0

# Copies $1 to $2, rewriting the "@<symbol>" function name to "@<model>".
install_renamed() {
  local src="$1" dst="$2" sym="$3" model="$4"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    $(basename "$src") -> ${dst#"$DEST_ROOT/"}  (@$sym -> @$model)"
    return
  fi

  mkdir -p "$(dirname "$dst")"
  # Replace only on a content change, so mtimes stay meaningful for the
  # build pipeline's "is the .ll newer than its source?" checks.
  sed "s/@${sym}\b/@${model}/g" "$src" > "$dst.tmp"
  if cmp -s "$dst.tmp" "$dst"; then rm -f "$dst.tmp"; else mv -f "$dst.tmp" "$dst"; fi
  echo "    $(basename "$src") -> ${dst#"$DEST_ROOT/"}  (@$sym -> @$model)"
}

for src in "$SRC_DIR"/*.mlir; do
  base="$(basename "$src" .mlir)"

  # Only files that look like "<number>_<ModelName>.mlir".
  if [[ ! "$base" =~ ^([0-9]+)_(.+)$ ]]; then
    continue
  fi
  num="${BASH_REMATCH[1]}"
  model="${BASH_REMATCH[2]}"
  dest="$DEST_ROOT/$(printf '%02d' "$num")_${model}"

  echo "$base.mlir -> $(basename "$dest")"

  # The graph function symbol comes from the file itself rather than being
  # guessed from the file name.
  syms=()
  while IFS= read -r s; do
    [[ -n "$s" ]] && syms+=("$s")
  done < <(grep -o 'func\.func[[:space:]]\+\(public[[:space:]]\+\)\?@[A-Za-z0-9_$.]\+' "$src" | sed 's/.*@//')

  if [[ ${#syms[@]} -eq 0 ]]; then
    echo "    no func.func symbol found, skipping model" >&2
    n_skipped=$((n_skipped + 1))
    continue
  fi
  if [[ ${#syms[@]} -gt 1 ]]; then
    echo "    multiple functions found (${syms[*]}), skipping model" >&2
    n_skipped=$((n_skipped + 1))
    continue
  fi
  sym="${syms[0]}"

  if [[ ! -d "$dest" ]]; then
    echo "    creating $(basename "$dest")"
    [[ $DRY_RUN -eq 0 ]] && mkdir -p "$dest/mlir_llvm"
    n_created=$((n_created + 1))
  fi

  install_renamed "$src" "$dest/mlir_llvm/${model}_mlir.mlir" "$sym" "$model"

  # The argument manifest travels with the graph. torch-mlir names its
  # arguments %arg0, %arg1, ... so the .mlir alone cannot say which weight
  # each one wants; export_mlir.py's <stem>.params.txt is the only record of
  # that, and gen_drivers.py needs it to bind the driver to the right offsets
  # in weights.bin (see scripts/param_manifest.py). Without it the mlir_llvm
  # driver falls back to positional matching, which is exactly the bug that
  # made every BatchNorm model read its weights out of phase.
  if [[ -f "$SRC_DIR/$base.params.txt" ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "    $base.params.txt -> $(basename "$dest")/mlir_llvm/${model}_mlir.params.txt"
    else
      cmp -s "$SRC_DIR/$base.params.txt" "$dest/mlir_llvm/${model}_mlir.params.txt" || cp "$SRC_DIR/$base.params.txt" "$dest/mlir_llvm/${model}_mlir.params.txt"
      echo "    $base.params.txt -> $(basename "$dest")/mlir_llvm/${model}_mlir.params.txt"
    fi
  else
    echo "    WARNING: no $base.params.txt beside the .mlir -- the mlir_llvm" >&2
    echo "             driver will have no argument names to bind by" >&2
    n_no_manifest=$((n_no_manifest + 1))
  fi

  n_copied=$((n_copied + 1))
done

echo
echo "models imported: $n_copied  dirs created: $n_created  models skipped: $n_skipped"
if [[ $n_no_manifest -gt 0 ]]; then
  echo "WARNING: $n_no_manifest model(s) arrived without a .params.txt manifest" >&2
fi
