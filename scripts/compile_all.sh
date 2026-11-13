#!/usr/bin/env bash
#
# compile_all.sh - build every <model>/<variant>/ under a models root into an
# executable: the variant's .ll (opt -> llc) linked with its generated driver
# <Name>_driver_{mim,mlir}.c (clang), written to <model>/bin/<variant>.
#
# Usage:
#   ./compile_all.sh [MODELS_ROOT] [OPT_LEVEL] [LLVM_LIB_DIR] [-portable] [-no-opt]
#
#   ./compile_all.sh ./models                 # -O3, host CPU (default)
#   ./compile_all.sh ./models O2
#   ./compile_all.sh ./models O3 /path/to/llvm-project/build/lib
#
# `opt` is required: llc only does code generation, while the loop and SLP
# vectorizers run in opt's middle-end. The lowered loops are shaped for the
# vectorizer, so skipping opt leaves them scalar (~11x slower on a VGG16 conv).
# opt and llc must come from the same LLVM as the mlir-translate that wrote the
# .ll files (textual IR is not stable across versions); this is checked below.
#
# Expected layout:
#   models/01_MLP/
#     mim_llvm/        MLP_mim.ll            MLP_driver_mim.c
#     mim_mlir_llvm/   MLP_mim_mlir_llvm.ll  MLP_driver_mlir.c
#     mlir_llvm/       MLP_mlir_llvm.ll      MLP_driver_mlir.c
#     MLP_weights.bin  MLP_input.bin
#     bin/             <- executables are written here
#
# A variant dir with a .ll but no driver is reported as failed; one without a
# .ll is skipped. Intermediate .opt.bc/.o files are removed after each build.
# COMPILE_ONLY="01_MLP 05_AlexNet" restricts the build to those model dirs.
#
set -euo pipefail

# ----------------------------------------------------------------------------
# Config / arg parsing
# ----------------------------------------------------------------------------

OPT_LEVEL="O3"
LLVM_LIB_DIR="${LLVM_LIB_DIR:-$(cd "$(dirname "$(command -v mlir-opt || echo .)")/../lib" 2>/dev/null && pwd)}"
NATIVE=true
RUN_OPT=true
MODELS_ROOT="."
root_explicitly_set=false

for arg in "$@"; do
  case "$arg" in
  O0 | O1 | O2 | O3 | Os | Oz)
    OPT_LEVEL="$arg"
    ;;
  -native)
    NATIVE=true # the default
    ;;
  -portable | -generic)
    # Generic x86-64 (SSE2) instead of the host CPU; only if the binaries
    # must run on another machine.
    NATIVE=false
    ;;
  -no-opt)
    # Skip opt (comparison only; see the header).
    RUN_OPT=false
    ;;
  *)
    if [[ -d "$arg" ]]; then
      # First directory argument: models root; second: LLVM lib dir.
      if [[ "$root_explicitly_set" == false ]]; then
        MODELS_ROOT="$arg"
        root_explicitly_set=true
      else
        LLVM_LIB_DIR="$arg"
      fi
    else
      echo "error: unknown argument '$arg'" >&2
      echo "usage: $0 [MODELS_ROOT] [OPT_LEVEL] [LLVM_LIB_DIR] [-portable] [-no-opt]" >&2
      exit 1
    fi
    ;;
  esac
done

MODELS_ROOT="$(cd "$MODELS_ROOT" && pwd)"

OPT_ARGS=("-${OPT_LEVEL}")
LLC_ARGS=("-${OPT_LEVEL}")
CLANG_ARGS=("-${OPT_LEVEL}")

# -no-pie: mlir-translate emits no PIC level, so llc may use absolute 32-bit
# addresses for globals, which a PIE link rejects.
CLANG_ARGS+=("-no-pie")

# opt needs an explicit -mtriple: the .ll has no target triple, and without one
# opt has no cost model and ignores -mcpu (reductions such as matmul then do not
# vectorize). The triple is passed even with -portable; only -mcpu is optional.
HOST_TRIPLE="$(llvm-config --host-target 2>/dev/null ||
  clang -dumpmachine 2>/dev/null ||
  echo x86_64-unknown-linux-gnu)"
OPT_ARGS+=("-mtriple=${HOST_TRIPLE}")

if [[ "$NATIVE" == true ]]; then
  OPT_ARGS+=("-mcpu=native")
  LLC_ARGS+=("-mcpu=native")
  CLANG_ARGS+=("-march=native")
fi

# ----------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------

if [[ ! -d "$LLVM_LIB_DIR" ]]; then
  echo "error: LLVM lib dir not found: $LLVM_LIB_DIR" >&2
  echo "       pass it as an argument or set LLVM_LIB_DIR env var" >&2
  exit 1
fi

required=(llc clang)
[[ "$RUN_OPT" == true ]] && required+=(opt)
for cmd in "${required[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required tool '$cmd' not found in PATH" >&2
    exit 1
  fi
done

# opt and llc must be the same LLVM version (see the header).
if [[ "$RUN_OPT" == true ]]; then
  opt_ver="$(opt --version 2>/dev/null | sed -n 's/.*LLVM version \([0-9]*\).*/\1/p' | head -1)"
  llc_ver="$(llc --version 2>/dev/null | sed -n 's/.*LLVM version \([0-9]*\).*/\1/p' | head -1)"
  if [[ -n "$opt_ver" && -n "$llc_ver" && "$opt_ver" != "$llc_ver" ]]; then
    echo "error: opt (LLVM $opt_ver) and llc (LLVM $llc_ver) are different versions" >&2
    echo "       both must match the mlir-translate that produced the .ll files" >&2
    echo "       opt: $(command -v opt)" >&2
    echo "       llc: $(command -v llc)" >&2
    exit 1
  fi
fi

# ----------------------------------------------------------------------------
# Discover <model>/<variant>/ directories
# ----------------------------------------------------------------------------

echo "==> Scanning ${MODELS_ROOT} for model/variant directories"
echo "==> Building with -${OPT_LEVEL} (LLVM libs: ${LLVM_LIB_DIR})"
echo

FAILED=()
BUILT=0

# Wall-clock seconds per step, written to <variant>/<Name>_compile_timing.json.
now() { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f", b - a }'; }

# mindepth/maxdepth 2 -> MODELS_ROOT/<model>/<variant>
while IFS= read -r dir_path; do
  variant_name="$(basename "$dir_path")"
  model_root_dir="$(dirname "$dir_path")"
  model_dir="$(basename "$model_root_dir")"
  rel_dir="${dir_path#"$MODELS_ROOT"/}"

  # Optional filter: COMPILE_ONLY="01_MLP 05_AlexNet"
  if [[ -n "${COMPILE_ONLY:-}" && " ${COMPILE_ONLY} " != *" ${model_dir} "* ]]; then
    continue
  fi

  # Find .ll file(s) in this variant dir (not recursing further).
  mapfile -t ll_files < <(find "$dir_path" -maxdepth 1 -name '*.ll' | sort)
  if [[ ${#ll_files[@]} -eq 0 ]]; then
    continue
  fi
  if [[ ${#ll_files[@]} -gt 1 ]]; then
    echo "!! ${rel_dir}: expected exactly one .ll file, found ${#ll_files[@]} -- using the first"
  fi
  ll_file="${ll_files[0]}"

  # The driver gen_drivers.py wrote: <Name>_driver_{mim,mlir}.c
  mapfile -t c_files < <(find "$dir_path" -maxdepth 1 \
    \( -name '*_driver_mim.c' -o -name '*_driver_mlir.c' \) | sort)

  if [[ ${#c_files[@]} -eq 0 ]]; then
    echo "!! skipping ${rel_dir}: found .ll but no generated driver .c"
    echo "   (run gen_drivers.py against ${MODELS_ROOT} first)"
    FAILED+=("$rel_dir (missing driver)")
    continue
  fi
  if [[ ${#c_files[@]} -gt 1 ]]; then
    echo "!! ${rel_dir}: expected exactly one driver .c file, found ${#c_files[@]} -- using the first"
  fi
  c_file="${c_files[0]}"

  c_base="$(basename "$c_file")"
  model_name="${c_base%_driver_mim.c}"
  model_name="${model_name%_driver_mlir.c}"

  model_bin_dir="${model_root_dir}/bin"
  mkdir -p "$model_bin_dir"
  bin_name="${variant_name}"
  obj_file="${ll_file%.ll}.o"
  bc_file="${ll_file%.ll}.opt.bc"

  echo "--> [${model_dir}/${variant_name}] (${model_name})"

  # Middle-end (vectorization); see the header.
  codegen_input="$ll_file"
  opt_s=0
  if [[ "$RUN_OPT" == true ]]; then
    t0=$(now)
    echo "    opt:   $(basename "$ll_file") -> $(basename "$bc_file")"
    if ! opt "${OPT_ARGS[@]}" "$ll_file" -o "$bc_file"; then
      echo "!! opt failed for ${rel_dir}"
      FAILED+=("$rel_dir (opt)")
      rm -f "$bc_file"
      continue
    fi
    codegen_input="$bc_file"
    opt_s=$(elapsed "$t0" "$(now)")
  fi

  echo "    llc:   $(basename "$codegen_input") -> $(basename "$obj_file")"
  t0=$(now)
  if ! llc "${LLC_ARGS[@]}" -filetype=obj "$codegen_input" -o "$obj_file"; then
    echo "!! llc failed for ${rel_dir}"
    FAILED+=("$rel_dir (llc)")
    rm -f "$obj_file" "$bc_file"
    continue
  fi

  llc_s=$(elapsed "$t0" "$(now)")

  echo "    clang: $(basename "$c_file") + $(basename "$obj_file") -> ${model_dir}/bin/${bin_name}"
  t0=$(now)
  if ! clang "${CLANG_ARGS[@]}" "$c_file" "$obj_file" -o "${model_bin_dir}/${bin_name}" \
    -L"${LLVM_LIB_DIR}" \
    -lmlir_c_runner_utils -lmlir_runner_utils \
    -Wl,-rpath,"${LLVM_LIB_DIR}" \
    -lm; then
    echo "!! clang failed for ${rel_dir}"
    FAILED+=("$rel_dir (clang)")
    rm -f "$obj_file" "$bc_file"
    continue
  fi

  link_s=$(elapsed "$t0" "$(now)")
  printf '{"opt_s": %s, "llc_s": %s, "link_s": %s}\n' "$opt_s" "$llc_s" "$link_s" \
    > "${dir_path}/${model_name}_compile_timing.json"

  rm -f "$obj_file" "$bc_file"
  echo "    OK"
  echo
  BUILT=$((BUILT + 1))
done < <(find "$MODELS_ROOT" -mindepth 2 -maxdepth 2 -type d | sort)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

echo "==> Done. ${BUILT} binaries built (one bin/ dir per model under ${MODELS_ROOT})."
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "All discovered variants built successfully."
else
  echo "Some variants failed:"
  for f in "${FAILED[@]}"; do
    echo "  - $f"
  done
  exit 1
fi
