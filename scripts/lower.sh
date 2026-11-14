#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 [--baseline] <dir> [<dir> ...]

Each <dir> is a variant directory (e.g. 01_MLP/mlir_llvm) or a models root;
it is searched for directories holding one source .mlir (not *_lowered.mlir).
Each is lowered to <dir>/<Model>_lowered.mlir and <dir>/<Model>_<dir>.ll,
where <Model> is the parent directory without its "NN_" prefix.

Modes:
  (default)    transform schedule: convolutions get the output-width loop
               innermost, matmuls are tiled 32x32x32; then bufferize and
               lower to the LLVM dialect.
  --baseline   global --affine-loop-tile + --affine-loop-fusion instead of the
               schedule, for comparison only.

Env:
  LOWER_SH_DEBUG_PRINT_PASSES=1   print the mlir-opt pass list per dir
  LOWER_SH_FUSE=1                 add --linalg-fuse-elementwise-ops (slower;
                                  see the comment at that pass)
EOF
  exit 1
}

# Why this pipeline: lowered as is, a convolution's reductions are innermost,
# leaving a scalar accumulator in memory that LLVM cannot vectorize. Moving the
# output-width loop innermost makes the inner loop elementwise and unit-stride,
# so LLVM's loop vectorizer handles it (VGG16: ~13x). The reductions keep their
# order, so results are bit-identical. Matmuls are tiled 32^3 instead, inside
# the schedule, since global affine tiling slows the interchanged convolutions.
# --convert-linalg-to-loops is used because the tiled loops have min() bounds
# that the affine lowering rejects.

# MODE: "fast" (default) | "baseline" (--baseline)
MODE="fast"
dirs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
  --baseline | --tile)
    MODE="baseline"
    shift
    ;;
  --max)
    echo "note: --max is the default mode; ignoring the flag." >&2
    shift
    ;;
  -h | --help) usage ;;
  --)
    shift
    break
    ;;
  -*)
    echo "unknown flag: $1" >&2
    usage
    ;;
  *)
    dirs+=("$1")
    shift
    ;;
  esac
done
dirs+=("$@")
[[ ${#dirs[@]} -gt 0 ]] || usage

# Expand each argument into every directory below it that holds a source .mlir.
input_args=("${dirs[@]}")
expanded_dirs=()
for d in "${input_args[@]}"; do
  d=${d%/}
  if [[ ! -d "$d" ]]; then
    echo "error: '$d' is not a directory, skipping" >&2
    continue
  fi
  while IFS= read -r -d '' sub; do
    match="$(find "$sub" -maxdepth 1 -name '*.mlir' ! -name '*_lowered.mlir' -print -quit)"
    if [[ -n "$match" ]]; then
      expanded_dirs+=("$sub")
    fi
  done < <(find "$d" -type d -print0)
done

# Deduplicate, keeping discovery order.
dirs=()
declare -A seen=()
for d in "${expanded_dirs[@]}"; do
  if [[ -z "${seen[$d]:-}" ]]; then
    seen["$d"]=1
    dirs+=("$d")
  fi
done

if [[ ${#dirs[@]} -eq 0 ]]; then
  echo "error: no directory containing a source .mlir file found under: ${input_args[*]}" >&2
  exit 1
fi

echo "==> found ${#dirs[@]} director$([[ ${#dirs[@]} -eq 1 ]] && echo y || echo ies) to lower:" >&2
printf '      %s\n' "${dirs[@]}" >&2
echo >&2

BUF='allow-return-allocs-from-loops bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map'

# Transform schedule (default mode), written to a temp file. Ops are matched by
# structure (rank + mulf/addf contraction body), not by name, so it applies to
# both frontends: torch-mlir's named ops are generalized first, and MimIR emits
# only linalg.generic. Pooling has no mulf/addf body and is left alone.
TRANSFORM_FILE=""
if [[ "$MODE" == "fast" ]]; then
  TRANSFORM_FILE="$(mktemp --suffix=.mlir)"
  trap 'rm -f "$TRANSFORM_FILE"' EXIT
  cat >"$TRANSFORM_FILE" <<'EOF'
module attributes {transform.with_named_sequence} {

  // ---- actions ---------------------------------------------------------

  // NCHW convolution, rank 7: (n, f, oh, ow, c, kh, kw)
  //                        -> (n, f, oh, c, kh, kw, ow)
  // Moves output-width innermost. Reduction dims keep their relative order,
  // so the result is bit-identical.
  transform.named_sequence @conv_nchw_ow_innermost(%op: !transform.any_op {transform.consumed}) {
    transform.structured.interchange %op iterator_interchange = [0, 1, 2, 4, 5, 6, 3]
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  // Grouped NCHW convolution, rank 8: (n, g, fg, oh, ow, c, kh, kw)
  // Same idea; `ow` is at position 4 here.
  transform.named_sequence @conv_ngchw_ow_innermost(%op: !transform.any_op {transform.consumed}) {
    transform.structured.interchange %op iterator_interchange = [0, 1, 2, 3, 5, 6, 7, 4]
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  // Depthwise NCHW convolution, rank 6: (n, oh, ow, c, kh, kw)
  // `ow` is at position 2.
  transform.named_sequence @conv_depthwise_ow_innermost(%op: !transform.any_op {transform.consumed}) {
    transform.structured.interchange %op iterator_interchange = [0, 1, 3, 4, 5, 2]
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  // Matmul, rank 3: (m, n, k), tiled 32^3.
  transform.named_sequence @tile_matmul(%op: !transform.any_op {transform.consumed}) {
    %t, %l0, %l1, %l2 = transform.structured.tile_using_for %op tile_sizes [32, 32, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }

  // Batch matmul, rank 4: (b, m, n, k). Batch left whole.
  transform.named_sequence @tile_batch_matmul(%op: !transform.any_op {transform.consumed}) {
    %t, %l0, %l1, %l2, %l3 = transform.structured.tile_using_for %op tile_sizes [1, 32, 32, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }

  // mim's convolution, rank 10: the 7 conv loops with co, oh and ci each
  // split in two. Whatever order mim picks for the co/oh blocks, `ow` is
  // loop 5 -- the last parallel one -- and the four reductions follow it.
  transform.named_sequence @conv_mim_ow_innermost(%op: !transform.any_op {transform.consumed}) {
    transform.structured.interchange %op iterator_interchange = [0, 1, 2, 3, 4, 6, 7, 8, 9, 5]
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  // mim's batched matmul over a split reduction, rank 5: (b, m, n, k_outer,
  // k_inner). Move `n` innermost, then tile 32^3 like torch-mlir's batch
  // matmul, since the right operand may be strided along n.
  transform.named_sequence @bmm_mim_n_innermost(%op: !transform.any_op {transform.consumed}) {
    %i = transform.structured.interchange %op iterator_interchange = [0, 1, 3, 4, 2]
        : (!transform.any_op) -> !transform.any_op
    // Loops are now (b, m, k_outer, k_inner, n); k_inner is 4 (1 if k % 4 != 0).
    %t, %l0, %l1, %l2, %l3 = transform.structured.tile_using_for %i tile_sizes [1, 32, 8, 0, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }

  // ---- matchers --------------------------------------------------------
  // rank == N and a mulf/addf contraction body (excludes pooling and
  // elementwise ops).

  transform.named_sequence @match_rank3(%c0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.structured %c0 : !transform.any_op {
    ^bb0(%c: !transform.any_op):
      %rank = transform.match.structured.rank %c : (!transform.any_op) -> !transform.param<i64>
      %n = transform.param.constant 3 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %n : !transform.param<i64>
      transform.match.structured.body %c { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op
      transform.match.structured.yield
    }
    transform.yield %c0 : !transform.any_op
  }

  transform.named_sequence @match_rank4(%c0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.structured %c0 : !transform.any_op {
    ^bb0(%c: !transform.any_op):
      %rank = transform.match.structured.rank %c : (!transform.any_op) -> !transform.param<i64>
      %n = transform.param.constant 4 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %n : !transform.param<i64>
      transform.match.structured.body %c { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op
      transform.match.structured.yield
    }
    transform.yield %c0 : !transform.any_op
  }

  transform.named_sequence @match_rank6(%c0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.structured %c0 : !transform.any_op {
    ^bb0(%c: !transform.any_op):
      %rank = transform.match.structured.rank %c : (!transform.any_op) -> !transform.param<i64>
      %n = transform.param.constant 6 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %n : !transform.param<i64>
      transform.match.structured.dim %c[4] { reduction } : !transform.any_op
      transform.match.structured.body %c { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op
      transform.match.structured.yield
    }
    transform.yield %c0 : !transform.any_op
  }

  // mim's batched matmul with its reduction split in two: rank 5, the
  // shape-only third input, and loops 3 and 4 both reductions.
  transform.named_sequence @match_rank5_bmm(%c0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.structured %c0 : !transform.any_op {
    ^bb0(%c: !transform.any_op):
      %rank = transform.match.structured.rank %c : (!transform.any_op) -> !transform.param<i64>
      %n = transform.param.constant 5 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %n : !transform.param<i64>
      %nin = transform.match.structured.num_inputs %c : (!transform.any_op) -> !transform.param<i64>
      %three = transform.param.constant 3 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %nin, %three : !transform.param<i64>
      transform.match.structured.dim %c[3] { reduction } : !transform.any_op
      transform.match.structured.yield
    }
    transform.yield %c0 : !transform.any_op
  }

  // mim's convolution: rank 10 with the shape-only third input.
  transform.named_sequence @match_rank10_3in(%c0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.structured %c0 : !transform.any_op {
    ^bb0(%c: !transform.any_op):
      %rank = transform.match.structured.rank %c : (!transform.any_op) -> !transform.param<i64>
      %n = transform.param.constant 10 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %n : !transform.param<i64>
      %nin = transform.match.structured.num_inputs %c : (!transform.any_op) -> !transform.param<i64>
      %three = transform.param.constant 3 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %nin, %three : !transform.param<i64>
      transform.match.structured.yield
    }
    transform.yield %c0 : !transform.any_op
  }

  transform.named_sequence @match_rank7(%c0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.structured %c0 : !transform.any_op {
    ^bb0(%c: !transform.any_op):
      %rank = transform.match.structured.rank %c : (!transform.any_op) -> !transform.param<i64>
      %n = transform.param.constant 7 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %n : !transform.param<i64>
      transform.match.structured.body %c { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op
      transform.match.structured.yield
    }
    transform.yield %c0 : !transform.any_op
  }

  transform.named_sequence @match_rank8(%c0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.structured %c0 : !transform.any_op {
    ^bb0(%c: !transform.any_op):
      %rank = transform.match.structured.rank %c : (!transform.any_op) -> !transform.param<i64>
      %n = transform.param.constant 8 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %n : !transform.param<i64>
      transform.match.structured.body %c { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op
      transform.match.structured.yield
    }
    transform.yield %c0 : !transform.any_op
  }

  // mim's fully-connected contractions are left alone: mim already makes the
  // contraction the unit-stride loop (marked reassoc), which LLVM vectorizes;
  // tiling it would split that loop.

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.foreach_match in %root
      @match_rank7 -> @conv_nchw_ow_innermost,
      @match_rank8 -> @conv_ngchw_ow_innermost,
      @match_rank6 -> @conv_depthwise_ow_innermost,
      @match_rank10_3in -> @conv_mim_ow_innermost,
      @match_rank5_bmm -> @bmm_mim_n_innermost,
      @match_rank3 -> @tile_matmul,
      @match_rank4 -> @tile_batch_matmul
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
EOF
  # Loaded with --transform-preload-library, run with --transform-interpreter.
fi

FAILED=()
SKIPPED=()
LOWERED=0

# Wall-clock seconds of the steps of one directory, written to
# <dir>/<Model>_lower_timing.json for the compile-time report.
now() { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f", b - a }'; }

for dir in "${dirs[@]}"; do
  dir=${dir%/}

  if [[ ! -d "$dir" ]]; then
    echo "error: '$dir' is not a directory, skipping" >&2
    FAILED+=("$dir (not a directory)")
    continue
  fi

  # Source: the one *.mlir in this dir that is not a lowered output.
  candidates=()
  while IFS= read -r -d '' f; do
    candidates+=("$f")
  done < <(find "$dir" -maxdepth 1 -name '*.mlir' ! -name '*_lowered.mlir' -print0)

  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "warn: no source .mlir found in '$dir', skipping" >&2
    continue
  fi
  if [[ ${#candidates[@]} -gt 1 ]]; then
    echo "error: multiple candidate source .mlir files in '$dir':" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    echo "  refusing to guess -- rename/remove extras so only one remains" >&2
    FAILED+=("$dir (multiple source .mlir candidates)")
    continue
  fi

  IN="${candidates[0]}"

  # 01_MLP/mlir_llvm -> MLP
  model_dir="$(basename "$(dirname "$dir")")"
  model="${model_dir#*_}"
  [[ "$model" == "$model_dir" ]] && model="$model_dir"

  OUT="$dir/${model}_lowered.mlir"

  # torch-mlir unrolls RNN timesteps (34k-68k linalg.generic ops; the largest
  # other model has ~830), and bufferization never finishes on those: skip.
  if [[ "$(basename "$dir")" == "mlir_llvm" ]]; then
    nops="$(grep -c 'linalg.generic' "$IN" || true)"  # grep -c exits 1 on 0 matches
    if (( nops > 5000 )); then
      echo "!! skipping '$dir': torch-mlir unrolled this RNN to ${nops} linalg.generic ops (>5000); bufferize never finishes on it. The looped mim_mlir_llvm variant covers this model." >&2
      rm -f "$OUT" "$dir/${model}_mlir_llvm.ll"
      SKIPPED+=("$dir (unrolled RNN: ${nops} linalg.generic ops)")
      continue
    fi
  fi

  PASSES=()

  if [[ "$MODE" == "fast" ]]; then
    # Named ops -> linalg.generic, so the schedule sees both frontends alike.
    PASSES+=(--linalg-generalize-named-ops --canonicalize --cse)

    # Apply the schedule on tensors, before bufferization.
    PASSES+=(
      --transform-preload-library="transform-library-paths=$TRANSFORM_FILE"
      --transform-interpreter="entry-point=__transform_main"
      --canonicalize --cse
    )

    # --linalg-fuse-elementwise-ops is off by default: it fuses elementwise
    # producers into consumers with reduction loops, recomputing them per
    # iteration (AlexNet 1.66x slower). If enabled, it must run after the
    # schedule, or the matmuls no longer match it.
    if [[ -n "${LOWER_SH_FUSE:-}" ]]; then
      PASSES+=(--linalg-fuse-elementwise-ops --canonicalize --cse)
    fi
  fi

  PASSES+=(
    --one-shot-bufferize="$BUF"
    --buffer-deallocation-pipeline
    # Otherwise bufferization.clone reaches mlir-translate, which rejects it.
    --convert-bufferization-to-memref
  )

  if [[ "$MODE" == "baseline" ]]; then
    PASSES+=(
      --convert-linalg-to-affine-loops
      --affine-loop-tile=tile-size=32 --affine-loop-fusion
    )
  else
    # scf, not affine: the tiled loops have min() bounds.
    PASSES+=(--convert-linalg-to-loops)
  fi

  PASSES+=(
    --affine-scalrep
    --affine-loop-invariant-code-motion
    --lower-affine
    --convert-scf-to-cf
    --expand-strided-metadata
    # Again: --expand-strided-metadata emits affine.apply for tiled subviews.
    --lower-affine
    --convert-scf-to-cf
    --convert-vector-to-llvm
    --convert-ub-to-llvm --convert-cf-to-llvm
    --convert-arith-to-llvm
    # aligned_alloc instead of malloc + inttoptr alignment, which hides the
    # pointer from LLVM's alias analysis.
    --finalize-memref-to-llvm=use-aligned-alloc
    --convert-func-to-llvm
    --convert-index-to-llvm --convert-math-to-llvm
    --reconcile-unrealized-casts
  )

  variant_note=""
  [[ "$MODE" == "baseline" ]] && variant_note=" (--baseline, for comparison only)"
  echo "==> [$dir] $(basename "$IN") -> $(basename "$OUT")$variant_note"

  if [[ -n "${LOWER_SH_DEBUG_PRINT_PASSES:-}" ]]; then
    echo "---- exact PASSES array for $dir (MODE=$MODE) ----" >&2
    printf '%q ' "${PASSES[@]}" >&2
    echo >&2
    echo "----------------------------------------------------" >&2
  fi

  libm_s=0
  t0=$(now)
  if ! mlir-opt "$IN" "${PASSES[@]}" -o "$OUT"; then
    echo "!! mlir-opt failed for '$dir' -- skipping (leaving other directories to run)" >&2
    FAILED+=("$dir (mlir-opt)")
    rm -f "$OUT"
    continue
  fi

  mlir_opt_s=$(elapsed "$t0" "$(now)")

  # Math ops without an LLVM intrinsic (e.g. math.erf from GELU) become libm
  # calls; only for files that have them.
  if grep -q '= math\.' "$OUT"; then
    t0=$(now)
    echo "==> [$dir] leftover math ops -> libm calls"
    if ! mlir-opt "$OUT" --convert-math-to-libm --convert-func-to-llvm         --reconcile-unrealized-casts -o "$OUT.libm"; then
      echo "!! mlir-opt (math-to-libm) failed for '$dir' -- skipping" >&2
      FAILED+=("$dir (math-to-libm)")
      rm -f "$OUT.libm"
      continue
    fi
    mv "$OUT.libm" "$OUT"
    libm_s=$(elapsed "$t0" "$(now)")
  fi

  # 01_MLP/mlir_llvm -> 01_MLP/mlir_llvm/MLP_mlir_llvm.ll
  base="$(basename "$dir")"
  LL="$dir/${model}_${base}.ll"
  echo "==> [$dir] $(basename "$OUT") -> $(basename "$LL")"
  t0=$(now)
  if ! mlir-translate --mlir-to-llvmir "$OUT" -o "$LL"; then
    echo "!! mlir-translate failed for '$dir' -- skipping (leaving other directories to run)" >&2
    FAILED+=("$dir (mlir-translate)")
    rm -f "$LL"
    continue
  fi

  translate_s=$(elapsed "$t0" "$(now)")
  printf '{"mlir_opt_s": %s, "libm_s": %s, "mlir_translate_s": %s}\n' \
    "$mlir_opt_s" "$libm_s" "$translate_s" > "$dir/${model}_lower_timing.json"

  LOWERED=$((LOWERED + 1))
done

echo
echo "==> done. ${LOWERED}/${#dirs[@]} lowered successfully."
if [[ "$MODE" == "fast" ]]; then
  echo "    Compile the .ll with opt -O3 (compile_all.sh): the vectorization happens there."
fi
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  echo "Skipped ${#SKIPPED[@]} (unrolled torch-mlir RNN; covered by mim_mlir_llvm):"
  for s in "${SKIPPED[@]}"; do
    echo "  - $s"
  done
  echo
fi
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "All discovered directories lowered successfully."
else
  echo "Some directories failed:"
  for f in "${FAILED[@]}"; do
    echo "  - $f"
  done
  exit 1
fi
