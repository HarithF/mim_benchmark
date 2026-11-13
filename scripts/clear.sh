#!/usr/bin/env bash
# clear.sh -- delete pipeline outputs under $WORK for a clean run.
#
#   scripts/clear.sh            everything under $WORK (asks first)
#   scripts/clear.sh -y         ... without asking
#   scripts/clear.sh build      only the build stage's outputs (models/, stage/)
#   scripts/clear.sh reports    only reports/
#   scripts/clear.sh exports    only the exports (mimir/, torch_mlir/, jit_cache/)
#
# The repository and mimir-frontend are never touched.
source "$(dirname "${BASH_SOURCE[0]}")/../pipeline/common.sh"

what="all"
assume_yes=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes) assume_yes=1 ;;
    all|exports|build|reports) what="$arg" ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) die "unknown argument: $arg (see --help)" ;;
  esac
done

work="$(realpath -m "$WORK")"
for forbidden in / "$HOME" "$BENCH_ROOT" "$(realpath -m "$FRONTEND")"; do
  [[ "$work" != "$(realpath -m "$forbidden")" ]] || die "refusing to clear WORK=$work"
done

case "$what" in
  all)     targets=("$work") ;;
  exports) targets=("$MIMIR_OUT" "$TORCH_OUT" "$work/jit_cache" "$work/no_fixtures" "$LOGS/export_mimir" "$LOGS/export_torch") ;;
  build)   targets=("$MODELS_DIR" "$work/stage" "$LOGS/lower" "$LOGS/compile.log") ;;
  reports) targets=("$REPORTS_DIR") ;;
esac

existing=()
for t in "${targets[@]}"; do [[ -e "$t" ]] && existing+=("$t"); done
if [[ ${#existing[@]} -eq 0 ]]; then
  echo "nothing to clear ($what under $work)"
  exit 0
fi

echo "will delete:"
for t in "${existing[@]}"; do echo "  $t  ($(du -sh "$t" 2>/dev/null | cut -f1))"; done
if [[ "$assume_yes" != 1 ]]; then
  read -r -p "continue? [y/N] " answer
  [[ "$answer" == [yY]* ]] || { echo "aborted"; exit 1; }
fi
for t in "${existing[@]}"; do rm -rf -- "${t:?}"; done
echo "cleared ($what)"
