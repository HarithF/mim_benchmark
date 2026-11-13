#!/usr/bin/env bash
# Fetch mimir-frontend next to this repo and build it (MimIR, its plugins and
# the Python environment), so that FRONTEND's default (../mimir-frontend) works.
#
#   ./setup.sh                                  clone (if missing) + bootstrap
#   FRONTEND_REF=main ./setup.sh                another branch or tag
#   FRONTEND=/path/to/checkout ./setup.sh       use / create a checkout elsewhere
#
# Needs git, uv, cmake and a C++23 compiler (see mimir-frontend's README).
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="${FRONTEND:-$(cd "$BENCH_ROOT/.." && pwd)/mimir-frontend}"
FRONTEND_URL="${FRONTEND_URL:-https://github.com/HarithF/mimir-frontend.git}"
FRONTEND_REF="${FRONTEND_REF:-art-eval}"

if [[ -d "$FRONTEND/.git" ]]; then
  echo "==> using existing checkout $FRONTEND ($(git -C "$FRONTEND" rev-parse --abbrev-ref HEAD))"
else
  echo "==> cloning $FRONTEND_URL ($FRONTEND_REF) -> $FRONTEND"
  git clone --recursive --branch "$FRONTEND_REF" "$FRONTEND_URL" "$FRONTEND"
fi

echo "==> bootstrapping mimir-frontend"
cd "$FRONTEND"
./scripts/bootstrap_mimir.sh
# bootstrap builds the `mim` Python binding; build the rest of MimIR too, so the
# MLIR emitter plugin is present.
cmake --build MimIR/build -j "${MIMIR_BUILD_JOBS:-8}"
uv run --no-sync python -c "import mim, mimir_frontend.backend; print('mimir-frontend ready')"

if [[ "$FRONTEND" != "$(cd "$BENCH_ROOT/.." && pwd)/mimir-frontend" ]]; then
  echo "note: export FRONTEND=$FRONTEND before running ./run.sh"
fi
