# Pipeline configuration. Every value can be overridden from the environment,
# e.g. `REPS=10 ./run.sh report`.

# --- inputs ------------------------------------------------------------------
# mimir-frontend checkout (bootstrapped: `uv run --no-sync` must work there).
# Default: the sibling directory, as in rebuttal/ and in the fv-docker image.
FRONTEND="${FRONTEND:-$(cd "$BENCH_ROOT/.." && pwd)/mimir-frontend-nightly}"
# KernelBench level-3 sources; torch-mlir exports and torch references use these.
KERNELBENCH_DIR="${KERNELBENCH_DIR:-$FRONTEND/KernelBench/KernelBench/level3}"
# Python with torch + torch_mlir (export only).
TORCH_MLIR_PY="${TORCH_MLIR_PY:-python3}"
# Python with numpy + torch (data generation, drivers, references, reporter).
PYTHON="${PYTHON:-python3}"
# mlir-opt, mlir-translate, opt, llc and clang are taken from PATH. LLVM_LIB_DIR
# holds libmlir_c_runner_utils (compile_all.sh); default: lib/ next to the
# mlir-opt on PATH.
LLVM_LIB_DIR="${LLVM_LIB_DIR:-$(cd "$(dirname "$(command -v mlir-opt || echo .)")/../lib" 2>/dev/null && pwd)}"

# Models to run, one KernelBench stem per line (e.g. 44_MiniGPTBlock).
MODELS_FILE="${MODELS_FILE:-$BENCH_ROOT/models.txt}"

# --- target ----------------------------------------------------------------------
# cpu (default): mim_llvm, mim_mlir_llvm, mlir_llvm (torch-mlir baseline).
# gpu:  mim_nvptx (MimIR -> NVPTX, host code + embedded fatbin), iree_cuda (baseline).
# both: all five.
TARGET="${TARGET:-cpu}"
# GPU only: compute capability for the device code, e.g. 120; empty = ask nvidia-smi.
GPU_SM="${GPU_SM:-}"
# GPU only: libcuda link stub (the driver's libcuda is loaded at run time).
CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
# GPU only: Python with torch + iree-turbine + iree-base-{compiler,runtime}.
IREE_PY="${IREE_PY:-python3}"
IREE_TIMEOUT="${IREE_TIMEOUT:-1800}"

# --- outputs -------------------------------------------------------------------
# One work dir per target, so CPU and GPU results never mix.
if [[ "$TARGET" == cpu ]]; then WORK="${WORK:-$BENCH_ROOT/work}"; else WORK="${WORK:-$BENCH_ROOT/work_$TARGET}"; fi
MIMIR_OUT="${MIMIR_OUT:-$WORK/mimir}"          # MimIR CPU export: <k>/mimir_graph_*.{ll,mlir}
TORCH_OUT="${TORCH_OUT:-$WORK/torch_mlir}"     # torch-mlir export: <k>.mlir + <k>.params.txt
MIMIR_GPU_OUT="${MIMIR_GPU_OUT:-$WORK/mimir_gpu}" # MimIR GPU export: <k>/mimir_graph_*.ll (+ _dev.ll)
IREE_OUT="${IREE_OUT:-$WORK/iree}"             # IREE export: <k>/<k>_cuda.vmfb + <k>.iree.json
MODELS_DIR="${MODELS_DIR:-$WORK/models}"       # benchmark tree: NN_<Model>/{<variant>/,bin/,...}
REPORTS_DIR="${REPORTS_DIR:-$WORK/reports}"
LOGS="${LOGS:-$WORK/logs}"

# --- 1. export -------------------------------------------------------------------
# MimIR frontend at KernelBench's native sizes (no fixture overlay, divisor 1).
MIMIR_TIMEOUT="${MIMIR_TIMEOUT:-3600}" # seconds per model
MIMIR_MAX_MEMORY_GB="${MIMIR_MAX_MEMORY_GB:-40}"
MIMIR_MAX_FP_ITERS="${MIMIR_MAX_FP_ITERS:-8192}"
TORCH_EXPORT_TIMEOUT="${TORCH_EXPORT_TIMEOUT:-1800}"

# --- 2. build --------------------------------------------------------------------
DATA_SEED="${DATA_SEED:-1}"
LOWER_JOBS="${LOWER_JOBS:-4}"       # lower.sh runs in parallel, one variant dir per job
LOWER_TIMEOUT="${LOWER_TIMEOUT:-0}" # seconds per variant dir; 0 = no limit
OPT_LEVEL="${OPT_LEVEL:-O3}"
# Build only models whose frontend run passed on BOTH MimIR paths (results JSON).
# Set to 1 to import whatever .ll/.mlir the sweep left behind.
ALLOW_UNVERIFIED="${ALLOW_UNVERIFIED:-0}"

# --- 3. report -------------------------------------------------------------------
REPS="${REPS:-5}"
PIN_CPU="${PIN_CPU:-2}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}" # seconds per variant run
RUN_ID="${RUN_ID:-}"               # default: timestamp
