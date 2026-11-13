# KernelBench level-3 benchmark: MimIR vs torch-mlir

Compiles KernelBench level-3 models three ways and times them with one harness:

| variant          | frontend                         | to LLVM IR                               |
|------------------|----------------------------------|------------------------------------------|
| `mim_llvm`       | mimir-frontend, LLVM path        | MimIR emits `.ll` directly               |
| `mim_mlir_llvm`  | mimir-frontend, MLIR path        | `.mlir` -> `scripts/lower.sh` -> `.ll`   |
| `mlir_llvm`      | torch-mlir (`export_mlir.py`)    | `.mlir` -> `scripts/lower.sh` -> `.ll`   |

All three are compiled by the same `opt -O3` / `llc` / `clang` step and driven
by C drivers from `scripts/gen_drivers.py`: the same data loading, a forked
child per call, and `clock_gettime` around nothing but the model call. Only
the call's argument list differs (raw pointers vs MLIR memref descriptors).

`models.txt` holds the 36 level-3 models that compile on all three paths. The
LSTM/GRU models (36-42) are left out: torch-mlir unrolls their recurrence into
34k-68k ops, which its MLIR lowering does not finish.

## Setup

```bash
./setup.sh      # clone mimir-frontend next to this repo (../mimir-frontend) and build it
```

`setup.sh` clones [`mimir-frontend`](https://github.com/HarithF/mimir-frontend) (branch `art-eval`; `FRONTEND_URL`,
`FRONTEND_REF` and `FRONTEND` override it, and an existing checkout is reused)
and runs its `bootstrap_mimir.sh`, which builds MimIR with its plugins and the
Python environment. The LLVM/MLIR tools and a Python with torch-mlir have to be
provided separately (see the table below); the fv-docker image provides all of
it.

## Usage

```bash
./run.sh                    # export -> build -> report, every model in models.txt
./run.sh export             # 1. MimIR frontend (.ll + .mlir) and torch-mlir (.mlir)
./run.sh build              # 2. import, data, lower, drivers, compile, references
./run.sh report             # 3. run all variants, compare with PyTorch, timings
./run.sh all --only "1_MLP 44_MiniGPTBlock"      # a subset
./run.sh build --force                  # redo instead of skipping finished work
pipeline/build.sh lower                 # one sub-step: import|data|lower|drivers|compile|refs
scripts/clear.sh                        # delete everything under work/ for a clean run (asks first)
scripts/clear.sh reports -y             # or only: exports | build | reports
```

Settings live in `config.sh`; every one can be overridden from the environment
(`REPS=10 PIN_CPU=4 ./run.sh report`). Paths that must exist:

| variable          | what                                              | default                          |
|-------------------|---------------------------------------------------|----------------------------------|
| `FRONTEND`        | bootstrapped mimir-frontend checkout              | `../mimir-frontend`              |
| `TORCH_MLIR_PY`   | Python with `torch` + `torch_mlir`                | `python3` (`/opt/torch-venv/bin/python` in Docker) |
| `PYTHON`          | Python with `numpy` + `torch`                     | `python3`                        |
| `LLVM_LIB_DIR`    | dir with `libmlir_c_runner_utils.so`              | `lib/` next to the `mlir-opt` on PATH |
| PATH              | `mlir-opt`, `mlir-translate`, `opt`, `llc`, `clang` | |

Everything is written under `work/`:

```
work/mimir/<k>/            frontend sweep output; results/<k>.json = pass/fail per path
work/torch_mlir/<k>.mlir   torch-mlir export (+ <k>.params.txt argument names)
work/models/NN_<Model>/    benchmark tree: mim_llvm/ mim_mlir_llvm/ mlir_llvm/ bin/ torch_ref/
                           <Model>_{weights,input}.bin, _params.json, _ref.bin
work/reports/<run_id>/     reporter.py JSON per model, timing.csv, compile_times.{md,csv}, reporter.log
work/logs/                 per-model export and lowering logs, compile.log
```

## The stages

**1. export** (`pipeline/export.sh`)
- MimIR: `run_kernelbench_mimir.py` per model at KernelBench's own shapes
  (`--size-divisor 1` and an empty fixture overlay). It compiles the LLVM path,
  runs it and compares with eager PyTorch, and emits and verifies the MLIR.
- torch-mlir: `scripts/export_mlir.py`, which passes floating-point buffers as
  arguments and writes `<k>.params.txt` naming every argument.

**2. build** (`pipeline/build.sh`)
1. *import*: only models that passed on both MimIR paths and have a torch-mlir
   export (`ALLOW_UNVERIFIED=1` relaxes the first condition).
2. *data* (`scripts/gen_data.py`): random weights and input, seeded per
   model, plus `<Model>_params.json` naming every slot of `weights.bin`.
3. *lower* (`scripts/lower.sh`) both MLIR variants, `LOWER_JOBS` in parallel.
4. *drivers* (`scripts/gen_drivers.py`): every weight argument of every
   variant bound by name to its slot in `weights.bin`.
5. *compile* (`scripts/compile_all.sh`): `opt -O3` (host CPU), `llc`, `clang`.
6. *refs* (`scripts/produce_refs.py`): the PyTorch output, cached per model.

**3. report** (`pipeline/report.sh`): `reporter.py` per model (`--interleave`,
pinned to `PIN_CPU`), then `timing_to_csv.py`. A model passes when every
variant matches the PyTorch reference (rtol 1e-3, atol 1e-2).

It also writes `compile_times.md`/`.csv` (`scripts/compile_times.py`): per model
and path, the seconds of every compile step (the frontend (MimIR, or
torch-mlir's conversion to linalg), `mlir-opt`, `mlir-translate`, `opt`, `llc`
and the final link) plus `to_llvm_ir_s`, PyTorch graph to LLVM IR.
Tracing the model into an FX graph (`trace_s`: Dynamo for MimIR, `torch.export`
for torch-mlir) is a separate column, so `to_llvm_ir_s` compares the compilers
from the same starting point; `end_to_end_s` adds it back. The steps record
their times while running (`export_mlir.py`, `lower.sh`, `compile_all.sh`, the
frontend's manifests and its results JSON).

## Things worth knowing

- **Unrolled RNNs.** torch-mlir lowers `nn.LSTM`/`nn.GRU` through PyTorch's
  decompositions, which unroll every timestep (34k-68k `linalg.generic` ops;
  MimIR keeps an `scf.for`). `lower.sh` skips any torch-mlir file above 5000
  ops, because bufferization never finishes on them.
- **Memory.** MimIR's LLVM path frees no intermediates; the drivers fork per
  call, so this cannot accumulate across reps, but a single call of a large
  model needs tens of GB.

## Files

```
setup.sh               fetch and build mimir-frontend
run.sh                 the three stages
config.sh              all settings
models.txt             the model list (KernelBench stems)
pipeline/              export.sh, build.sh, report.sh, common.sh
scripts/               the tools the stages call
```
