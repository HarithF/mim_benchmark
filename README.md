# KernelBench level-3 benchmark: MimIR vs torch-mlir (CPU) and IREE (GPU)

Compiles KernelBench level-3 models several ways and times them with one
harness. `TARGET` (or `./run.sh --target`) picks the variants: `cpu` (default),
`gpu`, or `both`. On the CPU:

| variant          | frontend                         | to LLVM IR                               |
|------------------|----------------------------------|------------------------------------------|
| `mim_llvm`       | mimir-frontend, LLVM path        | MimIR emits `.ll` directly               |
| `mim_mlir_llvm`  | mimir-frontend, MLIR path        | `.mlir` -> `scripts/lower.sh` -> `.ll`   |
| `mlir_llvm`      | torch-mlir (`export_mlir.py`)    | `.mlir` -> `scripts/lower.sh` -> `.ll`   |

All three are compiled by the same `opt -O3` / `llc` / `clang` step and driven
by C drivers from `scripts/gen_drivers.py`: the same data loading, a forked
child per call, and `clock_gettime` around nothing but the model call. Only
the call's argument list differs (raw pointers vs MLIR memref descriptors).

On the GPU (`TARGET=gpu`):

| variant      | frontend                                 | to a GPU binary                                   |
|--------------|------------------------------------------|---------------------------------------------------|
| `mim_nvptx`  | mimir-frontend with `MIMIR_TARGET=nvptx` | host `.ll` with an embedded fatbin, same C driver |
| `iree_cuda`  | iree-turbine (`scripts/export_iree.py`)  | `iree-compile` for CUDA, `scripts/iree_driver.py` |

`mim_nvptx` needs a frontend with the nvptx target (`MIMIR_TARGET=nvptx`) and the CUDA
toolkit's `ptxas` and `fatbinary` on PATH. Both are timed like the CPU variants,
one fresh process per call: the timed call includes CUDA setup, module load and
the weight upload (MimIR's binary also copies data per op), as well as the call.

`models.txt` holds the 35 level-3 models that compile on all three CPU paths and
on the GPU. The RNN/LSTM/GRU models (33-42) are left out: they take more than one
input (the sequence plus the initial hidden/cell state), and the drivers only
support single-input models. MimIR can emit code for them. 30_SwinTransformerV2 is
left out because its runtime checks cannot be lowered on the GPU yet.

Results and the artifact-evaluation write-up (including what each compile-time step means) are in
[`docs/`](docs/README.md).

## Setup

```bash
./setup.sh      # clone mimir-frontend next to this repo (../mimir-frontend) and build it
```

`setup.sh` clones [`mimir-frontend`](https://github.com/HarithF/mimir-frontend) (branch `art-eval`; `FRONTEND_URL`,
`FRONTEND_REF` and `FRONTEND` override it, and an existing checkout is reused)
and runs its `bootstrap_mimir.sh`, which builds MimIR with its plugins and the
Python environment. The LLVM/MLIR tools and a Python with torch-mlir have to be
provided separately (see the table below); the [Docker image](https://doi.org/10.5281/zenodo.24303581) on Zenodo, which contains this
harness, provides all of it and can be used instead.

## Usage

```bash
./run.sh                    # export -> build -> report, every model in models.txt
./run.sh --target gpu       # the same for the GPU variants (work_gpu/); --target both: all five
./run.sh export             # 1. MimIR frontend (.ll + .mlir) and torch-mlir (.mlir)
./run.sh build              # 2. import, data, lower, drivers, compile, references
./run.sh report             # 3. run all variants, compare with PyTorch, timings
./run.sh all --only "1_MLP 44_MiniGPTBlock"      # a subset
./run.sh build --force                  # redo instead of skipping finished work
scripts/geomean.py work/reports/<run_id> --models models.txt   # geometric means over a model list
pipeline/build.sh lower                 # one sub-step: import|data|lower|drivers|compile|refs
scripts/clear.sh                        # delete everything under the target's work dir (asks first)
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
| GPU only: `GPU_SM`  | compute capability, e.g. `120`                  | from `nvidia-smi`                |
| GPU only: `CUDA_HOME` | CUDA toolkit (`lib64/stubs/libcuda.so` to link; `ptxas`, `fatbinary` on PATH for the export) | `/opt/cuda` |
| GPU only: `IREE_PY` | Python with `torch`, `iree-turbine`, `iree-base-compiler`, `iree-base-runtime` | `python3` |

Everything is written under `work/` (`work_gpu/` or `work_both/` for the other
targets, so their results never mix):

```
work/mimir/<k>/            frontend sweep output; results/<k>.json = pass/fail per path
work/torch_mlir/<k>.mlir   torch-mlir export (+ <k>.params.txt argument names)
work/mimir_gpu/<k>/        GPU frontend sweep (TARGET=gpu|both)
work/iree/<k>/             IREE export: <k>_cuda.vmfb + <k>.iree.json (TARGET=gpu|both)
work/models/NN_<Model>/    benchmark tree: mim_llvm/ mim_mlir_llvm/ mlir_llvm/ bin/ torch_ref/
                           <Model>_{weights,input}.bin, _params.json, _ref.bin
work/reports/<run_id>/     reporter.py JSON per model, timing.csv, compile_times.{md,csv}, summary.md, reporter.log
work/logs/                 per-model export and lowering logs, compile.log
```

## The stages

**1. export** (`pipeline/export.sh`)
- MimIR: `run_kernelbench_mimir.py` per model at KernelBench's own shapes
  (`--size-divisor 1` and an empty fixture overlay). It compiles the LLVM path,
  runs it and compares with eager PyTorch, and emits and verifies the MLIR.
- torch-mlir: `scripts/export_mlir.py`, which passes floating-point buffers as
  arguments and writes `<k>.params.txt` naming every argument.
- GPU: the same MimIR runner with `MIMIR_TARGET=nvptx`, and
  `scripts/export_iree.py`, which externalizes parameters by name so the IREE
  driver binds them from the same `weights.bin`.

**2. build** (`pipeline/build.sh`)
1. *import*: CPU: only models that passed on both MimIR paths and have a
   torch-mlir export (`ALLOW_UNVERIFIED=1` relaxes the first condition). GPU:
   `mim_nvptx` if its run passed and the fatbin is embedded, then `iree_cuda`.
2. *data* (`scripts/gen_data.py`): random weights and input, seeded per
   model, plus `<Model>_params.json` naming every slot of `weights.bin`. The
   layout comes from MimIR's `.ll` (`mim_llvm`, else `mim_nvptx`, which has the
   same signature), so CPU and GPU runs get identical data.
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

Last, `scripts/geomean.py` prints the geometric means and writes them to
`summary.md`. For run time, it takes each variant's median over the
reference's (`mlir_llvm` on the CPU, `iree_cuda` on the GPU) per model, and
averages over the models that have a median for every variant of the target;
the per-model ratios are in `summary.md` too. For compile time, it averages
every column of `compile_times.csv` per path, over the models that have the
column for every path. `--models FILE` restricts both to the models a file
lists, e.g. to compare a run of every model with a smaller selection.

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
