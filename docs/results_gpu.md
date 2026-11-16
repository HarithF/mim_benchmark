# GPU results

> **Placeholder.** The final GPU numbers are still to be measured. This page will hold them
> in the same layout as [`results_cpu.md`](results_cpu.md), with the raw data in
> `results/gpu/` (`timing.csv`, `compile_times.csv`, and `summary.md` with the geometric means
> from `scripts/geomean.py`).

## Setup

| | |
|---|---|
| GPU | _TBD_ (model, memory, compute capability, driver) |
| Host CPU | _TBD_ |
| Measurement | 10 reps per variant (`REPS=10`), interleaved, one fresh process per call (CUDA setup, module load and weight upload are part of every timed call, for both variants), median reported |
| Correctness | every variant against the PyTorch output, rtol 1e-3, atol 1e-2 |

To reproduce: `REPS=10 ./run.sh --target gpu` (see [`artifact_evaluation.md`](artifact_evaluation.md)).

## Runtime

Median runtime per call; the ratio divides by the IREE baseline `iree_cuda`.

| model | `mim_nvptx` (ms) | `iree_cuda` (ms) | `mim_nvptx` / `iree_cuda` |
|---|---:|---:|---:|
| _TBD_ | | | |
| **geometric mean** | | | |

## Compile time

| variant | `trace_s` | `frontend_s` | `to_llvm_ir_s` | `opt_s` | `llc_s` | `link_s` | `total_s` | `end_to_end_s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mim_nvptx` | | | | | | | | |
| `iree_cuda` | | | | | | | | |

