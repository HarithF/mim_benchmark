# GPU results

Sweep of the benchmark models on the GPU (`./run.sh --target gpu --force`), report run
`gpuconcat`. The raw data is in [`results/gpu/`](results/gpu/): `timing.csv` (per-model
median/min/max per variant), `compile_times.csv` (every compile step per model and variant) and
`summary.md` (the geometric means, from `scripts/geomean.py`).
How the numbers are measured is described in [`artifact_evaluation.md`](artifact_evaluation.md).

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti (16 GB, compute capability 8.9, `sm_89`), driver 615.78.08 |
| Host | Intel Core i5-10600KF, 12 CPUs, 31 GiB; runs pinned to core 2; governor `schedutil` (frequency not pinned); memory limited to 24 GB, no swap (`systemd-run -p MemoryMax=24G -p MemorySwapMax=0`) |
| Software | mimir-frontend branch `concat-gpu` (`61e5909`), MimIR `6a1b4fa`; IREE 3.14.0, iree-turbine 3.9.0; PyTorch 2.15.0 (CPU build) |
| Measurement | 5 reps per variant, interleaved, one fresh process per call (CUDA setup, module load and weight upload are part of every timed call, for both variants), median reported |
| Correctness | every variant against the PyTorch output, rtol 1e-3, atol 1e-2 |

## Runtime

33 models were measured. Of the 35 models in `models.txt`, `31_VisionAttention` is left out
(its 8.6 GB attention matrix does not fit this GPU's 16 GB, for `mim_nvptx` and for `iree_cuda`)
and `10_ResNet101` is left out because its `iree_cuda` output fails the correctness check
(max diff / scale 2.8; `mim_nvptx` passes); with it, the geometric mean is 1.22 (34 models).
Both variants pass the correctness check on all 33 models. Median runtime per call;
the ratio column divides by the IREE baseline `iree_cuda` (below 1 = faster than IREE).

| model | `mim_nvptx` (ms) | `iree_cuda` (ms) | `mim_nvptx` / `iree_cuda` |
|---|---:|---:|---:|
| 01_MLP | 702.0 | 822.4 | 0.85 |
| 02_ShallowWideMLP | 1968.2 | 2335.0 | 0.84 |
| 03_DeepNarrowMLP | 216.5 | 280.4 | 0.77 |
| 04_LeNet5 | 132.1 | 138.5 | 0.95 |
| 05_AlexNet | 7213.8 | 12251.6 | 0.59 |
| 06_GoogleNetInceptionModule | 2009.8 | 2272.5 | 0.88 |
| 07_GoogleNetInceptionV1 | 298.1 | 204.0 | 1.46 |
| 08_ResNetBasicBlock | 184.3 | 201.6 | 0.91 |
| 09_ResNet18 | 167.2 | 163.0 | 1.03 |
| 11_VGG16 | 1110.2 | 509.3 | 2.18 |
| 12_VGG19 | 1381.0 | 582.0 | 2.37 |
| 13_DenseNet121TransitionLayer | 463.1 | 749.9 | 0.62 |
| 14_DenseNet121DenseBlock | 810.3 | 453.8 | 1.79 |
| 15_DenseNet121 | 526.6 | 268.8 | 1.96 |
| 16_DenseNet201 | 994.0 | 474.3 | 2.10 |
| 18_SqueezeNet | 1924.0 | 2281.4 | 0.84 |
| 19_MobileNetV1 | 185.7 | 175.7 | 1.06 |
| 20_MobileNetV2 | 173.5 | 168.8 | 1.03 |
| 21_EfficientNetMBConv | 694.4 | 1067.9 | 0.65 |
| 22_EfficientNetB0 | 178.0 | 169.3 | 1.05 |
| 23_EfficientNetB1 | 157.0 | 157.5 | 1.00 |
| 24_EfficientNetB2 | 130.7 | 142.6 | 0.92 |
| 25_ShuffleNetUnit | 1206.0 | 655.8 | 1.84 |
| 26_ShuffleNet | 1297.8 | 839.1 | 1.55 |
| 27_RegNet | 413.5 | 224.4 | 1.84 |
| 28_VisionTransformer | 237.1 | 185.9 | 1.28 |
| 29_SwinMLP | 573.8 | 255.4 | 2.25 |
| 43_MinGPTCausalAttention | 1965.7 | 793.1 | 2.48 |
| 44_MiniGPTBlock | 3719.7 | 1656.7 | 2.25 |
| 45_UNetSoftmax | 932.9 | 500.5 | 1.86 |
| 46_NetVladWithGhostClusters | 508.9 | 341.9 | 1.49 |
| 47_NetVladNoGhostClusters | 498.3 | 341.5 | 1.46 |
| 50_ReLUSelfAttention | 636.4 | 295.3 | 2.15 |
| **geometric mean** | | | **1.28** |

The geometric mean is that of `scripts/geomean.py`: per model the `mim_nvptx` median over
`iree_cuda`'s, averaged over the 33 models that have both variants.

`mim_nvptx` is slower than IREE in the geometric mean (1.28x) and faster on 12 of 33 models;
the ratios range from 0.59 (`05_AlexNet`) to 2.48 (`43_MinGPTCausalAttention`).

## Compile time

Seconds per model and variant; the steps are defined in
[`artifact_evaluation.md`](artifact_evaluation.md#e2-compile-time). For `mim_nvptx`,
`to_llvm_ir_s` is PyTorch graph to LLVM IR (equal to `frontend_s`), `total_s` adds `opt`, `llc`
and the link, `end_to_end_s` adds tracing the model. `iree_cuda` reports only `frontend_s` and
`total_s`; it has no separate LLVM steps and no separate trace time, so its `end_to_end_s`
equals its `total_s`.

Geometric mean per step over all 33 models, in seconds (`scripts/geomean.py` for the columns
both variants have; the `mim_nvptx`-only rows are computed the same way from
`compile_times.csv`):

| column | models | `mim_nvptx` | `iree_cuda` |
|---|---:|---:|---:|
| `trace_s` | 33 | 0.22 |  |
| `frontend_s` | 33 | 0.64 | 1.43 |
| `to_llvm_ir_s` | 33 | 0.64 |  |
| `opt_s` | 33 | 0.05 |  |
| `llc_s` | 33 | 0.11 |  |
| `link_s` | 33 | 0.11 |  |
| `total_s` | 33 | 0.98 | 2.75 |
| `end_to_end_s` | 33 | 1.22 | 2.75 |

Sums over all 33 models:

| variant | `trace_s` | `frontend_s` | `to_llvm_ir_s` | `opt_s` | `llc_s` | `link_s` | `total_s` | `end_to_end_s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mim_nvptx` | 12.5 | 45.9 | 45.9 | 8.3 | 9.8 | 3.6 | 67.6 | 80.0 |
| `iree_cuda` |  | 63.9 |  |  |  |  | 138.9 | 138.9 |

Per model (the full table with every step is `results/gpu/compile_times.csv`):

| model | variant | `trace_s` | `frontend_s` | `opt_s` | `llc_s` | `link_s` | `total_s` | `end_to_end_s` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01_MLP | `mim_nvptx` | 0.07 | 0.18 | 0.01 | 0.02 | 0.10 | 0.32 | 0.39 |
| 01_MLP | `iree_cuda` |  | 1.86 |  |  |  | 2.86 | 2.86 |
| 02_ShallowWideMLP | `mim_nvptx` | 0.06 | 0.10 | 0.01 | 0.02 | 0.10 | 0.23 | 0.29 |
| 02_ShallowWideMLP | `iree_cuda` |  | 4.46 |  |  |  | 5.30 | 5.30 |
| 03_DeepNarrowMLP | `mim_nvptx` | 0.12 | 0.27 | 0.03 | 0.06 | 0.10 | 0.45 | 0.56 |
| 03_DeepNarrowMLP | `iree_cuda` |  | 0.85 |  |  |  | 1.77 | 1.77 |
| 04_LeNet5 | `mim_nvptx` | 0.08 | 0.16 | 0.03 | 0.04 | 0.10 | 0.33 | 0.41 |
| 04_LeNet5 | `iree_cuda` |  | 0.57 |  |  |  | 1.04 | 1.04 |
| 05_AlexNet | `mim_nvptx` | 0.11 | 0.27 | 0.02 | 0.04 | 0.10 | 0.43 | 0.55 |
| 05_AlexNet | `iree_cuda` |  | 0.92 |  |  |  | 1.74 | 1.74 |
| 06_GoogleNetInceptionModule | `mim_nvptx` | 0.08 | 0.22 | 0.02 | 0.04 | 0.10 | 0.38 | 0.45 |
| 06_GoogleNetInceptionModule | `iree_cuda` |  | 0.79 |  |  |  | 1.50 | 1.50 |
| 07_GoogleNetInceptionV1 | `mim_nvptx` | 0.38 | 1.65 | 0.12 | 0.30 | 0.11 | 2.18 | 2.56 |
| 07_GoogleNetInceptionV1 | `iree_cuda` |  | 1.73 |  |  |  | 5.17 | 5.17 |
| 08_ResNetBasicBlock | `mim_nvptx` | 0.10 | 0.14 | 0.02 | 0.03 | 0.10 | 0.29 | 0.39 |
| 08_ResNetBasicBlock | `iree_cuda` |  | 0.65 |  |  |  | 1.08 | 1.08 |
| 09_ResNet18 | `mim_nvptx` | 0.35 | 0.73 | 0.07 | 0.18 | 0.11 | 1.09 | 1.43 |
| 09_ResNet18 | `iree_cuda` |  | 1.74 |  |  |  | 3.24 | 3.24 |
| 11_VGG16 | `mim_nvptx` | 0.16 | 0.45 | 0.03 | 0.08 | 0.10 | 0.66 | 0.82 |
| 11_VGG16 | `iree_cuda` |  | 1.11 |  |  |  | 2.12 | 2.12 |
| 12_VGG19 | `mim_nvptx` | 0.17 | 0.51 | 0.04 | 0.10 | 0.10 | 0.74 | 0.92 |
| 12_VGG19 | `iree_cuda` |  | 1.19 |  |  |  | 2.20 | 2.20 |
| 13_DenseNet121TransitionLayer | `mim_nvptx` | 0.08 | 0.08 | 0.01 | 0.02 | 0.09 | 0.21 | 0.29 |
| 13_DenseNet121TransitionLayer | `iree_cuda` |  | 0.77 |  |  |  | 1.36 | 1.36 |
| 14_DenseNet121DenseBlock | `mim_nvptx` | 0.20 | 0.39 | 0.04 | 0.08 | 0.10 | 0.61 | 0.81 |
| 14_DenseNet121DenseBlock | `iree_cuda` |  | 0.95 |  |  |  | 1.72 | 1.72 |
| 15_DenseNet121 | `mim_nvptx` | 1.61 | 4.24 | 0.61 | 1.31 | 0.15 | 6.31 | 7.92 |
| 15_DenseNet121 | `iree_cuda` |  | 5.89 |  |  |  | 16.12 | 16.12 |
| 16_DenseNet201 | `mim_nvptx` | 2.59 | 13.52 | 5.76 | 3.91 | 0.21 | 23.40 | 25.99 |
| 16_DenseNet201 | `iree_cuda` |  | 9.73 |  |  |  | 36.18 | 36.18 |
| 18_SqueezeNet | `mim_nvptx` | 0.22 | 0.74 | 0.05 | 0.13 | 0.11 | 1.03 | 1.25 |
| 18_SqueezeNet | `iree_cuda` |  | 1.17 |  |  |  | 2.52 | 2.52 |
| 19_MobileNetV1 | `mim_nvptx` | 0.45 | 0.83 | 0.08 | 0.23 | 0.12 | 1.26 | 1.71 |
| 19_MobileNetV1 | `iree_cuda` |  | 2.12 |  |  |  | 3.78 | 3.78 |
| 20_MobileNetV2 | `mim_nvptx` | 0.80 | 1.51 | 0.14 | 0.52 | 0.14 | 2.31 | 3.10 |
| 20_MobileNetV2 | `iree_cuda` |  | 3.77 |  |  |  | 6.49 | 6.49 |
| 21_EfficientNetMBConv | `mim_nvptx` | 0.12 | 0.14 | 0.02 | 0.03 | 0.10 | 0.29 | 0.41 |
| 21_EfficientNetMBConv | `iree_cuda` |  | 0.73 |  |  |  | 1.36 | 1.36 |
| 22_EfficientNetB0 | `mim_nvptx` | 0.74 | 1.25 | 0.11 | 0.28 | 0.13 | 1.77 | 2.50 |
| 22_EfficientNetB0 | `iree_cuda` |  | 3.38 |  |  |  | 5.75 | 5.75 |
| 23_EfficientNetB1 | `mim_nvptx` | 0.49 | 0.76 | 0.06 | 0.17 | 0.11 | 1.11 | 1.60 |
| 23_EfficientNetB1 | `iree_cuda` |  | 2.06 |  |  |  | 3.75 | 3.75 |
| 24_EfficientNetB2 | `mim_nvptx` | 0.41 | 1.53 | 0.06 | 0.15 | 0.11 | 1.85 | 2.27 |
| 24_EfficientNetB2 | `iree_cuda` |  | 1.82 |  |  |  | 3.20 | 3.20 |
| 25_ShuffleNetUnit | `mim_nvptx` | 0.12 | 0.18 | 0.02 | 0.04 | 0.10 | 0.33 | 0.45 |
| 25_ShuffleNetUnit | `iree_cuda` |  | 0.84 |  |  |  | 1.43 | 1.43 |
| 26_ShuffleNet | `mim_nvptx` | 0.65 | 1.39 | 0.13 | 0.34 | 0.13 | 1.99 | 2.63 |
| 26_ShuffleNet | `iree_cuda` |  | 3.35 |  |  |  | 5.37 | 5.37 |
| 27_RegNet | `mim_nvptx` | 0.17 | 0.33 | 0.03 | 0.07 | 0.10 | 0.52 | 0.70 |
| 27_RegNet | `iree_cuda` |  | 0.97 |  |  |  | 1.70 | 1.70 |
| 28_VisionTransformer | `mim_nvptx` | 0.37 | 2.70 | 0.17 | 0.42 | 0.11 | 3.40 | 3.77 |
| 28_VisionTransformer | `iree_cuda` |  | 1.75 |  |  |  | 2.96 | 2.96 |
| 29_SwinMLP | `mim_nvptx` | 0.95 | 3.88 | 0.23 | 0.56 | 0.12 | 4.78 | 5.72 |
| 29_SwinMLP | `iree_cuda` |  | 3.44 |  |  |  | 6.03 | 6.03 |
| 43_MinGPTCausalAttention | `mim_nvptx` | 0.09 | 0.98 | 0.03 | 0.05 | 0.10 | 1.16 | 1.25 |
| 43_MinGPTCausalAttention | `iree_cuda` |  | 0.66 |  |  |  | 1.60 | 1.60 |
| 44_MiniGPTBlock | `mim_nvptx` | 0.13 | 1.26 | 0.05 | 0.09 | 0.10 | 1.50 | 1.63 |
| 44_MiniGPTBlock | `iree_cuda` |  | 0.82 |  |  |  | 1.74 | 1.74 |
| 45_UNetSoftmax | `mim_nvptx` | 0.37 | 3.51 | 0.14 | 0.34 | 0.11 | 4.10 | 4.47 |
| 45_UNetSoftmax | `iree_cuda` |  | 1.85 |  |  |  | 3.60 | 3.60 |
| 46_NetVladWithGhostClusters | `mim_nvptx` | 0.09 | 0.93 | 0.02 | 0.04 | 0.10 | 1.09 | 1.18 |
| 46_NetVladWithGhostClusters | `iree_cuda` |  | 0.69 |  |  |  | 1.49 | 1.49 |
| 47_NetVladNoGhostClusters | `mim_nvptx` | 0.09 | 0.93 | 0.02 | 0.04 | 0.10 | 1.10 | 1.18 |
| 47_NetVladNoGhostClusters | `iree_cuda` |  | 0.68 |  |  |  | 1.28 | 1.28 |
| 50_ReLUSelfAttention | `mim_nvptx` | 0.08 | 0.16 | 0.04 | 0.06 | 0.09 | 0.35 | 0.43 |
| 50_ReLUSelfAttention | `iree_cuda` |  | 0.59 |  |  |  | 1.46 | 1.46 |
