# GPU results

Sweep of the benchmark models on the GPU (`./run.sh --target gpu`), report runs
`old25` and `new11`. The raw data is in [`results/gpu/`](results/gpu/): `timing.csv` (per-model
median/min/max per variant), `compile_times.csv` (every compile step per model and variant) and
`summary.md` (the geometric means, from `scripts/geomean.py`).
How the numbers are measured is described in [`artifact_evaluation.md`](artifact_evaluation.md).

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti (16 GB, compute capability 8.9, `sm_89`), driver 615.78.08 |
| Host | Intel Core i5-10600KF, 12 CPUs, 31 GiB; runs pinned to core 2; governor `schedutil` (frequency not pinned); memory limited to 24 GB, no swap (`systemd-run -p MemoryMax=24G -p MemorySwapMax=0`) |
| Software | mimir-frontend branch `art-eval` (`1cf3acb`), MimIR `93a90de`; IREE 3.14.0, iree-turbine 3.9.0; PyTorch 2.15.0 (CPU build) |
| Measurement | 5 reps per variant, interleaved, one fresh process per call (CUDA setup, module load and weight upload are part of every timed call, for both variants), median reported |
| Correctness | every variant against the PyTorch output, rtol 1e-3, atol 1e-2 |

## Runtime

33 models were measured. Of the 35 models in `models.txt`, `31_VisionAttention` is left out
(its 8.6 GB attention matrix does not fit this GPU's 16 GB, for `mim_nvptx` and for `iree_cuda`)
and `10_ResNet101` is left out because its `iree_cuda` output fails the correctness check
(max diff / scale 2.8; `mim_nvptx` passes); with it, the geometric mean is 1.48 (34 models).
Both variants pass the correctness check on all 33 models. Median runtime per call;
the ratio column divides by the IREE baseline `iree_cuda` (below 1 = faster than IREE).

| model | `mim_nvptx` (ms) | `iree_cuda` (ms) | `mim_nvptx` / `iree_cuda` |
|---|---:|---:|---:|
| 01_MLP | 706.9 | 816.6 | 0.87 |
| 02_ShallowWideMLP | 1959.5 | 2360.7 | 0.83 |
| 03_DeepNarrowMLP | 216.1 | 281.3 | 0.77 |
| 04_LeNet5 | 139.3 | 138.0 | 1.01 |
| 05_AlexNet | 7839.8 | 12254.3 | 0.64 |
| 06_GoogleNetInceptionModule | 2387.8 | 2272.9 | 1.05 |
| 07_GoogleNetInceptionV1 | 386.0 | 196.2 | 1.97 |
| 08_ResNetBasicBlock | 244.1 | 191.0 | 1.28 |
| 09_ResNet18 | 188.2 | 164.6 | 1.14 |
| 11_VGG16 | 1276.9 | 511.4 | 2.50 |
| 12_VGG19 | 1574.7 | 591.6 | 2.66 |
| 13_DenseNet121TransitionLayer | 465.4 | 758.2 | 0.61 |
| 14_DenseNet121DenseBlock | 1430.5 | 458.4 | 3.12 |
| 15_DenseNet121 | 814.8 | 272.9 | 2.99 |
| 16_DenseNet201 | 1507.1 | 474.0 | 3.18 |
| 18_SqueezeNet | 3670.5 | 2278.9 | 1.61 |
| 19_MobileNetV1 | 253.7 | 175.6 | 1.44 |
| 20_MobileNetV2 | 253.1 | 162.1 | 1.56 |
| 21_EfficientNetMBConv | 1311.0 | 1058.4 | 1.24 |
| 22_EfficientNetB0 | 259.2 | 161.0 | 1.61 |
| 23_EfficientNetB1 | 214.9 | 160.4 | 1.34 |
| 24_EfficientNetB2 | 136.3 | 144.0 | 0.95 |
| 25_ShuffleNetUnit | 1321.0 | 665.6 | 1.98 |
| 26_ShuffleNet | 1430.6 | 843.7 | 1.70 |
| 27_RegNet | 506.5 | 220.4 | 2.30 |
| 28_VisionTransformer | 238.5 | 185.1 | 1.29 |
| 29_SwinMLP | 617.7 | 254.8 | 2.42 |
| 43_MinGPTCausalAttention | 1965.4 | 796.1 | 2.47 |
| 44_MiniGPTBlock | 3718.1 | 1660.2 | 2.24 |
| 45_UNetSoftmax | 1410.6 | 494.9 | 2.85 |
| 46_NetVladWithGhostClusters | 512.7 | 344.8 | 1.49 |
| 47_NetVladNoGhostClusters | 499.6 | 340.5 | 1.47 |
| 50_ReLUSelfAttention | 674.1 | 299.9 | 2.25 |
| **geometric mean** | | | **1.56** |

The geometric mean is that of `scripts/geomean.py`: per model the `mim_nvptx` median over
`iree_cuda`'s, averaged over the 33 models that have both variants.

`mim_nvptx` is slower than IREE in the geometric mean (1.56x) and faster on 6 of 33 models;
the ratios range from 0.61 (`13_DenseNet121TransitionLayer`) to 3.18 (`16_DenseNet201`).

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
| `trace_s` | 33 | 0.23 |  |
| `frontend_s` | 33 | 0.67 | 1.46 |
| `to_llvm_ir_s` | 33 | 0.67 |  |
| `opt_s` | 33 | 0.11 |  |
| `llc_s` | 33 | 0.13 |  |
| `link_s` | 33 | 0.11 |  |
| `total_s` | 33 | 1.14 | 2.77 |
| `end_to_end_s` | 33 | 1.40 | 2.77 |

Sums over all 33 models:

| variant | `trace_s` | `frontend_s` | `to_llvm_ir_s` | `opt_s` | `llc_s` | `link_s` | `total_s` | `end_to_end_s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mim_nvptx` | 12.7 | 73.1 | 73.1 | 22.1 | 12.0 | 3.9 | 111.1 | 123.9 |
| `iree_cuda` |  | 65.4 |  |  |  |  | 140.8 | 140.8 |

Per model (the full table with every step is `results/gpu/compile_times.csv`):

| model | variant | `trace_s` | `frontend_s` | `opt_s` | `llc_s` | `link_s` | `total_s` | `end_to_end_s` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01_MLP | `mim_nvptx` | 0.10 | 0.18 | 0.02 | 0.02 | 0.22 | 0.44 | 0.54 |
| 01_MLP | `iree_cuda` |  | 2.21 |  |  |  | 2.81 | 2.81 |
| 02_ShallowWideMLP | `mim_nvptx` | 0.06 | 0.10 | 0.01 | 0.02 | 0.10 | 0.23 | 0.30 |
| 02_ShallowWideMLP | `iree_cuda` |  | 4.50 |  |  |  | 5.55 | 5.55 |
| 03_DeepNarrowMLP | `mim_nvptx` | 0.12 | 0.27 | 0.03 | 0.05 | 0.10 | 0.45 | 0.57 |
| 03_DeepNarrowMLP | `iree_cuda` |  | 0.94 |  |  |  | 1.64 | 1.64 |
| 04_LeNet5 | `mim_nvptx` | 0.08 | 0.16 | 0.02 | 0.03 | 0.10 | 0.31 | 0.38 |
| 04_LeNet5 | `iree_cuda` |  | 0.57 |  |  |  | 1.05 | 1.05 |
| 05_AlexNet | `mim_nvptx` | 0.12 | 0.27 | 0.06 | 0.06 | 0.10 | 0.49 | 0.60 |
| 05_AlexNet | `iree_cuda` |  | 0.93 |  |  |  | 1.67 | 1.67 |
| 06_GoogleNetInceptionModule | `mim_nvptx` | 0.08 | 0.21 | 0.05 | 0.05 | 0.10 | 0.42 | 0.50 |
| 06_GoogleNetInceptionModule | `iree_cuda` |  | 0.79 |  |  |  | 1.51 | 1.51 |
| 07_GoogleNetInceptionV1 | `mim_nvptx` | 0.38 | 1.73 | 0.78 | 0.48 | 0.11 | 3.10 | 3.48 |
| 07_GoogleNetInceptionV1 | `iree_cuda` |  | 1.75 |  |  |  | 5.19 | 5.19 |
| 08_ResNetBasicBlock | `mim_nvptx` | 0.10 | 0.14 | 0.03 | 0.04 | 0.10 | 0.31 | 0.42 |
| 08_ResNetBasicBlock | `iree_cuda` |  | 0.66 |  |  |  | 1.10 | 1.10 |
| 09_ResNet18 | `mim_nvptx` | 0.35 | 0.71 | 0.29 | 0.26 | 0.11 | 1.37 | 1.73 |
| 09_ResNet18 | `iree_cuda` |  | 1.74 |  |  |  | 3.14 | 3.14 |
| 11_VGG16 | `mim_nvptx` | 0.16 | 0.43 | 0.11 | 0.11 | 0.10 | 0.76 | 0.91 |
| 11_VGG16 | `iree_cuda` |  | 1.13 |  |  |  | 2.11 | 2.11 |
| 12_VGG19 | `mim_nvptx` | 0.18 | 0.49 | 0.14 | 0.13 | 0.10 | 0.86 | 1.04 |
| 12_VGG19 | `iree_cuda` |  | 1.19 |  |  |  | 2.12 | 2.12 |
| 13_DenseNet121TransitionLayer | `mim_nvptx` | 0.09 | 0.08 | 0.02 | 0.02 | 0.10 | 0.22 | 0.31 |
| 13_DenseNet121TransitionLayer | `iree_cuda` |  | 0.77 |  |  |  | 1.28 | 1.28 |
| 14_DenseNet121DenseBlock | `mim_nvptx` | 0.21 | 0.40 | 0.17 | 0.16 | 0.10 | 0.83 | 1.03 |
| 14_DenseNet121DenseBlock | `iree_cuda` |  | 0.96 |  |  |  | 1.73 | 1.73 |
| 15_DenseNet121 | `mim_nvptx` | 1.64 | 7.65 | 4.66 | 1.83 | 0.15 | 14.29 | 15.93 |
| 15_DenseNet121 | `iree_cuda` |  | 5.94 |  |  |  | 15.85 | 15.85 |
| 16_DenseNet201 | `mim_nvptx` | 2.60 | 36.61 | 12.45 | 4.54 | 0.31 | 53.92 | 56.51 |
| 16_DenseNet201 | `iree_cuda` |  | 10.29 |  |  |  | 37.57 | 37.57 |
| 18_SqueezeNet | `mim_nvptx` | 0.22 | 0.74 | 0.19 | 0.19 | 0.10 | 1.22 | 1.44 |
| 18_SqueezeNet | `iree_cuda` |  | 1.21 |  |  |  | 2.61 | 2.61 |
| 19_MobileNetV1 | `mim_nvptx` | 0.48 | 0.83 | 0.20 | 0.24 | 0.12 | 1.39 | 1.87 |
| 19_MobileNetV1 | `iree_cuda` |  | 2.18 |  |  |  | 3.76 | 3.76 |
| 20_MobileNetV2 | `mim_nvptx` | 0.80 | 1.50 | 0.34 | 0.49 | 0.15 | 2.47 | 3.27 |
| 20_MobileNetV2 | `iree_cuda` |  | 3.76 |  |  |  | 6.23 | 6.23 |
| 21_EfficientNetMBConv | `mim_nvptx` | 0.12 | 0.14 | 0.03 | 0.04 | 0.10 | 0.30 | 0.42 |
| 21_EfficientNetMBConv | `iree_cuda` |  | 0.83 |  |  |  | 1.42 | 1.42 |
| 22_EfficientNetB0 | `mim_nvptx` | 0.74 | 1.27 | 0.25 | 0.37 | 0.13 | 2.03 | 2.77 |
| 22_EfficientNetB0 | `iree_cuda` |  | 3.34 |  |  |  | 5.81 | 5.81 |
| 23_EfficientNetB1 | `mim_nvptx` | 0.50 | 0.78 | 0.13 | 0.20 | 0.12 | 1.22 | 1.71 |
| 23_EfficientNetB1 | `iree_cuda` |  | 2.21 |  |  |  | 3.93 | 3.93 |
| 24_EfficientNetB2 | `mim_nvptx` | 0.43 | 1.77 | 0.11 | 0.18 | 0.11 | 2.16 | 2.60 |
| 24_EfficientNetB2 | `iree_cuda` |  | 1.81 |  |  |  | 3.70 | 3.70 |
| 25_ShuffleNetUnit | `mim_nvptx` | 0.12 | 0.18 | 0.03 | 0.04 | 0.10 | 0.35 | 0.47 |
| 25_ShuffleNetUnit | `iree_cuda` |  | 0.84 |  |  |  | 1.44 | 1.44 |
| 26_ShuffleNet | `mim_nvptx` | 0.66 | 1.40 | 0.33 | 0.43 | 0.14 | 2.30 | 2.96 |
| 26_ShuffleNet | `iree_cuda` |  | 3.34 |  |  |  | 5.43 | 5.43 |
| 27_RegNet | `mim_nvptx` | 0.18 | 0.33 | 0.07 | 0.08 | 0.10 | 0.58 | 0.76 |
| 27_RegNet | `iree_cuda` |  | 0.95 |  |  |  | 1.64 | 1.64 |
| 28_VisionTransformer | `mim_nvptx` | 0.38 | 2.76 | 0.18 | 0.41 | 0.11 | 3.45 | 3.83 |
| 28_VisionTransformer | `iree_cuda` |  | 1.73 |  |  |  | 3.15 | 3.15 |
| 29_SwinMLP | `mim_nvptx` | 0.98 | 4.07 | 0.39 | 0.66 | 0.12 | 5.25 | 6.23 |
| 29_SwinMLP | `iree_cuda` |  | 3.54 |  |  |  | 6.13 | 6.13 |
| 43_MinGPTCausalAttention | `mim_nvptx` | 0.10 | 1.03 | 0.03 | 0.05 | 0.09 | 1.20 | 1.31 |
| 43_MinGPTCausalAttention | `iree_cuda` |  | 0.66 |  |  |  | 1.57 | 1.57 |
| 44_MiniGPTBlock | `mim_nvptx` | 0.13 | 1.28 | 0.05 | 0.09 | 0.10 | 1.52 | 1.65 |
| 44_MiniGPTBlock | `iree_cuda` |  | 0.81 |  |  |  | 1.90 | 1.90 |
| 45_UNetSoftmax | `mim_nvptx` | 0.38 | 3.54 | 0.89 | 0.59 | 0.11 | 5.14 | 5.51 |
| 45_UNetSoftmax | `iree_cuda` |  | 1.84 |  |  |  | 3.59 | 3.59 |
| 46_NetVladWithGhostClusters | `mim_nvptx` | 0.09 | 0.96 | 0.03 | 0.04 | 0.10 | 1.13 | 1.21 |
| 46_NetVladWithGhostClusters | `iree_cuda` |  | 0.69 |  |  |  | 1.46 | 1.46 |
| 47_NetVladNoGhostClusters | `mim_nvptx` | 0.09 | 0.94 | 0.03 | 0.04 | 0.10 | 1.11 | 1.20 |
| 47_NetVladNoGhostClusters | `iree_cuda` |  | 0.70 |  |  |  | 1.30 | 1.30 |
| 50_ReLUSelfAttention | `mim_nvptx` | 0.08 | 0.16 | 0.04 | 0.06 | 0.10 | 0.35 | 0.43 |
| 50_ReLUSelfAttention | `iree_cuda` |  | 0.60 |  |  |  | 1.43 | 1.43 |
