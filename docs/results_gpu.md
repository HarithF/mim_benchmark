# GPU results

Full sweep of the benchmark models on the GPU (`./run.sh all --target gpu`), report run
`20261115_225105`. The raw data is in [`results/gpu/`](results/gpu/): `timing.csv` (per-model
median/min/max per variant), `compile_times.csv` (every compile step per model and variant) and
`summary.md` (the geometric means, from `scripts/geomean.py`).
How the numbers are measured is described in [`artifact_evaluation.md`](artifact_evaluation.md).

## Setup

| | |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB (40 GB, compute capability 8.0, `sm_80`), driver 580.178.04 |
| Host | AMD EPYC 7662, 8 CPUs, 64 GB requested; runs pinned to core 2; governor `schedutil` (frequency not pinned) |
| Measurement | 5 reps per variant, interleaved, one fresh process per call (CUDA setup, module load and weight upload are part of every timed call, for both variants), median reported |
| Correctness | every variant against the PyTorch output, rtol 1e-3, atol 1e-2 |

## Runtime

35 models were measured (`30_SwinTransformerV2` was not built). `mim_nvptx` passes the
correctness check on 34 of them and has no PyTorch reference on the 35th (‡); `iree_cuda`
passes on 33, fails the check on one (†) and does not run on one (‡). Median runtime per call;
the ratio column divides by the IREE baseline `iree_cuda` (below 1 = faster than IREE).

| model | `mim_nvptx` (ms) | `iree_cuda` (ms) | `mim_nvptx` / `iree_cuda` |
|---|---:|---:|---:|
| 01_MLP | 577.7 | 912.1 | 0.63 |
| 02_ShallowWideMLP | 1656.7 | 1999.0 | 0.83 |
| 03_DeepNarrowMLP | 263.3 | 375.6 | 0.70 |
| 04_LeNet5 | 229.9 | 254.5 | 0.90 |
| 05_AlexNet | 6675.3 | 8062.0 | 0.83 |
| 06_GoogleNetInceptionModule | 3976.4 | 3425.7 | 1.16 |
| 07_GoogleNetInceptionV1 | 757.0 | 361.4 | 2.09 |
| 08_ResNetBasicBlock | 571.8 | 316.2 | 1.81 |
| 09_ResNet18 | 347.7 | 296.0 | 1.17 |
| 10_ResNet101 † | 1331.9 | 1647.7 | 0.81 |
| 11_VGG16 | 1407.4 | 473.1 | 2.97 |
| 12_VGG19 | 1693.7 | 516.2 | 3.28 |
| 13_DenseNet121TransitionLayer | 733.2 | 1311.7 | 0.56 |
| 14_DenseNet121DenseBlock | 4181.0 | 470.5 | 8.89 |
| 15_DenseNet121 | 1759.6 | 410.7 | 4.28 |
| 16_DenseNet201 | 3211.8 | 590.4 | 5.44 |
| 18_SqueezeNet | 8686.2 | 1453.7 | 5.98 |
| 19_MobileNetV1 | 517.3 | 297.2 | 1.74 |
| 20_MobileNetV2 | 554.4 | 314.4 | 1.76 |
| 21_EfficientNetMBConv | 3073.6 | 725.0 | 4.24 |
| 22_EfficientNetB0 | 558.7 | 319.3 | 1.75 |
| 23_EfficientNetB1 | 460.5 | 294.9 | 1.56 |
| 24_EfficientNetB2 | 271.9 | 276.0 | 0.99 |
| 25_ShuffleNetUnit | 1814.4 | 952.6 | 1.90 |
| 26_ShuffleNet | 1822.3 | 479.1 | 3.80 |
| 27_RegNet | 709.8 | 258.2 | 2.75 |
| 28_VisionTransformer | 364.3 | 286.3 | 1.27 |
| 29_SwinMLP | 787.9 | 324.7 | 2.43 |
| 31_VisionAttention | 10789.0 | 975.5 | 11.06 |
| 43_MinGPTCausalAttention | 3285.2 | 767.1 | 4.28 |
| 44_MiniGPTBlock ‡ | 4339.7 | failed | |
| 45_UNetSoftmax | 3146.0 | 396.0 | 7.94 |
| 46_NetVladWithGhostClusters | 800.7 | 726.8 | 1.10 |
| 47_NetVladNoGhostClusters | 898.0 | 747.3 | 1.20 |
| 50_ReLUSelfAttention | 1167.3 | 352.0 | 3.32 |
| **geometric mean** | | | **2.03** |

† `10_ResNet101`: the `iree_cuda` output fails the correctness check (max diff / scale 2.8;
`mim_nvptx` passes). Its timing is still in the table and in the geometric mean; without this
model the geometric mean is 2.09 (33 models).

‡ `44_MiniGPTBlock`: `iree_cuda` exits with an error on every rep (`parameters without a
weights.bin slot: mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight, mlp.c_proj.bias`), and
the PyTorch reference is skipped for the same weight mismatch, so the `mim_nvptx` output is
not checked for this model. It is left out of the geometric mean.

The geometric mean is that of `scripts/geomean.py`: per model the `mim_nvptx` median over
`iree_cuda`'s, averaged over the 34 models that have both variants.

`mim_nvptx` is slower than IREE in the geometric mean (2.03x) and faster on 8 of 34 models;
the ratios range from 0.56 (`13_DenseNet121TransitionLayer`) to 11.06 (`31_VisionAttention`).

## Compile time

Seconds per model and variant; the steps are defined in
[`artifact_evaluation.md`](artifact_evaluation.md#e2-compile-time). For `mim_nvptx`,
`to_llvm_ir_s` is PyTorch graph to LLVM IR (equal to `frontend_s`), `total_s` adds `opt`, `llc`
and the link, `end_to_end_s` adds tracing the model. `iree_cuda` reports only `frontend_s` and
`total_s`; it has no separate LLVM steps and no separate trace time, so its `end_to_end_s`
equals its `total_s`.

Geometric mean per step over all 35 models, in seconds (`scripts/geomean.py` for the columns
both variants have; the `mim_nvptx`-only rows are computed the same way from
`compile_times.csv`):

| column | models | `mim_nvptx` | `iree_cuda` |
|---|---:|---:|---:|
| `trace_s` | 35 | 0.36 |  |
| `frontend_s` | 35 | 1.38 | 2.43 |
| `to_llvm_ir_s` | 35 | 1.38 |  |
| `opt_s` | 35 | 0.30 |  |
| `llc_s` | 35 | 0.30 |  |
| `link_s` | 35 | 0.27 |  |
| `total_s` | 35 | 2.53 | 4.88 |
| `end_to_end_s` | 35 | 2.95 | 4.88 |

Sums over all 35 models:

| variant | `trace_s` | `frontend_s` | `to_llvm_ir_s` | `opt_s` | `llc_s` | `link_s` | `total_s` | `end_to_end_s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mim_nvptx` | 19.6 | 127.5 | 127.5 | 67.7 | 27.9 | 9.5 | 232.6 | 252.2 |
| `iree_cuda` |  | 111.1 |  |  |  |  | 247.8 | 247.8 |

Per model (the full table with every step is `results/gpu/compile_times.csv`):

| model | variant | `trace_s` | `frontend_s` | `opt_s` | `llc_s` | `link_s` | `total_s` | `end_to_end_s` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01_MLP | `mim_nvptx` | 0.15 | 0.23 | 0.06 | 0.06 | 0.22 | 0.57 | 0.72 |
| 01_MLP | `iree_cuda` |  | 4.03 |  |  |  | 5.12 | 5.12 |
| 02_ShallowWideMLP | `mim_nvptx` | 0.13 | 0.21 | 0.06 | 0.06 | 0.25 | 0.58 | 0.71 |
| 02_ShallowWideMLP | `iree_cuda` |  | 8.44 |  |  |  | 9.73 | 9.73 |
| 03_DeepNarrowMLP | `mim_nvptx` | 0.20 | 0.54 | 0.08 | 0.12 | 0.25 | 1.00 | 1.20 |
| 03_DeepNarrowMLP | `iree_cuda` |  | 1.39 |  |  |  | 2.87 | 2.87 |
| 04_LeNet5 | `mim_nvptx` | 0.14 | 0.35 | 0.07 | 0.07 | 0.24 | 0.73 | 0.88 |
| 04_LeNet5 | `iree_cuda` |  | 0.94 |  |  |  | 1.90 | 1.90 |
| 05_AlexNet | `mim_nvptx` | 0.20 | 0.51 | 0.12 | 0.12 | 0.25 | 1.00 | 1.20 |
| 05_AlexNet | `iree_cuda` |  | 1.89 |  |  |  | 3.24 | 3.24 |
| 06_GoogleNetInceptionModule | `mim_nvptx` | 0.15 | 0.42 | 0.14 | 0.14 | 0.23 | 0.93 | 1.08 |
| 06_GoogleNetInceptionModule | `iree_cuda` |  | 1.62 |  |  |  | 2.92 | 2.92 |
| 07_GoogleNetInceptionV1 | `mim_nvptx` | 0.52 | 3.00 | 1.93 | 0.82 | 0.28 | 6.03 | 6.55 |
| 07_GoogleNetInceptionV1 | `iree_cuda` |  | 2.58 |  |  |  | 8.53 | 8.53 |
| 08_ResNetBasicBlock | `mim_nvptx` | 0.20 | 0.31 | 0.09 | 0.09 | 0.23 | 0.72 | 0.92 |
| 08_ResNetBasicBlock | `iree_cuda` |  | 1.19 |  |  |  | 2.08 | 2.08 |
| 09_ResNet18 | `mim_nvptx` | 0.47 | 1.32 | 0.69 | 0.52 | 0.28 | 2.81 | 3.29 |
| 09_ResNet18 | `iree_cuda` |  | 2.54 |  |  |  | 5.26 | 5.26 |
| 10_ResNet101 | `mim_nvptx` | 1.88 | 4.69 | 2.46 | 2.60 | 0.41 | 10.17 | 12.05 |
| 10_ResNet101 | `iree_cuda` |  | 9.61 |  |  |  | 16.43 | 16.43 |
| 11_VGG16 | `mim_nvptx` | 0.26 | 0.84 | 0.25 | 0.22 | 0.26 | 1.56 | 1.82 |
| 11_VGG16 | `iree_cuda` |  | 2.26 |  |  |  | 4.18 | 4.18 |
| 12_VGG19 | `mim_nvptx` | 0.30 | 0.94 | 0.31 | 0.24 | 0.25 | 1.74 | 2.04 |
| 12_VGG19 | `iree_cuda` |  | 2.44 |  |  |  | 4.44 | 4.44 |
| 13_DenseNet121TransitionLayer | `mim_nvptx` | 0.16 | 0.21 | 0.06 | 0.06 | 0.24 | 0.57 | 0.73 |
| 13_DenseNet121TransitionLayer | `iree_cuda` |  | 1.76 |  |  |  | 2.70 | 2.70 |
| 14_DenseNet121DenseBlock | `mim_nvptx` | 0.32 | 0.74 | 0.40 | 0.36 | 0.25 | 1.75 | 2.08 |
| 14_DenseNet121DenseBlock | `iree_cuda` |  | 1.54 |  |  |  | 3.22 | 3.22 |
| 15_DenseNet121 | `mim_nvptx` | 2.07 | 11.32 | 13.75 | 4.10 | 0.34 | 29.51 | 31.58 |
| 15_DenseNet121 | `iree_cuda` |  | 8.04 |  |  |  | 24.31 | 24.31 |
| 16_DenseNet201 | `mim_nvptx` | 3.38 | 51.24 | 39.64 | 9.65 | 0.42 | 100.94 | 104.32 |
| 16_DenseNet201 | `iree_cuda` |  | 13.20 |  |  |  | 56.14 | 56.14 |
| 18_SqueezeNet | `mim_nvptx` | 0.33 | 1.35 | 0.42 | 0.34 | 0.27 | 2.38 | 2.71 |
| 18_SqueezeNet | `iree_cuda` |  | 1.89 |  |  |  | 4.43 | 4.43 |
| 19_MobileNetV1 | `mim_nvptx` | 0.60 | 1.47 | 0.46 | 0.47 | 0.29 | 2.69 | 3.28 |
| 19_MobileNetV1 | `iree_cuda` |  | 2.99 |  |  |  | 6.00 | 6.00 |
| 20_MobileNetV2 | `mim_nvptx` | 1.11 | 2.57 | 0.80 | 0.93 | 0.32 | 4.63 | 5.74 |
| 20_MobileNetV2 | `iree_cuda` |  | 5.05 |  |  |  | 9.72 | 9.72 |
| 21_EfficientNetMBConv | `mim_nvptx` | 0.20 | 0.30 | 0.08 | 0.09 | 0.24 | 0.71 | 0.91 |
| 21_EfficientNetMBConv | `iree_cuda` |  | 1.33 |  |  |  | 2.54 | 2.54 |
| 22_EfficientNetB0 | `mim_nvptx` | 0.96 | 2.20 | 0.59 | 0.70 | 0.29 | 3.78 | 4.74 |
| 22_EfficientNetB0 | `iree_cuda` |  | 4.86 |  |  |  | 9.35 | 9.35 |
| 23_EfficientNetB1 | `mim_nvptx` | 0.66 | 1.43 | 0.31 | 0.37 | 0.28 | 2.40 | 3.06 |
| 23_EfficientNetB1 | `iree_cuda` |  | 3.03 |  |  |  | 6.16 | 6.16 |
| 24_EfficientNetB2 | `mim_nvptx` | 0.60 | 2.94 | 0.26 | 0.34 | 0.27 | 3.80 | 4.41 |
| 24_EfficientNetB2 | `iree_cuda` |  | 2.67 |  |  |  | 5.26 | 5.26 |
| 25_ShuffleNetUnit | `mim_nvptx` | 0.20 | 0.36 | 0.09 | 0.10 | 0.25 | 0.79 | 0.99 |
| 25_ShuffleNetUnit | `iree_cuda` |  | 1.68 |  |  |  | 2.83 | 2.83 |
| 26_ShuffleNet | `mim_nvptx` | 0.85 | 2.45 | 0.70 | 0.82 | 0.30 | 4.27 | 5.12 |
| 26_ShuffleNet | `iree_cuda` |  | 4.67 |  |  |  | 8.36 | 8.36 |
| 27_RegNet | `mim_nvptx` | 0.28 | 0.62 | 0.17 | 0.26 | 0.25 | 1.30 | 1.57 |
| 27_RegNet | `iree_cuda` |  | 1.58 |  |  |  | 3.03 | 3.03 |
| 28_VisionTransformer | `mim_nvptx` | 0.51 | 4.80 | 0.37 | 0.84 | 0.26 | 6.28 | 6.79 |
| 28_VisionTransformer | `iree_cuda` |  | 2.49 |  |  |  | 5.04 | 5.04 |
| 29_SwinMLP | `mim_nvptx` | 1.27 | 6.98 | 1.11 | 1.49 | 0.29 | 9.87 | 11.14 |
| 29_SwinMLP | `iree_cuda` |  | 5.39 |  |  |  | 10.20 | 10.20 |
| 31_VisionAttention | `mim_nvptx` | 0.19 | 8.43 | 0.09 | 0.12 | 0.24 | 8.87 | 9.06 |
| 31_VisionAttention | `iree_cuda` |  | 1.12 |  |  |  | 2.59 | 2.59 |
| 43_MinGPTCausalAttention | `mim_nvptx` | 0.16 | 2.03 | 0.11 | 0.12 | 0.25 | 2.50 | 2.66 |
| 43_MinGPTCausalAttention | `iree_cuda` |  | 1.33 |  |  |  | 2.83 | 2.83 |
| 44_MiniGPTBlock | `mim_nvptx` | 0.23 | 2.46 | 0.15 | 0.20 | 0.25 | 3.05 | 3.28 |
| 44_MiniGPTBlock | `iree_cuda` |  | 1.41 |  |  |  | 3.29 | 3.29 |
| 45_UNetSoftmax | `mim_nvptx` | 0.51 | 6.00 | 1.64 | 1.16 | 0.28 | 9.09 | 9.60 |
| 45_UNetSoftmax | `iree_cuda` |  | 2.68 |  |  |  | 5.82 | 5.82 |
| 46_NetVladWithGhostClusters | `mim_nvptx` | 0.16 | 1.96 | 0.08 | 0.10 | 0.24 | 2.39 | 2.54 |
| 46_NetVladWithGhostClusters | `iree_cuda` |  | 1.16 |  |  |  | 2.57 | 2.57 |
| 47_NetVladNoGhostClusters | `mim_nvptx` | 0.17 | 1.89 | 0.07 | 0.10 | 0.25 | 2.30 | 2.47 |
| 47_NetVladNoGhostClusters | `iree_cuda` |  | 1.36 |  |  |  | 2.56 | 2.56 |
| 50_ReLUSelfAttention | `mim_nvptx` | 0.14 | 0.36 | 0.11 | 0.15 | 0.25 | 0.86 | 1.00 |
| 50_ReLUSelfAttention | `iree_cuda` |  | 0.94 |  |  |  | 2.21 | 2.21 |
