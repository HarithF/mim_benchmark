# CPU results

Full sweep of the 35 benchmark models on the CPU (`./run.sh`, `TARGET=cpu`), report run
`20261114_120909`. The raw data is in [`results/cpu/`](results/cpu/): `timing.csv` (per-model
median/min/max per variant), `compile_times.csv` (every compile step per model and variant) and
`summary.md` (the geometric means, from `scripts/geomean.py`).
How the numbers are measured is described in [`artifact_evaluation.md`](artifact_evaluation.md).

## Setup

| | |
|---|---|
| CPU | Intel Core Ultra 7 265H (16 cores), runs pinned to core 2; governor `performance`, turbo on |
| Memory / OS | 62 GB / Linux 6.18 (Manjaro) |
| Measurement | 5 reps per variant, interleaved, one forked process per call, median reported |
| Correctness | every variant against the PyTorch output, rtol 1e-3, atol 1e-2 |

## Runtime

All 35 models pass on all three variants (35/35). Median runtime per call; the
ratio columns divide by the torch-mlir baseline `mlir_llvm` (below 1 = faster than torch-mlir).

| model | `mim_llvm` (ms) | `mim_mlir_llvm` (ms) | `mlir_llvm` (ms) | `mim_llvm` / `mlir_llvm` | `mim_mlir_llvm` / `mlir_llvm` |
|---|---:|---:|---:|---:|---:|
| 01_MLP | 7521.8 | 9812.5 | 17491.0 | 0.43 | 0.56 |
| 02_ShallowWideMLP | 24590.8 | 32485.8 | 66178.2 | 0.37 | 0.49 |
| 03_DeepNarrowMLP | 2227.2 | 2634.1 | 2560.7 | 0.87 | 1.03 |
| 04_LeNet5 | 174.3 | 136.9 | 138.0 | 1.26 | 0.99 |
| 05_AlexNet | 65556.1 | 77212.2 | 59630.2 | 1.10 | 1.29 |
| 06_GoogleNetInceptionModule | 9451.6 | 17504.9 | 26331.6 | 0.36 | 0.66 |
| 07_GoogleNetInceptionV1 | 983.3 | 1186.7 | 1076.9 | 0.91 | 1.10 |
| 08_ResNetBasicBlock | 782.7 | 1303.6 | 1316.2 | 0.59 | 0.99 |
| 09_ResNet18 | 265.8 | 275.3 | 227.1 | 1.17 | 1.21 |
| 10_ResNet101 | 4638.6 | 5456.7 | 4746.6 | 0.98 | 1.15 |
| 11_VGG16 | 5847.3 | 7092.0 | 7393.0 | 0.79 | 0.96 |
| 12_VGG19 | 7400.5 | 8616.4 | 8840.2 | 0.84 | 0.97 |
| 13_DenseNet121TransitionLayer | 3346.2 | 3472.1 | 3504.0 | 0.95 | 0.99 |
| 14_DenseNet121DenseBlock | 6250.9 | 9935.2 | 10030.1 | 0.62 | 0.99 |
| 15_DenseNet121 | 1707.7 | 2312.5 | 2039.0 | 0.84 | 1.13 |
| 16_DenseNet201 | 3511.6 | 4904.3 | 3974.5 | 0.88 | 1.23 |
| 18_SqueezeNet | 17126.3 | 28296.5 | 29104.4 | 0.59 | 0.97 |
| 19_MobileNetV1 | 415.5 | 560.4 | 568.4 | 0.73 | 0.99 |
| 20_MobileNetV2 | 341.4 | 467.7 | 456.8 | 0.75 | 1.02 |
| 21_EfficientNetMBConv | 2839.4 | 5661.3 | 9750.6 | 0.29 | 0.58 |
| 22_EfficientNetB0 | 356.0 | 474.6 | 456.8 | 0.78 | 1.04 |
| 23_EfficientNetB1 | 180.0 | 332.8 | 320.0 | 0.56 | 1.04 |
| 24_EfficientNetB2 | 22.5 | 33.6 | 38.9 | 0.58 | 0.86 |
| 25_ShuffleNetUnit | 3292.1 | 7976.3 | 13209.9 | 0.25 | 0.60 |
| 26_ShuffleNet | 2670.2 | 3971.6 | 6584.0 | 0.41 | 0.60 |
| 27_RegNet | 2111.3 | 3190.0 | 3136.8 | 0.67 | 1.02 |
| 28_VisionTransformer | 490.8 | 509.6 | 538.9 | 0.91 | 0.95 |
| 29_SwinMLP | 3520.2 | 2789.8 | 2483.8 | 1.42 | 1.12 |
| 31_VisionAttention | 19145.2 | 19900.5 | 22656.7 | 0.85 | 0.88 |
| 43_MinGPTCausalAttention | 14532.5 | 11169.5 | 9703.3 | 1.50 | 1.15 |
| 44_MiniGPTBlock | 33000.9 | 33537.6 | 27366.9 | 1.21 | 1.23 |
| 45_UNetSoftmax | 8311.9 | 10758.6 | 10861.2 | 0.77 | 0.99 |
| 46_NetVladWithGhostClusters | 628.0 | 517.5 | 663.0 | 0.95 | 0.78 |
| 47_NetVladNoGhostClusters | 568.0 | 456.4 | 582.4 | 0.98 | 0.78 |
| 50_ReLUSelfAttention | 2993.2 | 2877.1 | 2700.8 | 1.11 | 1.07 |
| **geometric mean** | | | | **0.74** | **0.93** |

The geometric means are those of `scripts/geomean.py`: per model the variant's median over
`mlir_llvm`'s, averaged over every model that has all three variants.

`mim_llvm` is faster than torch-mlir on 28 of 35 models, `mim_mlir_llvm` on 20.

## Compile time

Seconds per model and variant; the steps are defined in
[`artifact_evaluation.md`](artifact_evaluation.md#e2-compile-time). `to_llvm_ir_s` is PyTorch graph to
LLVM IR, `total_s` adds `opt -O3`, `llc` and the link, `end_to_end_s` adds tracing the model.

Geometric mean per step over all 35 models, in seconds (`scripts/geomean.py`):

| column | models | `mim_llvm` | `mim_mlir_llvm` | `mlir_llvm` |
|---|---:|---:|---:|---:|
| `trace_s` | 35 | 0.09 | 0.18 | 0.78 |
| `frontend_s` | 35 | 0.42 | 0.38 | 0.48 |
| `to_llvm_ir_s` | 35 | 0.42 | 0.66 | 0.69 |
| `opt_s` | 35 | 0.62 | 1.61 | 1.48 |
| `llc_s` | 35 | 0.32 | 0.67 | 0.60 |
| `link_s` | 35 | 0.11 | 0.14 | 0.15 |
| `total_s` | 35 | 1.69 | 3.81 | 3.59 |
| `end_to_end_s` | 35 | 1.79 | 4.01 | 4.88 |

Sums over all 35 models:

| variant | `trace_s` | `frontend_s` | `mlir_opt_s` | `libm_s` | `mlir_translate_s` | `to_llvm_ir_s` | `opt_s` | `llc_s` | `link_s` | `total_s` | `end_to_end_s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mim_llvm` | 7.1 | 34.5 |  |  |  | 34.5 | 105.9 | 24.3 | 4.1 | 168.8 | 175.9 |
| `mim_mlir_llvm` | 10.8 | 15.1 | 10.8 | 0.3 | 3.0 | 29.2 | 254.7 | 47.0 | 5.4 | 336.3 | 347.1 |
| `mlir_llvm` | 28.8 | 22.8 | 10.2 | 0.3 | 3.1 | 36.4 | 374.8 | 58.7 | 7.8 | 477.7 | 506.5 |

Per model (the full table with every step is `results/cpu/compile_times.csv`):

| model | variant | `trace_s` | `to_llvm_ir_s` | `opt_s` | `llc_s` | `link_s` | `total_s` | `end_to_end_s` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01_MLP | `mim_llvm` | 0.01 | 0.08 | 0.24 | 0.13 | 0.31 | 0.76 | 0.78 |
| 01_MLP | `mim_mlir_llvm` | 0.07 | 0.49 | 0.13 | 0.13 | 0.10 | 0.85 | 0.93 |
| 01_MLP | `mlir_llvm` | 1.10 | 0.58 | 0.05 | 0.04 | 0.10 | 0.77 | 1.87 |
| 02_ShallowWideMLP | `mim_llvm` | 0.02 | 0.07 | 0.04 | 0.04 | 0.10 | 0.25 | 0.27 |
| 02_ShallowWideMLP | `mim_mlir_llvm` | 0.06 | 0.46 | 0.11 | 0.12 | 0.10 | 0.79 | 0.85 |
| 02_ShallowWideMLP | `mlir_llvm` | 0.62 | 0.38 | 0.05 | 0.04 | 0.10 | 0.57 | 1.19 |
| 03_DeepNarrowMLP | `mim_llvm` | 0.06 | 0.28 | 0.16 | 0.12 | 0.10 | 0.66 | 0.72 |
| 03_DeepNarrowMLP | `mim_mlir_llvm` | 0.09 | 0.41 | 0.93 | 0.76 | 0.10 | 2.21 | 2.30 |
| 03_DeepNarrowMLP | `mlir_llvm` | 0.64 | 0.43 | 0.28 | 0.15 | 0.10 | 0.96 | 1.60 |
| 04_LeNet5 | `mim_llvm` | 0.02 | 0.14 | 0.10 | 0.07 | 0.09 | 0.40 | 0.42 |
| 04_LeNet5 | `mim_mlir_llvm` | 0.07 | 0.38 | 0.26 | 0.14 | 0.10 | 0.90 | 0.96 |
| 04_LeNet5 | `mlir_llvm` | 0.61 | 0.34 | 0.42 | 0.23 | 0.10 | 1.10 | 1.72 |
| 05_AlexNet | `mim_llvm` | 0.05 | 0.22 | 0.21 | 0.14 | 0.10 | 0.67 | 0.72 |
| 05_AlexNet | `mim_mlir_llvm` | 0.10 | 0.42 | 0.38 | 0.25 | 0.10 | 1.16 | 1.26 |
| 05_AlexNet | `mlir_llvm` | 0.64 | 0.36 | 0.45 | 0.18 | 0.10 | 1.09 | 1.72 |
| 06_GoogleNetInceptionModule | `mim_llvm` | 0.03 | 0.20 | 0.31 | 0.25 | 0.09 | 0.85 | 0.88 |
| 06_GoogleNetInceptionModule | `mim_mlir_llvm` | 0.07 | 0.38 | 0.38 | 0.37 | 0.10 | 1.23 | 1.29 |
| 06_GoogleNetInceptionModule | `mlir_llvm` | 0.61 | 0.31 | 0.30 | 0.30 | 0.10 | 1.02 | 1.63 |
| 07_GoogleNetInceptionV1 | `mim_llvm` | 0.23 | 1.90 | 5.13 | 1.02 | 0.11 | 8.16 | 8.39 |
| 07_GoogleNetInceptionV1 | `mim_mlir_llvm` | 0.30 | 0.78 | 3.99 | 0.91 | 0.17 | 5.83 | 6.14 |
| 07_GoogleNetInceptionV1 | `mlir_llvm` | 0.76 | 0.82 | 4.38 | 0.68 | 0.17 | 6.04 | 6.79 |
| 08_ResNetBasicBlock | `mim_llvm` | 0.03 | 0.07 | 0.08 | 0.06 | 0.10 | 0.32 | 0.35 |
| 08_ResNetBasicBlock | `mim_mlir_llvm` | 0.09 | 0.37 | 0.53 | 0.47 | 0.10 | 1.47 | 1.56 |
| 08_ResNetBasicBlock | `mlir_llvm` | 0.63 | 0.36 | 0.48 | 0.47 | 0.11 | 1.41 | 2.04 |
| 09_ResNet18 | `mim_llvm` | 0.18 | 0.54 | 1.08 | 0.59 | 0.11 | 2.33 | 2.52 |
| 09_ResNet18 | `mim_mlir_llvm` | 0.26 | 0.64 | 3.20 | 0.76 | 0.15 | 4.75 | 5.01 |
| 09_ResNet18 | `mlir_llvm` | 0.79 | 0.92 | 3.13 | 0.85 | 2.62 | 7.52 | 8.31 |
| 10_ResNet101 | `mim_llvm` | 0.96 | 2.54 | 10.99 | 1.38 | 0.17 | 15.09 | 16.04 |
| 10_ResNet101 | `mim_mlir_llvm` | 1.00 | 2.55 | 35.56 | 4.24 | 0.49 | 42.85 | 43.85 |
| 10_ResNet101 | `mlir_llvm` | 1.28 | 4.40 | 30.45 | 4.10 | 0.39 | 39.34 | 40.63 |
| 11_VGG16 | `mim_llvm` | 0.08 | 0.35 | 0.50 | 0.34 | 0.10 | 1.28 | 1.36 |
| 11_VGG16 | `mim_mlir_llvm` | 0.12 | 0.45 | 1.81 | 0.72 | 0.10 | 3.08 | 3.20 |
| 11_VGG16 | `mlir_llvm` | 0.64 | 0.44 | 1.58 | 0.56 | 0.11 | 2.68 | 3.32 |
| 12_VGG19 | `mim_llvm` | 0.09 | 0.40 | 0.72 | 0.43 | 0.10 | 1.65 | 1.74 |
| 12_VGG19 | `mim_mlir_llvm` | 0.13 | 0.45 | 2.20 | 0.79 | 0.11 | 3.55 | 3.68 |
| 12_VGG19 | `mlir_llvm` | 0.66 | 0.46 | 2.05 | 0.68 | 0.11 | 3.30 | 3.95 |
| 13_DenseNet121TransitionLayer | `mim_llvm` | 0.02 | 0.06 | 0.04 | 0.03 | 0.09 | 0.22 | 0.24 |
| 13_DenseNet121TransitionLayer | `mim_mlir_llvm` | 0.07 | 0.35 | 0.04 | 0.04 | 0.10 | 0.53 | 0.60 |
| 13_DenseNet121TransitionLayer | `mlir_llvm` | 0.59 | 0.28 | 0.04 | 0.04 | 0.10 | 0.46 | 1.05 |
| 14_DenseNet121DenseBlock | `mim_llvm` | 0.07 | 0.19 | 0.35 | 0.29 | 0.10 | 0.93 | 1.00 |
| 14_DenseNet121DenseBlock | `mim_mlir_llvm` | 0.15 | 0.42 | 1.54 | 1.04 | 0.10 | 3.11 | 3.27 |
| 14_DenseNet121DenseBlock | `mlir_llvm` | 0.68 | 0.45 | 1.53 | 0.99 | 0.11 | 3.09 | 3.77 |
| 15_DenseNet121 | `mim_llvm` | 0.81 | 3.49 | 15.14 | 2.53 | 0.14 | 21.30 | 22.11 |
| 15_DenseNet121 | `mim_mlir_llvm` | 1.13 | 1.89 | 19.20 | 2.40 | 0.30 | 23.79 | 24.91 |
| 15_DenseNet121 | `mlir_llvm` | 1.47 | 2.88 | 18.48 | 3.05 | 0.31 | 24.72 | 26.19 |
| 16_DenseNet201 | `mim_llvm` | 1.29 | 8.50 | 38.46 | 5.23 | 0.17 | 52.36 | 53.65 |
| 16_DenseNet201 | `mim_mlir_llvm` | 2.06 | 3.56 | 45.25 | 4.59 | 0.49 | 53.90 | 55.96 |
| 16_DenseNet201 | `mlir_llvm` | 2.12 | 6.07 | 42.35 | 5.64 | 0.42 | 54.48 | 56.60 |
| 18_SqueezeNet | `mim_llvm` | 0.12 | 0.83 | 1.56 | 1.12 | 0.10 | 3.62 | 3.74 |
| 18_SqueezeNet | `mim_mlir_llvm` | 0.17 | 0.52 | 1.51 | 0.65 | 0.12 | 2.80 | 2.97 |
| 18_SqueezeNet | `mlir_llvm` | 0.73 | 0.53 | 1.59 | 0.81 | 0.13 | 3.06 | 3.78 |
| 19_MobileNetV1 | `mim_llvm` | 0.23 | 0.66 | 1.05 | 0.35 | 0.12 | 2.18 | 2.40 |
| 19_MobileNetV1 | `mim_mlir_llvm` | 0.36 | 0.67 | 5.07 | 1.02 | 0.17 | 6.92 | 7.29 |
| 19_MobileNetV1 | `mlir_llvm` | 0.81 | 1.10 | 4.65 | 1.09 | 0.17 | 7.01 | 7.82 |
| 20_MobileNetV2 | `mim_llvm` | 0.44 | 1.41 | 2.43 | 0.65 | 0.13 | 4.62 | 5.06 |
| 20_MobileNetV2 | `mim_mlir_llvm` | 0.60 | 1.06 | 15.79 | 2.46 | 0.26 | 19.57 | 20.17 |
| 20_MobileNetV2 | `mlir_llvm` | 1.09 | 2.06 | 14.17 | 2.53 | 0.27 | 19.02 | 20.11 |
| 21_EfficientNetMBConv | `mim_llvm` | 0.03 | 0.12 | 0.15 | 0.10 | 0.10 | 0.46 | 0.49 |
| 21_EfficientNetMBConv | `mim_mlir_llvm` | 0.10 | 0.76 | 0.54 | 0.34 | 0.11 | 1.75 | 1.85 |
| 21_EfficientNetMBConv | `mlir_llvm` | 0.62 | 0.35 | 0.49 | 0.31 | 0.11 | 1.26 | 1.88 |
| 22_EfficientNetB0 | `mim_llvm` | 0.38 | 1.29 | 1.97 | 0.63 | 0.13 | 4.02 | 4.40 |
| 22_EfficientNetB0 | `mim_mlir_llvm` | 0.60 | 0.92 | 12.76 | 2.13 | 0.23 | 16.04 | 16.63 |
| 22_EfficientNetB0 | `mlir_llvm` | 0.97 | 1.63 | 11.69 | 2.20 | 0.22 | 15.74 | 16.71 |
| 23_EfficientNetB1 | `mim_llvm` | 0.19 | 0.79 | 0.88 | 0.42 | 0.11 | 2.19 | 2.38 |
| 23_EfficientNetB1 | `mim_mlir_llvm` | 0.37 | 0.67 | 8.46 | 1.52 | 0.15 | 10.80 | 11.17 |
| 23_EfficientNetB1 | `mlir_llvm` | 0.87 | 1.02 | 7.60 | 1.64 | 0.15 | 10.42 | 11.29 |
| 24_EfficientNetB2 | `mim_llvm` | 0.18 | 0.53 | 0.47 | 0.33 | 0.12 | 1.45 | 1.63 |
| 24_EfficientNetB2 | `mim_mlir_llvm` | 0.32 | 0.65 | 1.40 | 0.55 | 0.15 | 2.74 | 3.06 |
| 24_EfficientNetB2 | `mlir_llvm` | 0.81 | 0.83 | 1.16 | 0.54 | 0.15 | 2.68 | 3.48 |
| 25_ShuffleNetUnit | `mim_llvm` | 0.04 | 0.17 | 0.16 | 0.15 | 0.10 | 0.58 | 0.63 |
| 25_ShuffleNetUnit | `mim_mlir_llvm` | 0.11 | 0.40 | 0.65 | 0.59 | 0.11 | 1.75 | 1.86 |
| 25_ShuffleNetUnit | `mlir_llvm` | 0.66 | 0.37 | 0.64 | 0.54 | 0.11 | 1.66 | 2.32 |
| 26_ShuffleNet | `mim_llvm` | 0.41 | 1.89 | 4.58 | 0.94 | 0.13 | 7.54 | 7.94 |
| 26_ShuffleNet | `mim_mlir_llvm` | 0.51 | 0.99 | 34.86 | 3.05 | 0.22 | 39.12 | 39.64 |
| 26_ShuffleNet | `mlir_llvm` | 0.98 | 1.87 | 33.64 | 3.02 | 0.23 | 38.76 | 39.73 |
| 27_RegNet | `mim_llvm` | 0.08 | 0.20 | 0.31 | 0.31 | 0.11 | 0.93 | 1.01 |
| 27_RegNet | `mim_mlir_llvm` | 0.14 | 0.46 | 1.71 | 0.72 | 0.12 | 3.01 | 3.15 |
| 27_RegNet | `mlir_llvm` | 0.69 | 0.45 | 1.48 | 0.57 | 0.12 | 2.62 | 3.31 |
| 28_VisionTransformer | `mim_llvm` | 0.19 | 1.99 | 3.12 | 0.98 | 0.11 | 6.21 | 6.40 |
| 28_VisionTransformer | `mim_mlir_llvm` | 0.30 | 1.76 | 7.14 | 3.27 | 0.13 | 12.30 | 12.60 |
| 28_VisionTransformer | `mlir_llvm` | 0.88 | 1.38 | 5.44 | 1.65 | 0.13 | 8.60 | 9.48 |
| 29_SwinMLP | `mim_llvm` | 0.53 | 1.86 | 7.34 | 2.03 | 0.12 | 11.35 | 11.88 |
| 29_SwinMLP | `mim_mlir_llvm` | 0.66 | 1.25 | 19.36 | 5.54 | 0.17 | 26.33 | 26.99 |
| 29_SwinMLP | `mlir_llvm` | 1.15 | 2.15 | 157.66 | 18.83 | 0.17 | 178.80 | 179.95 |
| 31_VisionAttention | `mim_llvm` | 0.03 | 2.19 | 0.18 | 0.15 | 0.10 | 2.63 | 2.65 |
| 31_VisionAttention | `mim_mlir_llvm` | 0.08 | 2.24 | 0.23 | 0.21 | 0.10 | 2.77 | 2.85 |
| 31_VisionAttention | `mlir_llvm` | 0.63 | 0.36 | 0.24 | 0.23 | 0.10 | 0.93 | 1.56 |
| 43_MinGPTCausalAttention | `mim_llvm` | 0.03 | 0.15 | 0.23 | 0.21 | 0.10 | 0.69 | 0.72 |
| 43_MinGPTCausalAttention | `mim_mlir_llvm` | 0.08 | 0.41 | 0.30 | 0.41 | 0.10 | 1.22 | 1.30 |
| 43_MinGPTCausalAttention | `mlir_llvm` | 0.65 | 0.34 | 0.21 | 0.27 | 0.10 | 0.93 | 1.58 |
| 44_MiniGPTBlock | `mim_llvm` | 0.04 | 0.23 | 0.29 | 0.25 | 0.10 | 0.87 | 0.91 |
| 44_MiniGPTBlock | `mim_mlir_llvm` | 0.11 | 0.45 | 0.53 | 0.53 | 0.11 | 1.61 | 1.72 |
| 44_MiniGPTBlock | `mlir_llvm` | 0.76 | 0.47 | 0.39 | 0.37 | 0.11 | 1.34 | 2.10 |
| 45_UNetSoftmax | `mim_llvm` | 0.20 | 0.86 | 7.24 | 2.70 | 0.12 | 10.92 | 11.12 |
| 45_UNetSoftmax | `mim_mlir_llvm` | 0.29 | 0.80 | 28.13 | 5.78 | 0.16 | 34.87 | 35.16 |
| 45_UNetSoftmax | `mlir_llvm` | 0.77 | 1.05 | 27.15 | 5.61 | 0.16 | 33.97 | 34.73 |
| 46_NetVladWithGhostClusters | `mim_llvm` | 0.02 | 0.09 | 0.09 | 0.07 | 0.10 | 0.35 | 0.37 |
| 46_NetVladWithGhostClusters | `mim_mlir_llvm` | 0.07 | 0.38 | 0.25 | 0.11 | 0.10 | 0.84 | 0.91 |
| 46_NetVladWithGhostClusters | `mlir_llvm` | 0.64 | 0.34 | 0.23 | 0.13 | 0.10 | 0.80 | 1.45 |
| 47_NetVladNoGhostClusters | `mim_llvm` | 0.02 | 0.09 | 0.08 | 0.06 | 0.10 | 0.34 | 0.35 |
| 47_NetVladNoGhostClusters | `mim_mlir_llvm` | 0.07 | 0.37 | 0.18 | 0.10 | 0.11 | 0.75 | 0.81 |
| 47_NetVladNoGhostClusters | `mlir_llvm` | 0.64 | 0.33 | 0.19 | 0.12 | 0.10 | 0.74 | 1.38 |
| 50_ReLUSelfAttention | `mim_llvm` | 0.02 | 0.10 | 0.24 | 0.18 | 0.10 | 0.62 | 0.64 |
| 50_ReLUSelfAttention | `mim_mlir_llvm` | 0.06 | 0.41 | 0.27 | 0.31 | 0.10 | 1.09 | 1.15 |
| 50_ReLUSelfAttention | `mlir_llvm` | 0.65 | 0.31 | 0.21 | 0.24 | 0.10 | 0.86 | 1.51 |
