# Geometric means: 20261114_120909

Restricted to the 35 models of `models.txt`.

`10_ResNet101` (0.98 `mim_llvm`, 1.15 `mim_mlir_llvm`) and `31_VisionAttention` (0.85, 0.88) are
included here but not in the GPU summary: on the GPU, IREE's output for `10_ResNet101` fails the
correctness check, and `31_VisionAttention` needs more than 16 GB of GPU memory.

## CPU run time relative to `mlir_llvm` (35 models)

| model | `mim_llvm` | `mim_mlir_llvm` |
|---|---:|---:|
| MLP | 0.43 | 0.56 |
| ShallowWideMLP | 0.37 | 0.49 |
| DeepNarrowMLP | 0.87 | 1.03 |
| LeNet5 | 1.26 | 0.99 |
| AlexNet | 1.10 | 1.29 |
| GoogleNetInceptionModule | 0.36 | 0.66 |
| GoogleNetInceptionV1 | 0.91 | 1.10 |
| ResNetBasicBlock | 0.59 | 0.99 |
| ResNet18 | 1.17 | 1.21 |
| ResNet101 | 0.98 | 1.15 |
| VGG16 | 0.79 | 0.96 |
| VGG19 | 0.84 | 0.97 |
| DenseNet121TransitionLayer | 0.95 | 0.99 |
| DenseNet121DenseBlock | 0.62 | 0.99 |
| DenseNet121 | 0.84 | 1.13 |
| DenseNet201 | 0.88 | 1.23 |
| SqueezeNet | 0.59 | 0.97 |
| MobileNetV1 | 0.73 | 0.99 |
| MobileNetV2 | 0.75 | 1.02 |
| EfficientNetMBConv | 0.29 | 0.58 |
| EfficientNetB0 | 0.78 | 1.04 |
| EfficientNetB1 | 0.56 | 1.04 |
| EfficientNetB2 | 0.58 | 0.86 |
| ShuffleNetUnit | 0.25 | 0.60 |
| ShuffleNet | 0.41 | 0.60 |
| RegNet | 0.67 | 1.02 |
| VisionTransformer | 0.91 | 0.95 |
| SwinMLP | 1.42 | 1.12 |
| VisionAttention | 0.85 | 0.88 |
| MinGPTCausalAttention | 1.50 | 1.15 |
| MiniGPTBlock | 1.21 | 1.23 |
| UNetSoftmax | 0.77 | 0.99 |
| NetVladWithGhostClusters | 0.95 | 0.78 |
| NetVladNoGhostClusters | 0.98 | 0.78 |
| ReLUSelfAttention | 1.11 | 1.07 |
| **geometric mean** | **0.74** | **0.93** |
| faster than the reference | 28 of 35 | 20 of 35 |

## CPU compile time, geometric mean in seconds

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

Columns as in `compile_times.md`; each is averaged over the models that have it for every path.
