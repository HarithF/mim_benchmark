# Geometric means: gpu

## GPU run time relative to `iree_cuda` (33 models)

| model | `mim_nvptx` |
|---|---:|
| MLP | 0.85 |
| ShallowWideMLP | 0.84 |
| DeepNarrowMLP | 0.77 |
| LeNet5 | 0.95 |
| AlexNet | 0.59 |
| GoogleNetInceptionModule | 0.88 |
| GoogleNetInceptionV1 | 1.46 |
| ResNetBasicBlock | 0.91 |
| ResNet18 | 1.03 |
| VGG16 | 2.18 |
| VGG19 | 2.37 |
| DenseNet121TransitionLayer | 0.62 |
| DenseNet121DenseBlock | 1.79 |
| DenseNet121 | 1.96 |
| DenseNet201 | 2.10 |
| SqueezeNet | 0.84 |
| MobileNetV1 | 1.06 |
| MobileNetV2 | 1.03 |
| EfficientNetMBConv | 0.65 |
| EfficientNetB0 | 1.05 |
| EfficientNetB1 | 1.00 |
| EfficientNetB2 | 0.92 |
| ShuffleNetUnit | 1.84 |
| ShuffleNet | 1.55 |
| RegNet | 1.84 |
| VisionTransformer | 1.28 |
| SwinMLP | 2.25 |
| MinGPTCausalAttention | 2.48 |
| MiniGPTBlock | 2.25 |
| UNetSoftmax | 1.86 |
| NetVladWithGhostClusters | 1.49 |
| NetVladNoGhostClusters | 1.46 |
| ReLUSelfAttention | 2.15 |
| **geometric mean** | **1.28** |
| faster than the reference | 12 of 33 |

## GPU compile time, geometric mean in seconds

| column | models | `mim_nvptx` | `iree_cuda` |
|---|---:|---:|---:|
| `frontend_s` | 33 | 0.64 | 1.43 |
| `total_s` | 33 | 0.98 | 2.75 |
| `end_to_end_s` | 33 | 1.22 | 2.75 |

Columns as in `compile_times.md`; each is averaged over the models that have it for every path.

Not included: `10_ResNet101` (IREE's CUDA output fails the correctness check) and
`31_VisionAttention` (needs more than 16 GB of GPU memory). Both work on the CPU, relative to
`mlir_llvm`: `10_ResNet101` 0.98 (`mim_llvm`) and 1.15 (`mim_mlir_llvm`), `31_VisionAttention`
0.85 and 0.88; see `../cpu/summary.md`.
