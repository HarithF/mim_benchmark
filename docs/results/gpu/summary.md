# Geometric means: gpu

## GPU run time relative to `iree_cuda` (33 models)

| model | `mim_nvptx` |
|---|---:|
| MLP | 0.87 |
| ShallowWideMLP | 0.83 |
| DeepNarrowMLP | 0.77 |
| LeNet5 | 1.01 |
| AlexNet | 0.64 |
| GoogleNetInceptionModule | 1.05 |
| GoogleNetInceptionV1 | 1.97 |
| ResNetBasicBlock | 1.28 |
| ResNet18 | 1.14 |
| VGG16 | 2.50 |
| VGG19 | 2.66 |
| DenseNet121TransitionLayer | 0.61 |
| DenseNet121DenseBlock | 3.12 |
| DenseNet121 | 2.99 |
| DenseNet201 | 3.18 |
| SqueezeNet | 1.61 |
| MobileNetV1 | 1.44 |
| MobileNetV2 | 1.56 |
| EfficientNetMBConv | 1.24 |
| EfficientNetB0 | 1.61 |
| EfficientNetB1 | 1.34 |
| EfficientNetB2 | 0.95 |
| ShuffleNetUnit | 1.98 |
| ShuffleNet | 1.70 |
| RegNet | 2.30 |
| VisionTransformer | 1.29 |
| SwinMLP | 2.42 |
| MinGPTCausalAttention | 2.47 |
| MiniGPTBlock | 2.24 |
| UNetSoftmax | 2.85 |
| NetVladWithGhostClusters | 1.49 |
| NetVladNoGhostClusters | 1.47 |
| ReLUSelfAttention | 2.25 |
| **geometric mean** | **1.56** |
| faster than the reference | 6 of 33 |

## GPU compile time, geometric mean in seconds

| column | models | `mim_nvptx` | `iree_cuda` |
|---|---:|---:|---:|
| `frontend_s` | 33 | 0.67 | 1.46 |
| `total_s` | 33 | 1.14 | 2.77 |
| `end_to_end_s` | 33 | 1.40 | 2.77 |

Columns as in `compile_times.md`; each is averaged over the models that have it for every path.
