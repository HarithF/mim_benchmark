# Documentation

- [`artifact_evaluation.md`](artifact_evaluation.md): what the artifact evaluates, how runtime and
  compile time are measured, and what every compile-time step means.
- [`results_cpu.md`](results_cpu.md): runtime and compile-time results on the CPU (35 models,
  `mim_llvm` and `mim_mlir_llvm` against torch-mlir). Raw data and geometric means in
  [`results/cpu/`](results/cpu/).
- [`results_gpu.md`](results_gpu.md): GPU results (33 models, `mim_nvptx` against IREE, on an
  RTX 4060 Ti). Raw data and geometric means in [`results/gpu/`](results/gpu/).
