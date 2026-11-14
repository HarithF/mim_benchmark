#!/usr/bin/env python3
"""Export a KernelBench model with iree-turbine and compile it with iree-compile.

    export_iree.py MODEL.py OUT_DIR [--target cuda|llvm-cpu] [--sm 86] [--timeout S]

Writes OUT_DIR/<stem>.mlir, OUT_DIR/<stem>_<target>.vmfb and OUT_DIR/<stem>.iree.json
(entry point, parameter scope and names, input/output shapes, timings). Parameters
and persistent buffers are externalized by name (`named_parameters()` names, as in
<Model>_params.json), so the driver can bind them from the benchmark's weights.bin
and the .vmfb holds code only. Needs a Python with torch + iree-turbine (IREE_PY).
"""
import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from iree.turbine import aot

SCOPE = "model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--target", default="cuda", choices=("cuda", "llvm-cpu"))
    ap.add_argument("--sm", default=None, help="CUDA compute capability, e.g. 86 (cuda only)")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = args.model.stem

    spec = importlib.util.spec_from_file_location("kb_model", args.model)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.Model(*mod.get_init_inputs()).eval()
    inputs = mod.get_inputs()

    t0 = time.perf_counter()
    aot.externalize_module_parameters(model, external_scope=SCOPE)
    exported = aot.export(model, *inputs)
    mlir = out / f"{stem}.mlir"
    exported.save_mlir(str(mlir))
    t1 = time.perf_counter()

    vmfb = out / f"{stem}_{args.target}.vmfb"
    iree_compile = shutil.which("iree-compile", path=str(Path(sys.executable).parent)) or "iree-compile"
    cmd = [iree_compile, str(mlir), "-o", str(vmfb)]
    if args.target == "cuda":
        cmd.append("--iree-hal-target-device=cuda")
        if args.sm:
            cmd.append(f"--iree-cuda-target=sm_{args.sm}")
    else:
        cmd += ["--iree-hal-target-device=local", "--iree-hal-local-target-device-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    t2 = time.perf_counter()
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        sys.exit(1)

    text = mlir.read_text()
    params = re.findall(r'parameter\.named<"' + SCOPE + r'"::"([^"]+)">', text)
    with torch.no_grad():
        outputs = model(*inputs)
    outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    meta = {
        "model": stem,
        "target": args.target,
        "sm": args.sm,
        "vmfb": vmfb.name,
        "entry": "main",
        "scope": SCOPE,
        "params": params,
        "inputs": [{"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")} for t in inputs],
        "outputs": [{"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")} for t in outputs],
        "timings": {"export_s": t1 - t0, "iree_compile_s": t2 - t1},
    }
    (out / f"{stem}.iree.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{stem}: export {t1 - t0:.1f}s, iree-compile {t2 - t1:.1f}s, {len(params)} params -> {vmfb.name}")


if __name__ == "__main__":
    main()
