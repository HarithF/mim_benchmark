#!/usr/bin/env python3
"""Benchmark driver for an IREE-compiled model, with the C drivers' command line.

    iree_driver.py META.iree.json <weights.bin> <input.bin> <reps> <output.bin> [run_id] [timing.json]

Weights are bound by name: <Model>_params.json (next to weights.bin) gives every
slot's name and offset, and those names are the parameters export_iree.py
externalized, so no .irpa file is needed. The device comes from META's target
(cuda -> "cuda", llvm-cpu -> "local-sync"; IREE_DEVICE overrides it).

Timing: the runtime, device, parameters and module are set up once; each rep
times one `main` call including the input upload and the output download
(to_host), i.e. steady state. `setup_ns` (session creation + parameter upload)
is reported separately.
"""
import glob
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np
import iree.runtime as rt

DTYPES = {"float32": np.float32, "int64": np.int64}


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    meta_path, weights_path, input_path, reps, output_path = sys.argv[1:6]
    run_id = sys.argv[6] if len(sys.argv) > 6 else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = sys.argv[7] if len(sys.argv) > 7 else None
    reps = int(reps)
    meta_dir = os.path.dirname(os.path.abspath(meta_path))
    meta = json.load(open(meta_path))

    manifests = glob.glob(os.path.join(os.path.dirname(os.path.abspath(weights_path)), "*_params.json"))
    if len(manifests) != 1:
        sys.exit(f"expected one *_params.json next to {weights_path}, found {manifests}")
    manifest = json.load(open(manifests[0]))
    blob = np.fromfile(weights_path, dtype=np.uint8)
    slots = {e["name"]: e for e in manifest["entries"] if e.get("named")}
    missing = [p for p in meta["params"] if p not in slots]
    if missing:
        sys.exit(f"parameters without a weights.bin slot: {missing}")

    device_name = os.environ.get("IREE_DEVICE") or {"cuda": "cuda", "llvm-cpu": "local-sync"}[meta["target"]]
    t0 = time.perf_counter_ns()
    config = rt.Config(device_name)
    index = rt.ParameterIndex()
    for name in meta["params"]:
        e = slots[name]
        nbytes = e["count"] * e["elem_size"]
        index.add_buffer(name, blob[e["byte_offset"] : e["byte_offset"] + nbytes].tobytes())
    params = rt.create_io_parameters_module(config.vm_instance, index.create_provider(scope=meta["scope"]))
    hal = rt.create_hal_module(config.vm_instance, config.device)
    model = rt.VmModule.mmap(config.vm_instance, os.path.join(meta_dir, meta["vmfb"]))
    modules = rt.load_vm_modules(params, hal, model, config=config)
    fn = getattr(modules[-1], meta["entry"])
    setup_ns = time.perf_counter_ns() - t0

    if len(meta["inputs"]) != 1:
        sys.exit("only single-input models are supported")
    spec = meta["inputs"][0]
    x = np.fromfile(input_path, dtype=DTYPES[spec["dtype"]]).reshape(spec["shape"])

    times, out = [], None
    for _ in range(max(reps, 1)):
        t = time.perf_counter_ns()
        res = fn(x)
        out = (res[0] if isinstance(res, (tuple, list)) else res).to_host()
        times.append(time.perf_counter_ns() - t)
    np.ascontiguousarray(out).tofile(output_path)

    report = {
        "model": manifest["model"],
        "pipeline": os.path.basename(meta_dir),
        "run_id": run_id,
        "weights_path": weights_path,
        "input_path": input_path,
        "output_path": output_path,
        "reps": reps,
        "median_ns": float(statistics.median(times)),
        "mean_ns": float(statistics.fmean(times)),
        "min_ns": float(min(times)),
        "max_ns": float(max(times)),
        "setup_ns": float(setup_ns),
        "device": device_name,
    }
    print(f"pipeline={report['pipeline']} device={device_name} reps={reps} median_ns={report['median_ns']} "
          f"setup_ns={setup_ns}")
    if json_path:
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote timing report to {json_path}")


if __name__ == "__main__":
    main()
