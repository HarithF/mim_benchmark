#!/usr/bin/env python3
"""Benchmark driver for an IREE-compiled model, with the C drivers' command line.

    iree_driver.py META.iree.json <weights.bin> <input.bin> <reps> <output.bin> [run_id] [timing.json]

Weights are bound by name: <Model>_params.json (next to weights.bin) gives every
slot's name and offset, and those names are the parameters export_iree.py
externalized, so no .irpa file is needed. The device comes from META's target
(cuda -> "cuda", llvm-cpu -> "local-sync"; IREE_DEVICE overrides it).

Timing follows the C drivers, which run every call in a fresh child that has the
weights in memory already: every rep runs in a fresh process and times the device
setup, the parameter upload, the module load and one `main` call including the
input upload and the output download. Reading weights.bin and the input and
importing the runtime are not timed.
"""
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import iree.runtime as rt

DTYPES = {"float32": np.float32, "int64": np.int64}
CHILD = "--child"


class Model:
    """Everything a call needs that is read from disk, so it stays untimed."""

    def __init__(self, meta_path, weights_path, input_path):
        self.meta_dir = os.path.dirname(os.path.abspath(meta_path))
        self.meta = json.load(open(meta_path))
        manifests = glob.glob(os.path.join(os.path.dirname(os.path.abspath(weights_path)), "*_params.json"))
        if len(manifests) != 1:
            sys.exit(f"expected one *_params.json next to {weights_path}, found {manifests}")
        self.manifest = json.load(open(manifests[0]))
        blob = np.fromfile(weights_path, dtype=np.uint8)
        slots = {e["name"]: e for e in self.manifest["entries"] if e.get("named")}
        missing = [p for p in self.meta["params"] if p not in slots]
        if missing:
            sys.exit(f"parameters without a weights.bin slot: {missing}")
        self.params = {}
        for name in self.meta["params"]:
            e = slots[name]
            self.params[name] = blob[e["byte_offset"] : e["byte_offset"] + e["count"] * e["elem_size"]].tobytes()

        if len(self.meta["inputs"]) != 1:
            sys.exit("only single-input models are supported")
        spec = self.meta["inputs"][0]
        self.x = np.fromfile(input_path, dtype=DTYPES[spec["dtype"]]).reshape(spec["shape"])
        self.device = os.environ.get("IREE_DEVICE") or {"cuda": "cuda", "llvm-cpu": "local-sync"}[self.meta["target"]]

    def run(self):
        """Sets the device and module up and calls `main` once; returns the output on the host."""
        config = rt.Config(self.device)
        index = rt.ParameterIndex()
        for name, data in self.params.items():
            index.add_buffer(name, data)
        params = rt.create_io_parameters_module(config.vm_instance, index.create_provider(scope=self.meta["scope"]))
        hal = rt.create_hal_module(config.vm_instance, config.device)
        model = rt.VmModule.mmap(config.vm_instance, os.path.join(self.meta_dir, self.meta["vmfb"]))
        modules = rt.load_vm_modules(params, hal, model, config=config)
        res = getattr(modules[-1], self.meta["entry"])(self.x)
        return (res[0] if isinstance(res, (tuple, list)) else res).to_host()


def child(meta_path, weights_path, input_path, output_path):
    """One rep: prints the elapsed ns; writes the output if output_path is not empty."""
    m = Model(meta_path, weights_path, input_path)
    t = time.perf_counter_ns()
    out = m.run()
    elapsed = time.perf_counter_ns() - t
    if output_path:
        np.ascontiguousarray(out).tofile(output_path)
    print(elapsed)


def rep(meta_path, weights_path, input_path, output_path, what):
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), CHILD, meta_path, weights_path, input_path, output_path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"{what} failed (exit {proc.returncode}):\n{proc.stderr}")
    return float(proc.stdout.strip().splitlines()[-1])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == CHILD:
        return child(*sys.argv[2:6])
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    meta_path, weights_path, input_path, reps, output_path = sys.argv[1:6]
    run_id = sys.argv[6] if len(sys.argv) > 6 else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = sys.argv[7] if len(sys.argv) > 7 else None
    reps = max(int(reps), 1)
    m = Model(meta_path, weights_path, input_path)

    # Like the C drivers: one warm-up call in its own process, then the timed ones.
    rep(meta_path, weights_path, input_path, "", "warm-up")
    times = [rep(meta_path, weights_path, input_path, output_path if i == reps - 1 else "", f"rep {i}")
             for i in range(reps)]

    report = {
        "model": m.manifest["model"],
        "pipeline": os.path.basename(m.meta_dir),
        "run_id": run_id,
        "weights_path": weights_path,
        "input_path": input_path,
        "output_path": output_path,
        "reps": reps,
        "median_ns": float(statistics.median(times)),
        "mean_ns": float(statistics.fmean(times)),
        "min_ns": float(min(times)),
        "max_ns": float(max(times)),
        "device": m.device,
    }
    print(f"pipeline={report['pipeline']} device={m.device} reps={reps} median_ns={report['median_ns']}")
    if json_path:
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote timing report to {json_path}")


if __name__ == "__main__":
    main()
