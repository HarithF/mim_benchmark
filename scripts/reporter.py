#!/usr/bin/env python3
"""
reporter.py -- run every compiled variant of each model, check its output
against a reference, and write one JSON report per model with correctness
and timings.

ROOT is a model directory (e.g. models/11_VGG16) or a directory of them. A
model directory holds <Name>_weights.bin, <Name>_input.bin (gen_data.py) and
bin/ with one executable per variant (compile_all.sh): mim_llvm,
mim_mlir_llvm, mlir_llvm. Whatever is in bin/ is run; nothing is hardcoded.

Reference: the PyTorch model in <model_dir>/torch_ref/*.py, with its weights
loaded BY NAME from weights.bin through <Name>_params.json (see
param_manifest.py) and run on input.bin. If there is no usable reference,
mim_llvm's output (or the first variant's) is the baseline instead.

PASS: max|pred - ref| / mean|ref| <= rtol + atol / mean|ref|, i.e. the error
is judged against the output's own scale. Variants sum in different orders,
so they are expected to be faithful, not bit-identical.

Timing: each driver writes a JSON timing report (median/mean/min/max over the
reps); the report carries it per variant and a `timing_summary` with
`rel_median`, the ratio to the fastest variant.

Report: <out_dir>/<model_dir>_<run_id>.json (out_dir: the model directory).

Usage:
    python3 reporter.py ./models/11_VGG16
    python3 reporter.py ./models --reps 5 --interleave --pin-cpu 2 --timeout 1200
"""

import argparse
import importlib.util
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

import param_manifest

# ---------------------------------------------------------------------------
# LLVM signature parsing (as in gen_data.py / gen_drivers.py): recovers a
# model's input/output shapes from its mim .ll, best effort.
# ---------------------------------------------------------------------------

DEFINE_RE = re.compile(
    r"define\s+.*?@(?P<name>[A-Za-z_][\w.]*)\s*\((?P<args>.*?)\)\s*\{",
    re.DOTALL,
)
ADDRSPACE_RE = re.compile(r"\baddrspace\(\d+\)\s*")


def strip_addrspace(s):
    return ADDRSPACE_RE.sub("", s)


def split_top_level(s, sep=","):
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def parse_array_type(type_str):
    s = type_str.strip()
    if s.startswith("["):
        if not s.endswith("]"):
            raise ValueError(f"Malformed array type: {type_str!r}")
        inner = s[1:-1].strip()
        depth, split_at, i = 0, None, 0
        while i < len(inner):
            c = inner[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            elif depth == 0 and inner[i : i + 3] == " x ":
                split_at = i
                break
            i += 1
        if split_at is None:
            raise ValueError(f"Could not find ' x ' separator in: {type_str!r}")
        n = int(inner[:split_at].strip())
        rest = inner[split_at + 3 :].strip()
        dims, base = parse_array_type(rest)
        return [n] + dims, base
    return [], s


def strip_name_suffix(param_str):
    s = param_str.strip()
    m = re.match(r"^(.*?)\s+%\S+$", s)
    return m.group(1).strip() if m else s


def parse_pointee_array(type_with_ptr):
    t = strip_addrspace(type_with_ptr).strip().rstrip("*").strip()
    return parse_array_type(t)


def classify_arg_types(arg_type_strs):
    cleaned = [strip_addrspace(t).strip() for t in arg_type_strs]
    if cleaned and all(t.startswith("[") for t in cleaned):
        return "raw_flat"
    return "unknown"


def parse_raw_signature(text):
    """(name, params [(dims, base)], return_dims, return_base) of a raw-pointer
    .ll, or None for another calling convention."""
    m = DEFINE_RE.search(text)
    if not m:
        return None
    name = m.group("name")
    args_str = m.group("args")
    ret_m = re.search(
        r"define\s+(.*?)\s+@" + re.escape(name) + r"\s*\(", text, re.DOTALL
    )
    ret_type_str = strip_addrspace(ret_m.group(1).strip()).rstrip("*").strip()

    raw_params = split_top_level(args_str)
    arg_type_strs = [strip_name_suffix(p) for p in raw_params]
    style = classify_arg_types(arg_type_strs)
    if style == "unknown":
        return None

    try:
        params = [parse_pointee_array(f) for f in arg_type_strs]
        return_dims, return_base = parse_array_type(ret_type_str)
    except Exception:
        return None
    return name, params, return_dims, return_base


def total_elems(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def drop_weight_bias_pairs(params, candidates):
    """Drop candidates that are a Linear/Conv weight (a multi-dim parameter
    directly followed by a 1-D bias of its leading size). Must agree with
    gen_data.py and gen_drivers.py on which parameter is the input."""
    kept = []
    for i in candidates:
        nxt_dims = params[i + 1][0] if i + 1 < len(params) else None
        is_weight_of_pair = (
            len(params[i][0]) > 1
            and nxt_dims is not None
            and len(nxt_dims) == 1
            and nxt_dims[0] == params[i][0][0]
        )
        if not is_weight_of_pair:
            kept.append(i)
    return kept or candidates


def identify_input_index(params, return_dims):
    """The input: the parameter whose leading dim is the batch size of the
    output, multi-dim preferred, largest on ties (as in gen_data.py and
    gen_drivers.py)."""
    batch = return_dims[0] if return_dims else None
    candidates = [
        i
        for i, (dims, _base) in enumerate(params)
        if dims and batch is not None and dims[0] == batch and len(dims) > 1
    ]
    if not candidates:
        candidates = [
            i
            for i, (dims, _base) in enumerate(params)
            if dims and batch is not None and dims[0] == batch
        ]
    if not candidates:
        candidates = list(range(len(params)))
    candidates = drop_weight_bias_pairs(params, candidates)
    return max(candidates, key=lambda i: total_elems(params[i][0]))


class ModelShape:
    __slots__ = ("name", "input_dims", "output_dims")

    def __init__(self, name, input_dims, output_dims):
        self.name = name
        self.input_dims = input_dims
        self.output_dims = output_dims


def find_shape_source(model_dir, expected_name=None):
    """ModelShape from the model's raw-pointer .ll (mim_llvm/<Name>_mim.ll
    preferred), or None."""
    candidates = []
    for dirpath, _dirs, filenames in os.walk(model_dir):
        for fn in filenames:
            if fn.endswith(".ll"):
                candidates.append(os.path.join(dirpath, fn))

    def sort_key(path):
        is_canonical = os.path.basename(
            os.path.dirname(path)
        ) == "mim_llvm" and os.path.basename(path).endswith("_mim.ll")
        return (0 if is_canonical else 1, path)

    for path in sorted(candidates, key=sort_key):
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            continue
        parsed = parse_raw_signature(text)
        if parsed is None:
            continue
        name, params, return_dims, _return_base = parsed
        if expected_name is not None and name != expected_name:
            continue
        if None in return_dims or any(None in d for d, _b in params):
            continue
        input_idx = identify_input_index(params, return_dims)
        input_dims = params[input_idx][0]
        return ModelShape(name, input_dims, return_dims)
    return None


# ---------------------------------------------------------------------------
# Model / pipeline discovery
# ---------------------------------------------------------------------------


def looks_like_model_dir(d):
    """A model dir directly contains '<Name>_weights.bin' (written by
    gen_data.py)."""
    if not os.path.isdir(d):
        return False
    try:
        return any(fn.endswith("_weights.bin") for fn in os.listdir(d))
    except OSError:
        return False


def find_model_dirs(root):
    if looks_like_model_dir(root):
        return [root]
    found = []
    for dirpath, dirnames, _filenames in os.walk(root):
        # Don't descend into a model's own bin/outputs/torch_ref dirs looking
        # for nested "models" -- they won't match anyway, but skip for speed.
        dirnames[:] = [d for d in dirnames if d not in ("bin", "outputs", "torch_ref")]
        if looks_like_model_dir(dirpath):
            found.append(dirpath)
            dirnames[:] = []  # don't look for models nested inside a model dir
    return sorted(found)


def find_weights_and_input(model_dir):
    """Returns (name, weights_path, input_path) or None. 'name' is the
    <Name> prefix shared by <Name>_weights.bin / <Name>_input.bin."""
    weights_candidates = sorted(
        f for f in os.listdir(model_dir) if f.endswith("_weights.bin")
    )
    if not weights_candidates:
        return None
    if len(weights_candidates) > 1:
        print(
            f"warn: multiple *_weights.bin in {model_dir}, using {weights_candidates[0]}",
            file=sys.stderr,
        )
    weights_file = weights_candidates[0]
    name = weights_file[: -len("_weights.bin")]
    weights_path = os.path.join(model_dir, weights_file)
    input_path = os.path.join(model_dir, f"{name}_input.bin")
    if not os.path.isfile(input_path):
        print(
            f"warn: found {weights_file} but no {name}_input.bin in {model_dir}",
            file=sys.stderr,
        )
        return None
    return name, weights_path, input_path


def find_pipeline_binaries(model_dir):
    """Returns {variant_name: path} for every executable file directly under
    model_dir/bin/, discovered dynamically -- not a hardcoded variant list."""
    bin_dir = os.path.join(model_dir, "bin")
    if not os.path.isdir(bin_dir):
        return {}
    pipelines = {}
    for fn in sorted(os.listdir(bin_dir)):
        path = os.path.join(bin_dir, fn)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            pipelines[fn] = path
    return pipelines


def find_torch_ref_module(model_dir, name):
    """Look for model_dir/torch_ref/*.py. Prefers '<Name>_param.py'; if
    exactly one .py file exists, uses that; otherwise warns and gives up."""
    ref_dir = os.path.join(model_dir, "torch_ref")
    if not os.path.isdir(ref_dir):
        return None
    py_files = sorted(f for f in os.listdir(ref_dir) if f.endswith(".py"))
    if not py_files:
        return None
    preferred = f"{name}_param.py"
    if preferred in py_files:
        return os.path.join(ref_dir, preferred)
    if len(py_files) == 1:
        return os.path.join(ref_dir, py_files[0])
    print(
        f"warn: multiple candidate torch reference files in {ref_dir} and "
        f"none named {preferred!r} -- skipping torch reference for {name}",
        file=sys.stderr,
    )
    return None


# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------


def run_torch_reference(module_path, weights_path, input_path, shape, manifest):
    """The PyTorch reference output as a flat float32 array, or None (with a
    warning) if it cannot be produced. `manifest` (<Name>_params.json) says
    which slice of weights.bin holds which named parameter."""
    # Each slot is read with its own element type: a blob may mix them
    # (30_SwinTransformerV2 has i64 buffers among float parameters).
    base_dtype = {
        "float": np.float32,
        "double": np.float64,
        "half": np.float16,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
    }
    try:
        import torch
    except ImportError:
        print("warn: torch not installed -- skipping torch reference", file=sys.stderr)
        return None

    spec = importlib.util.spec_from_file_location("torch_ref_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"warn: failed to import {module_path}: {e}", file=sys.stderr)
        return None

    if not hasattr(mod, "Model"):
        print(f"warn: {module_path} has no `Model` class -- skipping", file=sys.stderr)
        return None

    # Construct with get_init_inputs() (the KernelBench convention); no-arg and
    # num_classes= are fallbacks for files without it.
    model = None
    ctor_attempts = []
    if hasattr(mod, "get_init_inputs"):
        try:
            init = mod.get_init_inputs()
            init = list(init) if isinstance(init, (list, tuple)) else [init]
            ctor_attempts.append((tuple(init), {}))
        except Exception as e:
            print(
                f"warn: {module_path}'s get_init_inputs() raised: {e}", file=sys.stderr
            )
    # Fallbacks, for reference files that predate the convention.
    ctor_attempts.append(((), {}))
    if shape is not None and shape.output_dims:
        ctor_attempts.append(((), {"num_classes": shape.output_dims[-1]}))

    last_err = None
    for a, kw in ctor_attempts:
        try:
            model = mod.Model(*a, **kw)
            break
        except TypeError as e:
            last_err = e
            continue
    if model is None:
        print(
            f"warn: could not construct {module_path}'s Model() -- tried "
            f"get_init_inputs(), no-arg and num_classes=; last error: {last_err} "
            f"-- skipping torch reference",
            file=sys.stderr,
        )
        return None
    model.eval()

    # Bind weights.bin BY NAME through the manifest. The blob holds parameters
    # and buffers (e.g. BatchNorm running stats) in the mim argument order, so
    # reading it sequentially would misplace them.
    if manifest is None:
        print(
            f"warn: no weight manifest for {module_path} -- run gen_data.py to "
            f"write <Name>_params.json; skipping torch reference",
            file=sys.stderr,
        )
        return None

    blob = open(weights_path, "rb").read()
    # Start from the module's own state: anything the manifest does not cover
    # (BatchNorm's num_batches_tracked) keeps its constructed value.
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    unmatched = []
    try:
        for e in manifest["entries"]:
            # `ref_name`: the state_dict key of a slot whose mim name differs
            # (44's closure `mlpf_closure_0_cell_contents.c_fc.weight` is
            # `mlp.c_fc.weight`).
            key = e.get("ref_name", e["name"])
            if not e.get("named", True) or key not in state:
                unmatched.append(key)
                continue
            want = state[key].numel()
            if want != e["count"]:
                raise RuntimeError(
                    f"'{key}' is {e['count']} elements in the manifest but "
                    f"{want} in the model"
                )
            base = e.get("base", "float")
            dtype = base_dtype.get(base, np.float32)
            elem_bytes = e.get("elem_size") or dtype().nbytes
            nbytes = e["count"] * elem_bytes
            off = e.get("byte_offset", e["offset"] * elem_bytes)
            seg = blob[off : off + nbytes]
            if len(seg) != nbytes:
                raise RuntimeError(
                    f"weights.bin is too short for '{key}': wanted "
                    f"{e['count']} {base} at byte {off}, got {len(seg)}"
                )
            chunk = np.frombuffer(seg, dtype=dtype)
            state[key] = torch.from_numpy(
                chunk.reshape(tuple(state[key].shape)).copy()
            ).to(state[key].dtype)
    except Exception as e:
        print(
            f"warn: failed to bind weights.bin to {module_path}'s Model: {e}",
            file=sys.stderr,
        )
        return None

    if unmatched:
        # A slot with no matching parameter means the reference .py is a
        # different model than the compiled one: refuse rather than half-bind.
        print(
            f"warn: {len(unmatched)} of {len(manifest['entries'])} weight "
            f"slots have no matching parameter in {module_path} "
            f"({', '.join(unmatched[:5])}{', ...' if len(unmatched) > 5 else ''})"
            f" -- the reference .py does not match the compiled model; "
            f"skipping torch reference",
            file=sys.stderr,
        )
        return None

    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"warn: load_state_dict failed for {module_path}: {e}", file=sys.stderr)
        return None

    x = np.fromfile(input_path, dtype=np.float32)

    # Input shape: get_inputs() first; the .ll's shape only as a fallback,
    # since mim drops unit dims there (LeNet5's [4096, 1, 32, 32] input is
    # declared [4096 x [32 x [32 x float]]]).
    candidate_dims = []
    if hasattr(mod, "get_inputs"):
        try:
            gi = mod.get_inputs()
            gi = list(gi) if isinstance(gi, (list, tuple)) else [gi]
            for t in gi:
                if hasattr(t, "shape"):
                    candidate_dims.append(list(t.shape))
        except Exception as e:
            print(f"warn: {module_path}'s get_inputs() raised: {e}", file=sys.stderr)
    if shape is not None and shape.input_dims:
        candidate_dims.append(list(shape.input_dims))

    x_t = None
    for dims in candidate_dims:
        if total_elems(dims) == x.size:
            x_t = torch.from_numpy(x.reshape(tuple(dims)).copy())
            break
    if x_t is None:
        if candidate_dims:
            print(
                f"warn: input.bin has {x.size} floats, matching none of the "
                f"candidate shapes {candidate_dims} -- skipping torch reference",
                file=sys.stderr,
            )
            return None
        x_t = torch.from_numpy(x.copy())

    try:
        with torch.no_grad():
            out = model(x_t)
    except Exception as e:
        print(f"warn: forward pass failed for {module_path}: {e}", file=sys.stderr)
        return None

    return out.numpy().astype(np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# Running compiled pipelines
# ---------------------------------------------------------------------------


def cpu_freq_state():
    """Governor, turbo and temperatures of the CPU, recorded in every report:
    without a fixed frequency (governor=performance, no turbo) two runs of the
    same binary are not comparable."""
    def rd(path):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return None

    state = {
        "governor": rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "no_turbo": rd("/sys/devices/system/cpu/intel_pstate/no_turbo"),
    }
    temps = []
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        v = rd(z)
        if v and v.isdigit():
            temps.append(int(v) // 1000)
    if temps:
        state["temps_c"] = temps
    state["stable"] = state["governor"] == "performance" and state["no_turbo"] == "1"
    return state


def _aggregate_rounds(samples, reps):
    """Stats over the interleaved samples, in the drivers' format. `spread_pct`
    (interquartile range / median) shows how large a difference the run can
    resolve."""
    xs = sorted(samples)
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    median = xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0

    def q(frac):
        if n == 1:
            return xs[0]
        pos = frac * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)

    iqr = q(0.75) - q(0.25)
    return {
        "reps": reps,
        "rounds": n,
        "median_ns": round(median, 1),
        "mean_ns": round(sum(xs) / n, 1),
        "min_ns": round(xs[0], 1),
        "max_ns": round(xs[-1], 1),
        "iqr_ns": round(iqr, 1),
        "spread_pct": round(100.0 * iqr / median, 2) if median else None,
        "samples_ns": [round(x, 1) for x in xs],
    }


def run_pipelines_interleaved(
    pipelines, weights_path, input_path, out_dir, rounds, timeout, run_id,
    expected_out_size, pin_cpu,
):
    """Run one rep of every variant per round, round-robin, so all variants see
    the same thermal and frequency conditions (back-to-back runs bias the
    ratios against whichever runs later)."""
    samples = {p: [] for p in pipelines}
    last = {p: None for p in pipelines}
    for r in range(rounds):
        for pname, binary in sorted(pipelines.items()):
            arr, timing = run_pipeline(
                pname, binary, weights_path, input_path, out_dir,
                1, timeout, f"{run_id}_r{r}", expected_out_size, pin_cpu=pin_cpu,
            )
            if arr is not None:
                last[pname] = arr
            if timing and timing.get("median_ns") is not None:
                samples[pname].append(timing["median_ns"])
    return last, {p: _aggregate_rounds(v, rounds) for p, v in samples.items()}


def run_pipeline(
    pname,
    binary,
    weights_path,
    input_path,
    out_dir,
    reps,
    timeout,
    run_id,
    expected_size=None,
    pin_cpu=None,
):
    """Run one variant's driver. Returns (output array or None, timing dict or
    None); the timing comes from the JSON report the driver writes, tagged
    with run_id."""
    out_path = os.path.join(out_dir, f"output_{pname}.bin")
    json_path = os.path.join(out_dir, f"timing_{pname}_{run_id}.json")
    cmd = [binary, weights_path, input_path, str(reps), out_path, run_id, json_path]
    # CPU pinning (--pin-cpu / PIN_CPU): the forked per-rep children inherit
    # the affinity. On a hybrid CPU an unpinned rep can land on a P-, E- or
    # low-power core, which can nearly double its time. Pin all variants of a
    # model to the same CPU, or their ratios are meaningless.
    if pin_cpu:
        cmd = ["taskset", "-c", str(pin_cpu)] + cmd
    print(f"[{pname}] running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[{pname}] FAILED: timed out after {timeout}s", file=sys.stderr)
        return None, None
    if proc.stdout:
        print(f"[{pname}] stdout:\n{proc.stdout.strip()}")
    if proc.returncode != 0:
        print(
            f"[{pname}] FAILED (exit {proc.returncode}):\n{proc.stderr}",
            file=sys.stderr,
        )
        return None, None
    if not os.path.isfile(out_path):
        print(f"[{pname}] FAILED: no output file written", file=sys.stderr)
        return None, None
    arr = np.fromfile(out_path, dtype=np.float32)
    if expected_size is not None and arr.size != expected_size:
        print(
            f"[{pname}] FAILED: output has {arr.size} floats, expected {expected_size}",
            file=sys.stderr,
        )
        return None, None

    timing = None
    if os.path.isfile(json_path):
        try:
            with open(json_path) as f:
                timing = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"[{pname}] warn: could not read timing report {json_path}: {e}",
                file=sys.stderr,
            )
    else:
        print(
            f"[{pname}] warn: driver did not write expected timing report {json_path}",
            file=sys.stderr,
        )

    return arr, timing


# ---------------------------------------------------------------------------
# Comparison / reporting
# ---------------------------------------------------------------------------


def compare(pname, pred, ref, output_dims, atol, rtol):
    if pred.size != ref.size:
        return {
            "status": "FAIL",
            "error": f"size mismatch: pred={pred.size} ref={ref.size}",
        }

    diff = np.abs(pred - ref)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    denom = np.abs(ref).copy()
    denom[denom == 0] = 1e-12
    max_rel = float((diff / denom).max())

    # PASS/FAIL is judged against the tensor's scale (mean|ref|): an absolute
    # tolerance cannot fit every model's magnitude, and pointwise diff/|ref|
    # explodes where ref is near zero.
    scale = float(np.abs(ref).mean())
    if scale == 0.0:
        scale = 1.0
    scaled = diff / scale
    max_scaled = float(scaled.max())
    rms_scaled = float(np.sqrt(np.mean((diff / scale) ** 2)))
    close = bool(max_scaled <= rtol + atol / scale)

    result = {
        "ref_mean_abs": scale,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "max_rel_diff": max_rel,
        "max_diff_over_scale": max_scaled,
        "rms_diff_over_scale": rms_scaled,
        "frac_pointwise_rel_gt_1e3": float((diff / denom > 1e-3).mean()),
        "allclose": close,
    }

    top1_ok = True
    if output_dims is not None and len(output_dims) == 2:
        batch, _ = output_dims
        if pred.size == batch * output_dims[1]:
            pred2 = pred.reshape(output_dims)
            ref2 = ref.reshape(output_dims)
            pred_top1 = pred2.argmax(axis=1)
            ref_top1 = ref2.argmax(axis=1)
            match = int((pred_top1 == ref_top1).sum())
            top1_ok = match == batch
            result["top1_match"] = f"{match}/{batch}"
            mismatches = [
                {
                    "row": i,
                    "ref_argmax": int(ref_top1[i]),
                    "ref_value": float(ref2[i, ref_top1[i]]),
                    "pred_argmax": int(pred_top1[i]),
                    "pred_value": float(pred2[i, pred_top1[i]]),
                }
                for i in range(batch)
                if pred_top1[i] != ref_top1[i]
            ]
            if mismatches:
                result["top1_mismatches"] = mismatches

    ok = close and top1_ok
    result["status"] = "PASS" if ok else "FAIL"

    line = (
        f"[{pname}] {result['status']}  max_diff/scale={max_scaled:.3g}  "
        f"rms_diff/scale={rms_scaled:.3g}  (scale={scale:.3g}  "
        f"max_abs={max_abs:.3g}  pointwise_max_rel={max_rel:.3g})"
    )
    if "top1_match" in result:
        line += f"  top1_match={result['top1_match']}"
    print(line)
    for m in result.get("top1_mismatches", []):
        print(
            f"    row {m['row']}: ref argmax={m['ref_argmax']} "
            f"({m['ref_value']:.4f}) vs pred argmax={m['pred_argmax']} "
            f"({m['pred_value']:.4f})"
        )
    return result


TIMING_KEYS = ("reps", "median_ns", "mean_ns", "min_ns", "max_ns",
               "iqr_ns", "spread_pct", "rounds", "samples_ns")


def summarize_timings(timings):
    """The report's `timing_summary`: per-variant stats from the drivers' JSON
    reports, plus `median_ms` and `rel_median` (ratio to the fastest
    variant's median)."""
    stats = {}
    for pname, t in timings.items():
        if not t or t.get("median_ns") is None:
            continue
        stats[pname] = {k: t[k] for k in TIMING_KEYS if k in t}

    medians = [
        s["median_ns"] for s in stats.values() if (s.get("median_ns") or 0) > 0
    ]
    fastest = min(medians) if medians else None

    for s in stats.values():
        m = s.get("median_ns")
        if m is not None:
            s["median_ms"] = round(m / 1e6, 3)
        if fastest and m:
            s["rel_median"] = round(m / fastest, 3)
    return stats


def print_timing_table(name, stats, reps):
    """One block per model, fastest first."""
    if not stats:
        print(f"[{name}] no timing reports collected")
        return
    print(f"[{name}] timings (median of {reps} rep(s), fastest first):")
    order = sorted(stats.items(), key=lambda kv: kv[1].get("median_ns", float("inf")))
    width = max(len(p) for p, _ in order)
    for pname, s in order:
        med = s.get("median_ns")
        if med is None:
            print(f"    {pname:<{width}}  (no timing)")
            continue
        rel = s.get("rel_median")
        rel_s = f"{rel:>6.2f}x" if rel is not None else "     -"
        sp = s.get("spread_pct")
        # A delta smaller than the spread is not a measurement.
        sp_s = f"  spread {sp:5.1f}%{'  NOISY' if sp > 5 else ''}" if sp is not None else ""
        print(
            f"    {pname:<{width}}  {med / 1e6:10.2f} ms  {rel_s}   "
            f"(min {s.get('min_ns', 0) / 1e6:.2f}  "
            f"max {s.get('max_ns', 0) / 1e6:.2f}  "
            f"mean {s.get('mean_ns', 0) / 1e6:.2f}){sp_s}"
        )


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------


def process_model(model_dir, args):
    dir_basename = os.path.basename(os.path.normpath(model_dir))
    print(f"\n{'=' * 70}\n=== model: {dir_basename} ({model_dir}) ===\n{'=' * 70}")

    wi = find_weights_and_input(model_dir)
    if wi is None:
        print(
            f"[{dir_basename}] SKIP: no <Name>_weights.bin/_input.bin found",
            file=sys.stderr,
        )
        return None
    name, weights_path, input_path = wi

    pipelines = find_pipeline_binaries(model_dir)
    if not pipelines:
        print(
            f"[{dir_basename}] SKIP: no executables found under {model_dir}/bin "
            f"(run compile_all.sh first)",
            file=sys.stderr,
        )
        return None

    shape = find_shape_source(model_dir, expected_name=name)
    if shape is None:
        print(
            f"[{name}] note: could not recover input/output shapes from any .ll "
            f"-- comparisons will run on flat arrays (no top-1 report)"
        )

    out_dir = args.out_dir_per_model or os.path.join(model_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    expected_out_size = total_elems(shape.output_dims) if shape else None

    try:
        manifest = param_manifest.find_manifest(model_dir, name)
    except Exception as e:
        print(f"[{name}] warn: {e}", file=sys.stderr)
        manifest = None

    ref_module = find_torch_ref_module(model_dir, name)
    ref = None
    ref_source = None
    if ref_module is not None:
        print(f"[{name}] found torch reference: {ref_module}")
        ref = run_torch_reference(
            ref_module, weights_path, input_path, shape, manifest
        )
        if ref is not None:
            ref_source = f"torch:{os.path.relpath(ref_module, model_dir)}"

    print(f"[{name}] pipelines found: {', '.join(sorted(pipelines))}")

    outputs = {}
    timings = {}
    if args.interleave:
        outs, timings = run_pipelines_interleaved(
            pipelines,
            weights_path,
            input_path,
            out_dir,
            args.reps,
            args.timeout,
            args.run_id,
            expected_out_size,
            args.pin_cpu,
        )
        outputs.update(outs)
    else:
        for pname, binary in sorted(pipelines.items()):
            arr, timing = run_pipeline(
                pname,
                binary,
                weights_path,
                input_path,
                out_dir,
                args.reps,
                args.timeout,
                args.run_id,
                expected_out_size,
                pin_cpu=args.pin_cpu,
            )
            outputs[pname] = arr
            timings[pname] = timing

    baseline_name = None
    if ref is None:
        # No PyTorch reference: compare against one variant (mim_llvm first).
        available = [p for p, arr in outputs.items() if arr is not None]
        if not available:
            print(
                f"[{name}] SKIP: no reference available and no pipeline produced output",
                file=sys.stderr,
            )
            return None
        baseline_name = "mim_llvm" if "mim_llvm" in available else sorted(available)[0]
        ref = outputs[baseline_name]
        ref_source = f"baseline:{baseline_name}"
        print(
            f"[{name}] no torch reference -- using pipeline '{baseline_name}' as baseline"
        )

    results = {}
    all_ok = True
    for pname, pred in sorted(outputs.items()):
        if pname == baseline_name:
            results[pname] = {"status": "REFERENCE"}
        elif pred is None:
            results[pname] = {"status": "SKIPPED"}
            all_ok = False
        else:
            r = compare(
                pname,
                pred,
                ref,
                shape.output_dims if shape else None,
                args.atol,
                args.rtol,
            )
            results[pname] = r
            all_ok = all_ok and (r["status"] == "PASS")
        if timings.get(pname) is not None:
            results[pname]["timing"] = timings[pname]

    timing_summary = summarize_timings(timings)
    _cpu = cpu_freq_state()
    if not _cpu.get("stable"):
        print(
            f"[{name}] WARNING: cpu is governor={_cpu.get('governor')} "
            f"no_turbo={_cpu.get('no_turbo')} -- frequency is not pinned, so "
            f"cross-run comparisons are unreliable. Set governor=performance and "
            f"no_turbo=1 before trusting deltas.",
            file=sys.stderr,
        )
    print_timing_table(name, timing_summary, args.reps)

    report = {
        "model_dir": dir_basename,
        "model_name": name,
        "run_id": args.run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reference_source": ref_source,
        "reps": args.reps,
        "pinned_cpu": args.pin_cpu,
        "atol": args.atol,
        "rtol": args.rtol,
        "output_shape": shape.output_dims if shape else None,
        "timing_summary": timing_summary,
        "cpu_state": cpu_freq_state(),
        "interleaved": bool(args.interleave),
        "fastest_pipeline": (
            min(timing_summary, key=lambda p: timing_summary[p]["median_ns"])
            if timing_summary
            else None
        ),
        "pipelines": results,
        "all_pass": all_ok,
    }

    report_dir = args.out_dir_reports or model_dir
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{dir_basename}_{args.run_id}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{name}] report written: {report_path}")
    print(f"[{name}] " + ("ALL PASS" if all_ok else "SOME COMPARISONS FAILED/SKIPPED"))

    return all_ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "root",
        nargs="?",
        default=".",
        help="A single model directory, or a root containing many",
    )
    ap.add_argument(
        "--reps", type=int, default=1, help="reps passed to each driver binary"
    )
    ap.add_argument(
        "--interleave",
        action="store_true",
        help="run one rep of every pipeline per round, round-robin, instead of "
        "running each pipeline's reps back-to-back. Removes the execution-order "
        "bias from the cross-pipeline ratios and yields spread_pct per pipeline."
    )
    # PASS: max|pred-ref| / mean|ref| <= rtol + atol/mean|ref|, so rtol is the
    # allowed error relative to the output's typical magnitude (1e-3, ~50x the
    # fp32 summation-order noise measured on these models).
    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="allowed max error as a fraction of mean|ref| (default 1e-3)",
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="absolute slack, added as atol/mean|ref|; negligible unless "
        "outputs are tiny (default 1e-2)",
    )
    ap.add_argument(
        "--out-dir",
        dest="out_dir_per_model",
        default=None,
        help="where each pipeline's raw output_<name>.bin goes (default: <model_dir>/outputs)",
    )
    ap.add_argument(
        "--report-dir",
        dest="out_dir_reports",
        default=None,
        help="where each model's JSON report goes (default: the model's own directory)",
    )
    ap.add_argument(
        "--timeout", type=int, default=600, help="per-pipeline run timeout, in seconds"
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="Shared id passed to every driver invocation and used in every "
        "report/timing filename this run produces (default: a generated "
        "UTC timestamp). Pass one explicitly to tie a reporter.py run to "
        "an id from elsewhere.",
    )

    ap.add_argument(
        "--pin-cpu",
        default=os.environ.get("PIN_CPU"),
        help="pin every pipeline run to these CPUs via `taskset -c` (e.g. "
        "'2', '0-5', '2,3'); defaults to $PIN_CPU. Strongly recommended on "
        "hybrid P/E-core machines -- see the comment in run_pipeline(). The "
        "value is recorded in each report as \"pinned_cpu\".",
    )

    args = ap.parse_args()

    if args.pin_cpu and shutil.which("taskset") is None:
        print(
            "error: --pin-cpu given but `taskset` is not on PATH "
            "(install util-linux, or drop the flag)",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.run_id is None:
        args.run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    if not os.path.isdir(args.root):
        print(f"error: not a directory: {args.root}", file=sys.stderr)
        sys.exit(1)

    model_dirs = find_model_dirs(args.root)
    if not model_dirs:
        print(
            f"error: no model directories found under {args.root} "
            f"(looked for a directory directly containing *_weights.bin)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"==> run_id: {args.run_id}")
    print(f"==> found {len(model_dirs)} model(s):")
    for d in model_dirs:
        print(f"      {d}")

    overall_ok = True
    any_processed = False
    for model_dir in model_dirs:
        ok = process_model(model_dir, args)
        if ok is None:
            overall_ok = False
            continue
        any_processed = True
        overall_ok = overall_ok and ok

    print(f"\n{'=' * 70}")
    if not any_processed:
        print("No models were fully processed.")
        sys.exit(1)
    print("ALL PASS" if overall_ok else "SOME MODELS FAILED OR WERE SKIPPED")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
