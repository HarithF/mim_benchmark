#!/usr/bin/env python3
"""
param_manifest.py -- the single source of truth for how <Name>_weights.bin is
laid out.

All pipelines read the same weights.bin, but the frontends disagree on the
argument list: the input sits at a different position, and some take
parameters or buffers the other does not:

    mim:         (conv1.weight, x, bn1.running_mean, bn1.running_var, ...)
    torch-mlir:  (x, conv1.weight, bn1.weight, bn1.bias, ...)

So nothing binds by position. gen_data.py writes <Name>_params.json naming
every slot with its canonical torch name (the `state_dict` key) and byte
offset, and every consumer binds by that name, failing loudly when a name
does not resolve:

    gen_data.py     writes the names (demangled from the mim argument names)
    gen_drivers.py  maps each driver argument to its offset
    reporter.py     loads the PyTorch reference from the blob
    export_mlir.py  names torch-mlir's %argN in <stem>.params.txt
"""

import json
import os
import re

MANIFEST_VERSION = 1

# --------------------------------------------------------------------------
# mim argument-name demangling
# --------------------------------------------------------------------------
# mim names each argument after its attribute path, with the accessor kind
# spelled out; splitting on those keywords recovers the path even when a
# component contains underscores:
#
#   l_self_modules_conv1_parameters_weight__353262   -> conv1.weight
#   l_self_modules_bn1_buffers_running_mean__353946  -> bn1.running_mean
#   l_self_modules_features_modules_3_modules_0_parameters_weight__387142
#                                                    -> features.3.0.weight

_ACCESSOR_RE = re.compile(r"(?:^|_)(modules|parameters|buffers)_")

# Path components that are an artifact of how an attribute was reached.
_NOISE_COMPONENTS = ("dict",)


def demangle_mim_name(raw):
    """mim `.ll` / `.mlir` argument name -> canonical torch state_dict name, or
    None if the name has no attribute path (the input `l_x_`, anonymous `%_N`)."""
    if not raw:
        return None
    s = raw.lstrip("%")
    # The .ll appends '_<digits>', the .mlir does not; strip it and trailing
    # underscores (but keep a trailing digit that is part of the name).
    s = re.sub(r"_\d+$", "", s)
    s = s.rstrip("_")
    if not s.startswith("l_self"):
        return None
    s = s[len("l_self") :].lstrip("_")
    if not s:
        return None

    matches = list(_ACCESSOR_RE.finditer(s))
    if not matches:
        # A plain tensor attribute of self: `l_self_hidden` -> `hidden`.
        return s or None

    comps = []
    if matches[0].start() > 0:
        comps.append(s[: matches[0].start()])
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        piece = s[m.end() : end]
        if piece:
            comps.append(piece)

    while comps and comps[0] in _NOISE_COMPONENTS:
        comps.pop(0)
    return ".".join(comps) if comps else None


def is_mim_forward_arg(raw):
    """True if a mim argument is a forward() parameter (`l_x_`) rather than
    module state (`l_self_...`): this is how the model input is identified."""
    if not raw:
        return False
    s = raw.lstrip("%")
    return s.startswith("l_") and not s.startswith("l_self")


# --------------------------------------------------------------------------
# Roles and initialization
# --------------------------------------------------------------------------
# Each slot is initialized according to what it is, so activations stay O(1)
# through deep stacks and every op stays observable. running_var must be
# positive, or eval-mode BatchNorm takes sqrt() of a negative number.
#
#   conv/linear weight  Kaiming-normal, std = sqrt(2 / fan_in)
#   norm scale (gamma)  U(0.8, 1.2)
#   bias / beta         N(0, 0.1)
#   running_mean        N(0, 0.5)
#   running_var         U(0.5, 1.5)

ROLE_WEIGHT = "weight"  # conv / linear weight, rank >= 2
ROLE_NORM_SCALE = "norm_scale"  # BatchNorm/LayerNorm gamma, rank 1
ROLE_BIAS = "bias"  # conv / linear bias, norm beta
ROLE_RUNNING_MEAN = "running_mean"
ROLE_RUNNING_VAR = "running_var"
ROLE_OTHER = "other"  # anything we cannot name; classified by rank alone

# Integer buffers no eval-mode graph reads; never an argument or a blob slot.
NON_TENSOR_BUFFER_SUFFIXES = ("num_batches_tracked",)


def classify_role(canonical_name, dims):
    """Role of one blob slot, from its canonical name and shape."""
    rank = len(dims)
    leaf = canonical_name.rsplit(".", 1)[-1] if canonical_name else None

    if leaf == "running_mean":
        return ROLE_RUNNING_MEAN
    if leaf == "running_var":
        return ROLE_RUNNING_VAR
    if leaf == "weight":
        return ROLE_WEIGHT if rank >= 2 else ROLE_NORM_SCALE
    if leaf == "bias":
        return ROLE_BIAS
    # Unrecognised: init_array decides by rank.
    return ROLE_OTHER


def init_array(role, dims, base, rng, np):
    """Draw one slot's values. `np` is passed in so this module does not need
    numpy itself."""
    dtype = {
        "float": np.float32,
        "double": np.float64,
        "half": np.float16,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
    }[base]
    n = 1
    for d in dims:
        n *= d

    if not np.issubdtype(dtype, np.floating):
        info = np.iinfo(dtype)
        lo, hi = max(info.min, -128), min(info.max, 127)
        return rng.integers(low=lo, high=hi + 1, size=n, dtype=dtype), "integers"

    if role == ROLE_RUNNING_VAR:
        return rng.uniform(0.5, 1.5, size=n).astype(dtype), "uniform(0.5,1.5)"
    if role == ROLE_RUNNING_MEAN:
        return (rng.standard_normal(n) * 0.5).astype(dtype), "normal(0,0.5)"
    if role == ROLE_NORM_SCALE:
        return rng.uniform(0.8, 1.2, size=n).astype(dtype), "uniform(0.8,1.2)"
    if role == ROLE_BIAS:
        return (rng.standard_normal(n) * 0.1).astype(dtype), "normal(0,0.1)"

    # ROLE_WEIGHT, and ROLE_OTHER classified by rank.
    if role == ROLE_WEIGHT or (role == ROLE_OTHER and len(dims) >= 2):
        fan_in = 1
        for d in dims[1:]:
            fan_in *= d
        std = (2.0 / fan_in) ** 0.5 if fan_in > 0 else 1.0
        return (rng.standard_normal(n) * std).astype(dtype), (
            f"kaiming_normal(fan_in={fan_in},std={std:.4g})"
        )
    return (rng.standard_normal(n) * 0.1).astype(dtype), "normal(0,0.1)"


# --------------------------------------------------------------------------
# Manifest I/O
# --------------------------------------------------------------------------


def manifest_path(model_root, model_name):
    return os.path.join(model_root, f"{model_name}_params.json")


def write_manifest(path, manifest):
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")


def load_manifest(path):
    """Read a manifest, or return None if it is absent."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        m = json.load(f)
    if m.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"{path}: manifest version {m.get('version')} != expected "
            f"{MANIFEST_VERSION} -- re-run gen_data.py --force"
        )
    return m


def find_manifest(model_root, model_name):
    return load_manifest(manifest_path(model_root, model_name))


def entries_by_name(manifest):
    """{canonical name: entry} for the named slots (unnamed ones have a
    positional `arg<N>` key and are left out)."""
    return {
        e["name"]: e
        for e in manifest["entries"]
        if e.get("named", True)
    }


# --------------------------------------------------------------------------
# export_mlir.py's <stem>.params.txt (copied in as <Name>_mlir.params.txt)
# --------------------------------------------------------------------------
# Names torch-mlir's %argN arguments:
#
#   arg0: <model input 0>  shape=(10, 3, 224, 224)
#   arg1: conv1.weight  shape=(64, 3, 3, 3)  kind=param

_PARAMS_TXT_RE = re.compile(
    r"^arg(?P<idx>\d+):\s+(?P<name>.*?)\s+shape=\((?P<shape>[^)]*)\)"
    r"(?:\s+kind=(?P<kind>\w+))?\s*$"
)


def parse_params_txt(path):
    """Parse a .params.txt: one entry per argument position, None for a model
    input, else {'name', 'dims', 'kind'}."""
    args = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("func_name:"):
                continue
            m = _PARAMS_TXT_RE.match(line)
            if not m:
                continue
            idx = int(m.group("idx"))
            name = m.group("name").strip()
            shape = m.group("shape").strip()
            dims = [int(t.strip()) for t in shape.split(",") if t.strip()]
            if name.startswith("<model input"):
                args[idx] = None
            else:
                args[idx] = {
                    "name": name,
                    "dims": dims,
                    "kind": m.group("kind") or "param",
                }
    if not args:
        raise ValueError(f"{path}: no `argN:` lines found")
    n = max(args) + 1
    if sorted(args) != list(range(n)):
        raise ValueError(f"{path}: argument indices are not contiguous from 0")
    return [args[i] for i in range(n)]


def find_params_txt(mlir_path):
    """The .params.txt for a .mlir: <stem>.params.txt, else the only
    *.params.txt in its directory."""
    dirpath = os.path.dirname(mlir_path)
    stem = os.path.splitext(os.path.basename(mlir_path))[0]
    direct = os.path.join(dirpath, f"{stem}.params.txt")
    if os.path.exists(direct):
        return direct
    try:
        cands = sorted(
            fn for fn in os.listdir(dirpath) if fn.endswith(".params.txt")
        )
    except OSError:
        return None
    if len(cands) == 1:
        return os.path.join(dirpath, cands[0])
    return None
