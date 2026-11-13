#!/usr/bin/env python3
"""
gen_data.py -- write each model's input and weights, from its mim .ll signature.

For every <model>/mim_llvm/<Name>_mim.ll, writes into <model>/:

    <Name>_input.bin    the model input (standard normal, float32)
    <Name>_weights.bin  every other argument, in signature order
    <Name>_params.json  the manifest: each slot's torch name, shape, type and
                        byte offset (see param_manifest.py)

The input is the argument mim named after a forward() parameter. Weights are
initialized by role (param_manifest.init_array) with a seed per model, so the
data is reproducible. Parameters only torch-mlir takes get extra slots at the
end.

Usage:
    python gen_data.py /path/to/models_root [--seed N] [--dry-run] [--force]
"""

import argparse
import os
import re
import sys
import zlib

import numpy as np

import param_manifest

# ---------------------------------------------------------------------------
# LLVM type parsing
# ---------------------------------------------------------------------------

# Maps LLVM base type names -> (numpy dtype, size in bytes)
TYPE_SIZES = {
    "float": (np.float32, 4),
    "double": (np.float64, 8),
    "half": (np.float16, 2),
    "i8": (np.int8, 1),
    "i16": (np.int16, 2),
    "i32": (np.int32, 4),
    "i64": (np.int64, 8),
}

# Matches: define <ret_type> @<Name>(<args>) {
DEFINE_RE = re.compile(
    r"define\s+.*?@(?P<name>[A-Za-z_][\w.]*)\s*\((?P<args>.*?)\)\s*\{",
    re.DOTALL,
)

# Type qualifiers / parameter attributes that can decorate a type but carry no
# shape information, e.g. '[128 x [8192 x float]] addrspace(0)* nonnull'.
QUALIFIER_RE = re.compile(
    r"\s*\b(?:addrspace|dereferenceable(?:_or_null)?|align)\s*\(?\s*\d+\s*\)?"
)
BARE_ATTRS = ("nonnull", "noalias", "nocapture", "readonly", "writeonly", "readnone")


def clean_type(type_str):
    """LLVM type without pointer stars, qualifiers and attributes:
    '[128 x [8192 x float]] addrspace(0)*' -> '[128 x [8192 x float]]'."""
    s = type_str.strip()
    prev = None
    while prev != s:
        prev = s
        s = QUALIFIER_RE.sub("", s).strip()
        s = s.rstrip("*").strip()
        for attr in BARE_ATTRS:
            if s.endswith(attr):
                s = s[: -len(attr)].strip()
    return s


def split_top_level(s, sep=","):
    """Split a string on `sep`, but only at bracket-depth 0."""
    parts = []
    depth = 0
    cur = []
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
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
    """'[128 x [16384 x float]]' -> ([128, 16384], 'float'); 'float' -> ([], 'float')."""
    s = type_str.strip()
    if s.startswith("["):
        if not s.endswith("]"):
            raise ValueError(f"Malformed array type: {type_str!r}")
        inner = s[1:-1].strip()
        depth = 0
        split_at = None
        i = 0
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
    else:
        return [], s


def parse_param(param_str):
    """'[16384 x [16384 x float]]* %_96446' -> (dims, base, ssa_name)."""
    s = param_str.strip()
    m = re.match(r"^(.*?)\s+%(\S+)$", s)
    type_part = m.group(1).strip() if m else s
    ssa_name = m.group(2) if m else None
    type_part = clean_type(type_part)
    dims, base = parse_array_type(type_part)
    return dims, base, ssa_name


def parse_signature(ll_text, filename_hint=None):
    m = DEFINE_RE.search(ll_text)
    if not m:
        raise ValueError(
            f"Could not find a `define ... @Name(...)` line"
            + (f" in {filename_hint}" if filename_hint else "")
        )
    name = m.group("name")
    args_str = m.group("args")

    ret_m = re.search(
        r"define\s+(.*?)\s+@" + re.escape(name) + r"\s*\(", ll_text, re.DOTALL
    )
    ret_type_str = clean_type(ret_m.group(1))
    ret_dims, ret_base = parse_array_type(ret_type_str)

    params = []
    for p in split_top_level(args_str):
        dims, base, ssa_name = parse_param(p)
        params.append(
            {
                "dims": dims,
                "base": base,
                "raw": p,
                "ssa_name": ssa_name,
                "canonical": param_manifest.demangle_mim_name(ssa_name),
            }
        )

    return {
        "name": name,
        "return_dims": ret_dims,
        "return_base": ret_base,
        "params": params,
    }


# ---------------------------------------------------------------------------
# Input/weight classification
# ---------------------------------------------------------------------------


def total_elems(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def drop_weight_bias_pairs(params, candidates):
    """Drop candidates that are a Linear/Conv weight: a multi-dim parameter
    directly followed by a 1-D bias of its leading size."""
    kept = []
    for i in candidates:
        nxt = params[i + 1] if i + 1 < len(params) else None
        is_weight_of_pair = (
            len(params[i]["dims"]) > 1
            and nxt is not None
            and len(nxt["dims"]) == 1
            and nxt["dims"][0] == params[i]["dims"][0]
        )
        if not is_weight_of_pair:
            kept.append(i)
    return kept or candidates


def identify_input_index(sig):
    """Index of the model input: the argument mim named after a forward()
    parameter (`l_x_`); if there is none, the parameter whose leading dim is
    the output's batch size."""
    params = sig["params"]

    by_name = [
        i
        for i, p in enumerate(params)
        if param_manifest.is_mim_forward_arg(p.get("ssa_name"))
    ]
    if by_name:
        return max(by_name, key=lambda i: total_elems(params[i]["dims"]))

    batch = sig["return_dims"][0] if sig["return_dims"] else None

    candidates = [
        i
        for i, p in enumerate(params)
        if p["dims"]
        and batch is not None
        and p["dims"][0] == batch
        and len(p["dims"]) > 1
    ]
    if not candidates:
        # Fall back: any param whose leading dim matches batch, at all.
        candidates = [
            i
            for i, p in enumerate(params)
            if p["dims"] and batch is not None and p["dims"][0] == batch
        ]
    if not candidates:
        # Last resort: assume the largest tensor is the input.
        candidates = list(range(len(params)))

    candidates = drop_weight_bias_pairs(params, candidates)

    return max(candidates, key=lambda i: total_elems(params[i]["dims"]))


# ---------------------------------------------------------------------------
# Random data generation
# ---------------------------------------------------------------------------


def dtype_for(base):
    if base not in TYPE_SIZES:
        raise ValueError(f"Unsupported/unrecognized base type: {base!r}")
    return TYPE_SIZES[base][0]


def gen_input_array(dims, base, rng):
    """The model input: standard normal (unit variance, as init_array assumes)."""
    dtype = dtype_for(base)
    n = total_elems(dims)
    if np.issubdtype(dtype, np.floating):
        arr = rng.standard_normal(n).astype(dtype)
    else:
        info = np.iinfo(dtype)
        lo, hi = max(info.min, -128), min(info.max, 127)
        arr = rng.integers(low=lo, high=hi + 1, size=n, dtype=dtype)
    return arr


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def find_mim_llvm_ll_files(root):
    """Yield (ll_path, model_root_dir) for every .ll file found inside a
    directory named 'mim_llvm' anywhere under root."""
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "mim_llvm":
            model_root = os.path.dirname(dirpath)
            for fn in sorted(filenames):
                if fn.endswith(".ll"):
                    yield os.path.join(dirpath, fn), model_root


def find_sibling_params_txt(model_root):
    """Yield (variant_dir_name, path) for every `.params.txt` that
    export_mlir.py left in one of this model's variant directories."""
    for variant in sorted(os.listdir(model_root)):
        vdir = os.path.join(model_root, variant)
        if not os.path.isdir(vdir):
            continue
        for fn in sorted(os.listdir(vdir)):
            if fn.endswith(".params.txt"):
                yield variant, os.path.join(vdir, fn)


def append_foreign_params(manifest, model_root):
    """Add slots for parameters torch-mlir takes but mim does not (torch.export
    keeps unused parameters, e.g. 50's c_proj). They go after every mim slot,
    so the mim offsets do not change. Returns the added (name, variant)
    pairs."""
    known = {e["name"] for e in manifest["entries"] if e["named"]}
    # Foreign (torch.export-style) params are nn.Parameters, always float32.
    float_size = TYPE_SIZES["float"][1]
    byte_off = manifest["total_bytes"]
    added = []
    for variant, ptxt in find_sibling_params_txt(model_root):
        try:
            args = param_manifest.parse_params_txt(ptxt)
        except Exception as e:
            print(f"    NOTE: could not read {os.path.basename(ptxt)}: {e}")
            continue
        for a in args:
            if a is None or a["name"] in known:
                continue
            known.add(a["name"])
            count = total_elems(a["dims"])
            manifest["entries"].append(
                {
                    "name": a["name"],
                    "named": True,
                    # Not in the mim signature; bound by name only.
                    "mim_arg": None,
                    "mim_index": None,
                    "dims": a["dims"],
                    "base": "float",
                    "elem_size": float_size,
                    "count": count,
                    "offset": byte_off // 4,
                    "byte_offset": byte_off,
                    "role": param_manifest.classify_role(a["name"], a["dims"]),
                    "from_variant": variant,
                }
            )
            byte_off += count * float_size
            added.append((a["name"], variant))
    manifest["total_floats"] = sum(e["count"] for e in manifest["entries"])
    manifest["total_bytes"] = byte_off
    return added


_INT_DTYPES = {"i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64}


def model_integer_buffers(model_root, entries):
    """Integer slots (e.g. 30's relative_position_index lookup tables) get the
    constructed model's buffer values: random indices run out of range.
    Returns {slot name: array}; empty for an all-float model."""
    wanted = [e for e in entries if e["base"] in _INT_DTYPES]
    if not wanted:
        return {}
    import importlib.util

    import torch

    ref_dir = os.path.join(model_root, "torch_ref")
    pys = sorted(f for f in os.listdir(ref_dir) if f.endswith(".py")) if os.path.isdir(ref_dir) else []
    if len(pys) != 1:
        raise ValueError(
            f"{len(wanted)} integer slot(s) need the model's buffers, but "
            f"{ref_dir} does not hold exactly one .py ({pys})"
        )
    spec = importlib.util.spec_from_file_location("gen_data_ref", os.path.join(ref_dir, pys[0]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    torch.manual_seed(0)
    with torch.no_grad():
        model = mod.Model(*init).eval()
    state = dict(model.state_dict())
    out = {}
    for e in wanted:
        key = e.get("ref_name", e["name"])
        t = state.get(key)
        if t is None:  # plain tensor attribute, not a registered buffer
            obj = model
            for part in key.split("."):
                obj = getattr(obj, part, None)
            t = obj if isinstance(obj, torch.Tensor) else None
        if t is None:
            raise ValueError(f"integer slot {e['name']!r} has no buffer of that name in {pys[0]}")
        if t.numel() != e["count"] or t.is_floating_point():
            raise ValueError(
                f"integer slot {e['name']!r}: model buffer is {t.dtype} x {t.numel()}, "
                f"slot is {e['base']} x {e['count']}"
            )
        out[e["name"]] = t.detach().cpu().numpy().astype(_INT_DTYPES[e["base"]]).reshape(-1)
    return out


_CLOSURE_RE = re.compile(r"^\w+_closure_\d+_cell_contents\.(?P<rest>.+)$")


def alias_closure_slots(manifest):
    """44 reaches `mlp` through a lambda, so mim names its parameters
    `mlpf_closure_0_cell_contents.*` and torch-mlir `mlp.*`. The torch-only
    slots become copies of the closure slots, which get a `ref_name` alias for
    the PyTorch reference. Returns {torch-only slot: closure slot}."""
    foreign = [e for e in manifest["entries"] if e.get("from_variant")]
    copies = {}
    for e in manifest["entries"]:
        m = _CLOSURE_RE.match(e["name"]) if e.get("named") else None
        if not m:
            continue
        cands = [
            f for f in foreign
            if f["name"].split(".", 1)[-1] == m.group("rest") and f["count"] == e["count"]
        ]
        if len(cands) == 1:
            e["ref_name"] = cands[0]["name"]
            copies[cands[0]["name"]] = e["name"]
    return copies


def build_manifest(sig, input_idx, name, seed):
    """The blob layout: one entry per weight slot in write order, with its torch
    name, element type and byte offset. A blob may mix element types (30 has
    i64 buffers); `offset` (byte_offset // 4) only applies to all-float blobs."""
    params = sig["params"]

    entries = []
    byte_off = 0
    for i, p in enumerate(params):
        if i == input_idx:
            continue
        canonical = p["canonical"]
        named = canonical is not None
        count = total_elems(p["dims"])
        elem_size = TYPE_SIZES[p["base"]][1]
        role = param_manifest.classify_role(canonical, p["dims"])
        entries.append(
            {
                "name": canonical if named else f"arg{i}",
                "named": named,
                "mim_arg": p["ssa_name"],
                "mim_index": i,
                "dims": p["dims"],
                "base": p["base"],
                "elem_size": elem_size,
                "count": count,
                "offset": byte_off // 4,
                "byte_offset": byte_off,
                "role": role,
            }
        )
        byte_off += count * elem_size

    # Summary fields: `total_floats` is the element count; the file size is
    # `total_bytes`.
    weight_bases = sorted({e["base"] for e in entries})
    weight_base = "float" if "float" in weight_bases else (
        weight_bases[0] if weight_bases else sig["return_base"]
    )
    weight_elem_size = TYPE_SIZES[weight_base][1]
    total_elements = sum(e["count"] for e in entries)

    input_p = params[input_idx]
    return {
        "version": param_manifest.MANIFEST_VERSION,
        "model": name,
        "seed": seed,
        "source_ll": None,  # filled in by the caller, which knows the path
        "weights_file": f"{name}_weights.bin",
        "input_file": f"{name}_input.bin",
        "input": {
            "mim_arg": input_p["ssa_name"],
            "mim_index": input_idx,
            "dims": input_p["dims"],
            "base": input_p["base"],
            "count": total_elems(input_p["dims"]),
        },
        "output": {"dims": sig["return_dims"], "base": sig["return_base"]},
        "weight_base": weight_base,
        "weight_elem_size": weight_elem_size,
        "total_floats": total_elements,
        "total_bytes": byte_off,
        "entries": entries,
    }


def process_file(ll_path, model_root, rng, seed=None, dry_run=False, force=False):
    with open(ll_path, "r") as f:
        text = f.read()

    sig = parse_signature(text, filename_hint=ll_path)
    name = sig["name"]
    input_idx = identify_input_index(sig)

    weights_path = os.path.join(model_root, f"{name}_weights.bin")
    input_path = os.path.join(model_root, f"{name}_input.bin")
    mpath = param_manifest.manifest_path(model_root, name)

    manifest = build_manifest(sig, input_idx, name, seed)
    manifest["source_ll"] = os.path.relpath(ll_path, model_root)
    foreign = append_foreign_params(manifest, model_root)

    print(f"[{name}] {ll_path}")
    print(f"    return shape : {sig['return_dims']} x {sig['return_base']}")
    print(
        f"    input        : {sig['params'][input_idx]['dims']} "
        f"x {sig['params'][input_idx]['base']}  (arg {input_idx})"
    )
    unnamed = [e for e in manifest["entries"] if not e["named"]]
    for e in manifest["entries"]:
        idx = "-" if e["mim_index"] is None else str(e["mim_index"])
        tag = (
            f"  <- {e['from_variant']} only (unused by mim)"
            if e.get("from_variant")
            else ""
        )
        print(
            f"    param {idx:>3} [{e['role']:>12}] : "
            f"{e['dims']} @ +{e['byte_offset']}  {e['name']}{tag}"
        )
    if foreign:
        print(
            f"    NOTE: {len(foreign)} slot(s) added for another frontend's "
            f"parameters that the mim graph never reads "
            f"({', '.join(n for n, _ in foreign[:4])}"
            f"{', ...' if len(foreign) > 4 else ''})"
        )
    if unnamed:
        # Not fatal: unnamed slots are written, but bound by position only.
        print(
            f"    NOTE: {len(unnamed)} of {len(manifest['entries'])} slots have "
            f"no recoverable attribute path "
            f"({', '.join(e['mim_arg'] or '<anonymous>' for e in unnamed[:4])}"
            f"{', ...' if len(unnamed) > 4 else ''})"
        )

    if not force and not dry_run:
        if (
            os.path.exists(weights_path)
            or os.path.exists(input_path)
            or os.path.exists(mpath)
        ):
            print(f"    -> SKIP (already exists; use --force to overwrite)")
            return

    if dry_run:
        print(f"    -> (dry-run) would write {weights_path}")
        print(f"    -> (dry-run) would write {input_path}")
        print(f"    -> (dry-run) would write {mpath}")
        return

    # Input tensor
    input_param = sig["params"][input_idx]
    input_arr = gen_input_array(input_param["dims"], input_param["base"], rng)
    input_arr.tofile(input_path)

    # Weights: every non-input param, in manifest order (== signature order).
    # Every slot is drawn, so float weights stay the same; integer and aliased
    # slots then get their values replaced.
    int_values = model_integer_buffers(model_root, manifest["entries"])
    aliases = alias_closure_slots(manifest)
    written = {}
    with open(weights_path, "wb") as wf:
        for e in manifest["entries"]:
            arr, how = param_manifest.init_array(
                e["role"], e["dims"], e["base"], rng, np
            )
            if e["name"] in int_values:
                arr, how = int_values[e["name"]], "model buffer"
            elif e["name"] in aliases and aliases[e["name"]] in written:
                arr, how = written[aliases[e["name"]]], f"copy of {aliases[e['name']]}"
            e["init"] = how
            written[e["name"]] = arr
            wf.write(arr.tobytes())

    param_manifest.write_manifest(mpath, manifest)

    w_size = os.path.getsize(weights_path)
    i_size = os.path.getsize(input_path)
    expected = manifest["total_bytes"]
    if w_size != expected:
        raise RuntimeError(
            f"{weights_path}: wrote {w_size} bytes but the manifest describes "
            f"{expected} -- offsets would be wrong"
        )
    print(f"    -> wrote {weights_path} ({w_size:,} bytes)")
    print(f"    -> wrote {input_path} ({i_size:,} bytes)")
    print(f"    -> wrote {mpath}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", help="Root directory to scan for model variant folders")
    ap.add_argument(
        "--seed", type=int, default=None, help="Random seed (for reproducibility)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print parsed shapes, don't write files",
    )
    ap.add_argument(
        "--force", action="store_true", help="Overwrite existing .bin files"
    )
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"Not a directory: {args.root}", file=sys.stderr)
        sys.exit(1)

    found_any = False
    for ll_path, model_root in find_mim_llvm_ll_files(args.root):
        found_any = True
        try:
            model_name = os.path.splitext(os.path.basename(ll_path))[0]
            rng = np.random.default_rng(
                [
                    args.seed if args.seed is not None else 0,
                    zlib.crc32(model_name.encode()),
                ]
            )
            process_file(
                ll_path,
                model_root,
                rng,
                seed=args.seed,
                dry_run=args.dry_run,
                force=args.force,
            )
        except Exception as e:
            print(f"[ERROR] {ll_path}: {e}", file=sys.stderr)

    if not found_any:
        print(f"No mim_llvm/*.ll files found under {args.root}", file=sys.stderr)


if __name__ == "__main__":
    main()
