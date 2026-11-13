#!/usr/bin/env python3
"""
gen_drivers.py -- generate a C driver for every <model>/<variant>/*.ll.

The driver loads <Name>_weights.bin and <Name>_input.bin, calls the model
function, times it, and writes the output. Every weight argument is read from
its byte offset in weights.bin, looked up by name in <Name>_params.json.

Two calling conventions:
  * mim style: one pointer per argument; shapes come from the .ll signature.
  * mlir style (convert-func-to-llvm): each memref argument is expanded into
    (alloc, aligned, offset, sizes[R], strides[R]); shapes and argument names
    come from the sibling .mlir (and, for torch-mlir, its .params.txt).

Usage:
    python gen_drivers.py /path/to/models_root [--dry-run] [--force]
"""

import argparse
import os
import re
import sys

import param_manifest

# ---------------------------------------------------------------------------
# LLVM type parsing
# ---------------------------------------------------------------------------

DEFINE_RE = re.compile(
    r"define\s+(?P<ret>.*?)\s+@(?P<name>[A-Za-z_][\w.]*)\s*\((?P<args>.*?)\)\s*\{",
    re.DOTALL,
)

ADDRSPACE_RE = re.compile(r"\baddrspace\(\d+\)\s*")


def strip_addrspace(s):
    return ADDRSPACE_RE.sub("", s)


def split_top_level(s, sep=","):
    """Split a string on `sep`, but only at bracket/brace depth 0."""
    parts = []
    depth = 0
    cur = []
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
    """Parse a (possibly nested) LLVM array type, e.g.
    '[128 x [16384 x float]]' -> ([128, 16384], 'float').
    A plain scalar type like 'float' -> ([], 'float')."""
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


def total_elems(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def strip_name_suffix(param_str):
    """'[16384 x [16384 x float]]* %_96446' -> '[16384 x [16384 x float]]*'"""
    s = param_str.strip()
    m = re.match(r"^(.*?)\s+%\S+$", s)
    return m.group(1).strip() if m else s


def param_ssa_name(param_str):
    """'[64 x float]* %l_self_modules_bn1_buffers_running_mean__353946'
    -> 'l_self_modules_bn1_buffers_running_mean__353946', or None."""
    m = re.match(r"^.*?\s+%(\S+)$", param_str.strip())
    return m.group(1) if m else None


def parse_pointee_array(type_with_ptr):
    """Given a type string like '[16384 x [16384 x float]] addrspace(0)*'
    (trailing '%name' already stripped), strip addrspace/pointer and parse
    the pointee's array shape."""
    t = strip_addrspace(type_with_ptr).strip()
    t = t.rstrip("*").strip()
    return parse_array_type(t)


# ---------------------------------------------------------------------------
# MLIR signature parsing: shapes and argument names for an mlir-style .ll come
# from the .mlir next to it, which has the same argument order.
# ---------------------------------------------------------------------------

FUNC_FUNC_RE = re.compile(
    r"func\.func\s+(?:public\s+)?@(?P<name>[A-Za-z_][\w.$]*)\s*"
    r"\((?P<args>.*?)\)\s*(?:->\s*(?P<ret>[^\{]*?))?\s*\{",
    re.DOTALL,
)


def find_balanced(s, open_idx, open_ch="<", close_ch=">"):
    """s[open_idx] == open_ch; return index of the matching close_ch."""
    depth = 0
    i = open_idx
    while i < len(s):
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(
        f"Unbalanced '{open_ch}{close_ch}' starting at {open_idx} in {s!r}"
    )


def split_top_level_generic(s, sep=","):
    """Split on `sep` at depth 0, tracking all of ()[]{}<> as nesting."""
    opens, closes = "([{<", ")]}>"
    parts = []
    depth = 0
    cur = []
    for ch in s:
        if ch in opens:
            depth += 1
        elif ch in closes:
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def parse_mlir_shaped_type(type_str):
    """'tensor<96x3x11x11xf32>' or 'memref<96x3x11x11xf32, strided<...>>'
    -> ([96, 3, 11, 11], 'f32'). Raises if any dimension is dynamic ('?')."""
    s = type_str.strip()
    m = re.match(r"^(tensor|memref)\s*<", s)
    if not m:
        raise ValueError(f"Not a tensor/memref type: {s!r}")
    open_idx = m.end() - 1
    close_idx = find_balanced(s, open_idx, "<", ">")
    inner = s[open_idx + 1 : close_idx]
    # inner may be 'NxNx...xTYPE' optionally followed by ', strided<...>' etc.
    shape_and_type = split_top_level_generic(inner, sep=",")[0]
    tokens = shape_and_type.split("x")
    base = tokens[-1].strip()
    dims = []
    for t in tokens[:-1]:
        t = t.strip()
        if t == "?" or not t:
            raise ValueError(f"Dynamic/unknown dimension in type: {type_str!r}")
        dims.append(int(t))
    return dims, base


MLIR_BASE_TO_LLVM_BASE = {
    "f32": "float",
    "f64": "double",
    "f16": "half",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
}


def parse_mlir_func_signature(text, expected_name=None):
    """Return (name, params, results) for the first func.func in `text` (or the
    one named expected_name). params: (dims, base, arg_name) per argument;
    results: (dims, base) per result (`-> T` and `-> (T, U)` both accepted)."""
    for m in FUNC_FUNC_RE.finditer(text):
        name = m.group("name")
        if expected_name is not None and name != expected_name:
            continue
        args_str = m.group("args") or ""
        ret_str = (m.group("ret") or "").strip()

        params = []
        for arg in split_top_level_generic(args_str, sep=","):
            # 'ARGNAME: TYPE' -- split on the first top-level ':'
            parts = arg.split(":", 1)
            if len(parts) != 2:
                raise ValueError(f"Could not parse mlir arg: {arg!r}")
            arg_name = parts[0].strip().lstrip("%")
            type_str = parts[1].strip()
            dims, base = parse_mlir_shaped_type(type_str)
            params.append(
                (dims, MLIR_BASE_TO_LLVM_BASE.get(base, base), arg_name)
            )

        # Strip the parentheses around a multi-result list.
        if ret_str.startswith("(") and find_balanced(
            ret_str, 0, "(", ")"
        ) == len(ret_str) - 1:
            ret_str = ret_str[1:-1].strip()

        results = []
        for ret_part in split_top_level_generic(ret_str, sep=","):
            dims, base = parse_mlir_shaped_type(ret_part)
            results.append((dims, MLIR_BASE_TO_LLVM_BASE.get(base, base)))

        return name, params, results

    raise ValueError(
        f"No matching func.func found"
        + (f" for @{expected_name}" if expected_name else "")
    )


def find_sibling_mlir_shape_source(ll_path, expected_name):
    """Signature of func.func @expected_name from a .mlir next to `ll_path`
    (non-`_lowered` files first), or None."""
    dirpath = os.path.dirname(ll_path)
    try:
        candidates = sorted(fn for fn in os.listdir(dirpath) if fn.endswith(".mlir"))
    except OSError:
        return None
    # Try non-"_lowered" files first (cleaner tensor-dialect signatures),
    # then fall back to "_lowered" ones.
    candidates.sort(key=lambda fn: ("_lowered" in fn, fn))

    decl_re = re.compile(
        r"func\.func\s+(?:public\s+)?@" + re.escape(expected_name) + r"\s*\("
    )
    for fn in candidates:
        path = os.path.join(dirpath, fn)
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            continue
        try:
            name, raw_params, results = parse_mlir_func_signature(
                text, expected_name=expected_name
            )
        except Exception as e:
            # Warn if the file declares the function but can't be parsed.
            if decl_re.search(text):
                print(
                    f"    WARNING: {path} declares @{expected_name} but its "
                    f"signature could not be parsed: {e}"
                )
            continue
        params = [
            Param(dims, base, arg_name=arg_name)
            for dims, base, arg_name in raw_params
        ]
        sig = Signature(name, params, results, style="mlir_source")
        sig.source_file = path
        sig.name_source, sig.unsupported = attach_canonical_names(params, path)
        return sig
    return None


def attach_canonical_names(params, mlir_path):
    """Fill in `param.canonical`: from the mim .mlir's argument names, or from
    torch-mlir's .params.txt. Returns (name_source, unsupported_reason); a
    non-None reason means the model cannot be bound and must be refused."""
    for p in params:
        if param_manifest.is_mim_forward_arg(p.arg_name):
            p.role = "input"
    demangled = sum(1 for p in params if p.canonical)
    if demangled:
        return f"argument names in {os.path.basename(mlir_path)}", None

    ptxt = param_manifest.find_params_txt(mlir_path)
    if ptxt is None:
        return None, None
    try:
        args = param_manifest.parse_params_txt(ptxt)
    except Exception as e:
        print(f"    WARNING: could not parse {ptxt}: {e}")
        return None, None
    n_inputs = sum(1 for a in args if a is None)
    if n_inputs > 1:
        # Drivers take exactly one input (<Name>_input.bin).
        return None, (
            f"{os.path.basename(ptxt)} declares {n_inputs} model inputs, but "
            f"the harness supports exactly one (<Name>_input.bin). This model "
            f"needs multi-input support in gen_data.py and the driver "
            f"templates"
        )
    if len(args) != len(params):
        print(
            f"    WARNING: {os.path.basename(ptxt)} describes {len(args)} "
            f"arguments but {os.path.basename(mlir_path)} declares "
            f"{len(params)} -- ignoring the manifest"
        )
        return None, None
    for p, a in zip(params, args):
        if a is None:
            # '<model input N>': read from input.bin.
            p.role = "input"
            continue
        n = 1
        for d in a["dims"]:
            n *= d
        if n != total_elems(p.dims):
            print(
                f"    WARNING: {os.path.basename(ptxt)} says arg for "
                f"'{a['name']}' holds {n} elements but the graph declares "
                f"{total_elems(p.dims)} -- ignoring the manifest"
            )
            return None, None
        p.canonical = a["name"]
    return os.path.basename(ptxt), None


# ---------------------------------------------------------------------------
# Top-level signature parsing / classification
# ---------------------------------------------------------------------------


class Param:
    __slots__ = (
        "dims",
        "base",
        "role",
        "cname",
        "arg_name",
        "canonical",
        "byte_offset",
    )

    def __init__(self, dims, base, arg_name=None):
        self.dims = dims
        self.base = base
        self.role = None  # 'input' | 'weight' | 'bias'
        self.cname = None  # e.g. 'x', 'w3', 'b3'
        # Name in the artifact, and the torch parameter name (the key into
        # <Name>_params.json).
        self.arg_name = arg_name
        self.canonical = param_manifest.demangle_mim_name(arg_name)
        # Byte offset in weights.bin (from the manifest).
        self.byte_offset = None


class Signature:
    def __init__(self, name, params, results, style, mlir_groups=None):
        self.name = name
        # Where the argument names came from (.mlir-sourced signatures).
        self.name_source = None
        # Why the model cannot be bound (e.g. several inputs), if it can't.
        self.unsupported = None
        self.params = params  # list[Param], in declaration order
        self.results = results  # list[(dims, base)], one per returned value
        self.style = style  # 'raw_flat' | 'mlir_memref' | 'unknown'
        self.mlir_groups = (
            mlir_groups  # list of (start, end, rank) for mlir_memref style
        )

    # Convenience accessors for the first (usually only) result.
    @property
    def return_dims(self):
        return self.results[0][0] if self.results else []

    @property
    def return_base(self):
        return self.results[0][1] if self.results else None


def extract_define(text, filename_hint=None):
    m = DEFINE_RE.search(text)
    if not m:
        raise ValueError(
            "Could not find a `define ... @Name(...) {` line"
            + (f" in {filename_hint}" if filename_hint else "")
        )
    return m.group("name"), m.group("ret").strip(), m.group("args")


def classify_arg_types(arg_type_strs):
    cleaned = [strip_addrspace(t).strip() for t in arg_type_strs]
    if cleaned and all(t.startswith("[") for t in cleaned):
        return "raw_flat"
    if cleaned and all(t == "ptr" or re.match(r"^i\d+$", t) for t in cleaned):
        return "mlir_memref"
    return "unknown"


def parse_memref_descriptor_rank(desc_str, filename_hint=None):
    """Rank of one convert-func-to-llvm memref descriptor, which looks like
    `{ ptr, ptr, i64, [R x i64], [R x i64] }` (sizes then strides)."""
    sizes = re.findall(r"\[(\d+)\s*x\s*i64\]", desc_str)
    if not sizes:
        return 0  # rank-0 memref: '{ ptr, ptr, i64 }'
    if len(sizes) != 2 or sizes[0] != sizes[1]:
        raise ValueError(
            f"Expected one sizes[] and one strides[] array of equal rank in "
            f"memref descriptor {desc_str!r} in {filename_hint}"
        )
    return int(sizes[0])


def parse_mlir_memref_return_ranks(ret_str, filename_hint=None):
    """Ranks of the memref descriptors in a convert-func-to-llvm return type:
    one bare descriptor, or a struct of descriptors for several results."""
    s = ret_str.strip()
    if not s.startswith("{"):
        raise ValueError(f"Could not parse mlir return type: {ret_str!r}")
    inner = s[1 : find_balanced(s, 0, "{", "}")]
    fields = split_top_level_generic(inner, sep=",")
    if fields and all(f.startswith("{") for f in fields):
        return [parse_memref_descriptor_rank(f, filename_hint) for f in fields]
    return [parse_memref_descriptor_rank(s, filename_hint)]


def parse_signature(text, filename_hint=None):
    name, ret_str, args_str = extract_define(text, filename_hint)
    raw_params = split_top_level(args_str)

    # Get each param's *type* (name stripped) for classification.
    arg_type_strs = [strip_name_suffix(p) for p in raw_params]
    style = classify_arg_types(arg_type_strs)

    results = []
    mlir_groups = None
    params = []

    if style == "raw_flat":
        for t, raw in zip(arg_type_strs, raw_params):
            dims, base = parse_pointee_array(t)
            p = Param(dims, base, arg_name=param_ssa_name(raw))
            # mim names forward() arguments `l_<name>_`: that is the input.
            if param_manifest.is_mim_forward_arg(p.arg_name):
                p.role = "input"
            params.append(p)
        results = [parse_pointee_array(ret_str)]

    elif style == "mlir_memref":
        types = arg_type_strs  # 'ptr' or 'iN', one per scalarized field
        groups = []
        i = 0
        n = len(types)
        while i < n:
            if types[i] != "ptr" or i + 1 >= n or types[i + 1] != "ptr":
                raise ValueError(
                    f"Unexpected mlir_memref layout at arg {i} in {filename_hint}"
                )
            j = i + 2
            if j >= n or types[j] != "i64":
                raise ValueError(f"Expected offset i64 at arg {j} in {filename_hint}")
            j += 1
            start = j
            while j < n and types[j] == "i64":
                j += 1
            total_i64 = j - start
            if total_i64 % 2 != 0:
                raise ValueError(
                    f"Odd number of size/stride i64 fields ({total_i64}) at "
                    f"arg {start} in {filename_hint}"
                )
            rank = total_i64 // 2
            groups.append((i, j, rank))
            params.append(
                Param(dims=[None] * rank, base="float")
            )  # dims filled in later
            i = j
        mlir_groups = groups

        # Result dims are filled in from the shape source later.
        results = [
            ([None] * rank, "float")
            for rank in parse_mlir_memref_return_ranks(ret_str, filename_hint)
        ]

    else:
        pass  # unknown -- leave params empty, caller will skip

    return Signature(name, params, results, style, mlir_groups)


# ---------------------------------------------------------------------------
# Input/weight role assignment
# ---------------------------------------------------------------------------


def drop_weight_bias_pairs(params, candidates):
    """Drop candidates that are a Linear/Conv weight: a multi-dim parameter
    directly followed by a 1-D bias of its leading size. Breaks the tie when a
    weight has the input's shape. Never empties the candidate set."""
    kept = []
    for i in candidates:
        nxt = params[i + 1] if i + 1 < len(params) else None
        is_weight_of_pair = (
            len(params[i].dims) > 1
            and nxt is not None
            and len(nxt.dims) == 1
            and nxt.dims[0] == params[i].dims[0]
        )
        if not is_weight_of_pair:
            kept.append(i)
    return kept or candidates


def assign_roles(params, return_dims):
    """Set .role and .cname on params in place. An input already identified by
    name is kept; otherwise the input is picked by shape."""
    preset = [i for i, p in enumerate(params) if p.role == "input"]
    if preset:
        input_idx = max(preset, key=lambda i: total_elems(params[i].dims))
        return _finish_roles(params, input_idx)

    batch = return_dims[0] if return_dims else None

    candidates = [
        i
        for i, p in enumerate(params)
        if p.dims and p.dims[0] == batch and len(p.dims) > 1 and None not in p.dims
    ]
    if not candidates:
        candidates = [
            i
            for i, p in enumerate(params)
            if p.dims and p.dims[0] == batch and None not in p.dims
        ]
    if not candidates:
        candidates = [i for i in range(len(params)) if None not in params[i].dims]
    if not candidates:
        candidates = list(range(len(params)))

    candidates = drop_weight_bias_pairs(params, candidates)

    input_idx = max(
        candidates,
        key=lambda i: total_elems(params[i].dims) if None not in params[i].dims else -1,
    )
    return _finish_roles(params, input_idx)


def _finish_roles(params, input_idx):
    wc = 0
    bc = 0
    for i, p in enumerate(params):
        if i == input_idx:
            p.role = "input"
            p.cname = "x"
        elif len(p.dims) > 1:
            wc += 1
            p.role = "weight"
            p.cname = f"w{wc}"
        else:
            bc += 1
            p.role = "bias"
            p.cname = f"b{bc}"
    return input_idx


# ---------------------------------------------------------------------------
# Locating a "shape source" for mlir_memref-style signatures
# ---------------------------------------------------------------------------


def find_all_ll_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.endswith(".ll"):
                yield os.path.join(dirpath, fn)


def find_model_root(ll_path):
    """The model directory a variant directory lives in: `.../08_X/mlir_llvm/
    Foo.ll` -> `.../08_X`. That is where gen_data.py puts the blob and its
    manifest."""
    return os.path.dirname(os.path.dirname(os.path.abspath(ll_path)))


def bind_to_manifest(params, manifest, model_name):
    """Give every non-input parameter its byte offset in weights.bin.

    Returns (mode, message), mode being one of:
      'by_name'         every parameter matched by name
      'by_name+index'   as above, unnamed ones placed by declaration index
      'by_name_unread'  as 'by_name'; some blob slots are not used here
      'positional'      no names: declaration order (counts/sizes checked)
      'error'           mismatch; no driver may be written"""
    weights = [p for p in params if p.role != "input"]
    entries = manifest["entries"]
    elem_size = manifest.get("weight_elem_size", 4)

    named = [p for p in weights if p.canonical]
    if named:
        by_index = {e["mim_index"]: e for e in entries}
        unnamed = [p for p in weights if not p.canonical]
        if unnamed:
            # Unnamed parameters are placed by position, only if every named
            # one sits at the index the manifest recorded for it.
            in_mim_order = all(
                p.canonical is None
                or (
                    by_index.get(i) is not None
                    and by_index[i]["name"] == p.canonical
                )
                for i, p in enumerate(params)
            )
            if not in_mim_order:
                return (
                    "error",
                    f"{len(unnamed)} of {len(weights)} parameters have no "
                    f"recoverable name ({', '.join(p.cname for p in unnamed[:6])}) "
                    f"and this artifact's argument order is not the one "
                    f"{model_name}_params.json was built from, so they cannot "
                    f"be placed by position either",
                )
        by_name = param_manifest.entries_by_name(manifest)
        unknown = [
            p.canonical for p in weights if p.canonical and p.canonical not in by_name
        ]
        if unknown:
            return (
                "error",
                f"{len(unknown)} parameter(s) are not in "
                f"{model_name}_params.json ({', '.join(unknown[:6])}) -- the "
                f"blob and this artifact describe different models; re-run "
                f"gen_data.py --force after re-importing",
            )
        resolved = []
        for i, p in enumerate(params):
            if p.role == "input":
                continue
            e = by_name[p.canonical] if p.canonical else by_index.get(i)
            if e is None:
                return (
                    "error",
                    f"parameter {p.cname} (index {i}) has no name and no "
                    f"manifest slot at that index",
                )
            if e["count"] != total_elems(p.dims):
                return (
                    "error",
                    f"'{e['name']}' holds {total_elems(p.dims)} elements "
                    f"here but {e['count']} in the manifest",
                )
            p.canonical = p.canonical or e["name"]
            p.byte_offset = e["byte_offset"]
            resolved.append(e["name"])
        # Two arguments resolving to one slot would read the same weights
        # twice and leave another slot unread.
        if len(set(resolved)) != len(resolved):
            dup = sorted({n for n in resolved if resolved.count(n) > 1})
            return ("error", f"duplicate parameter name(s): {', '.join(dup)}")
        bound = set(resolved)
        unread = [e["name"] for e in entries if e["name"] not in bound]
        if unread:
            # Unread slots are harmless with name binding (frontends can
            # differ in which parameters they take), but not with positional.
            if unnamed:
                return (
                    "error",
                    f"{len(resolved)} of {len(entries)} slots in "
                    f"{model_name}_params.json were bound -- {len(unread)} "
                    f"would go unread, and {len(unnamed)} parameter(s) here "
                    f"are placed by declaration index, which requires an "
                    f"exact slot correspondence",
                )
            return (
                "by_name_unread",
                f"matched by name against {model_name}_params.json; "
                f"{len(unread)} slot(s) in the blob unread by this frontend "
                f"({', '.join(unread[:6])})",
            )
        mode = "by_name" if not unnamed else "by_name+index"
        return (mode, f"matched by name against {model_name}_params.json")

    # No names anywhere (a struct-of-pointers .ll, or a torch-mlir graph
    # imported without its .params.txt).
    if len(weights) != len(entries):
        return (
            "error",
            f"no argument names available, and positional binding is not "
            f"possible either: {len(weights)} weight parameters here vs "
            f"{len(entries)} slots in {model_name}_params.json",
        )
    total_here = sum(total_elems(p.dims) for p in weights)
    if total_here != manifest["total_floats"]:
        return (
            "error",
            f"no argument names available, and positional binding disagrees "
            f"on size: {total_here} elements here vs "
            f"{manifest['total_floats']} in {model_name}_params.json",
        )
    offset = 0
    for p in weights:
        p.byte_offset = offset * elem_size
        offset += total_elems(p.dims)
    return (
        "positional",
        f"NO argument names available -- offsets assigned in declaration "
        f"order. Counts and total size agree with {model_name}_params.json, "
        f"but the ORDER is unverified",
    )


def validate_input_against_manifest(params, manifest, model_name):
    """Check the input parameter's size against the input gen_data.py wrote."""
    inp = [p for p in params if p.role == "input"]
    if len(inp) != 1:
        return f"expected exactly one input parameter, found {len(inp)}"
    want = manifest.get("input", {}).get("count")
    if want is None:
        return None
    got = total_elems(inp[0].dims)
    if got != want:
        return (
            f"the parameter picked as the input holds {got} elements but "
            f"{model_name}_input.bin holds {want} -- the input was "
            f"misidentified for this artifact"
        )
    return None


# ---------------------------------------------------------------------------
# C code generation: mim style
# ---------------------------------------------------------------------------


def row_major_strides(dims):
    strides = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        strides[i] = strides[i + 1] * dims[i + 1]
    return strides


def c_read_floats_helper():
    """C helpers that read each weight at its byte offset from the manifest."""
    return """static float *read_floats_at(FILE *f, long byte_offset, size_t n,
                             const char *what) {
  float *p = (float *)malloc(n * sizeof(float));
  if (!p) {
    fprintf(stderr, "out of memory reading %s (%zu floats)\\n", what, n);
    exit(1);
  }
  if (fseek(f, byte_offset, SEEK_SET) != 0) {
    fprintf(stderr, "seek to offset %ld for %s failed\\n", byte_offset, what);
    exit(1);
  }
  size_t got = fread(p, sizeof(float), n, f);
  if (got != n) {
    fprintf(stderr, "short read on %s at offset %ld: wanted %zu floats, got "
                    "%zu\\n", what, byte_offset, n, got);
    exit(1);
  }
  return p;
}

static float *read_floats(FILE *f, size_t n, const char *what) {
  float *p = (float *)malloc(n * sizeof(float));
  if (!p) {
    fprintf(stderr, "out of memory reading %s (%zu floats)\\n", what, n);
    exit(1);
  }
  size_t got = fread(p, sizeof(float), n, f);
  if (got != n) {
    fprintf(stderr, "short read on %s: wanted %zu floats, got %zu\\n", what, n,
            got);
    exit(1);
  }
  return p;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
"""


# C type + reader for each non-float element type (float: c_read_floats_helper).
_C_TYPES = {
    "double": "double",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
}

# (C element type, reader function) for a graph `base`. `float` maps to the
# historic read_floats_at.
_C_BASE = {"float": ("float", "read_floats_at")}
_C_BASE.update({b: (t, f"read_{b}_at") for b, t in _C_TYPES.items()})


def c_typed_read_helpers(bases):
    """One `read_<base>_at` C helper per non-float element type a driver needs
    (e.g. i64 buffers)."""
    lines = []
    for base in sorted(bases):
        if base == "float":
            continue
        ctype = _C_TYPES[base]
        fn = f"read_{base}_at"
        lines.append(
            f"""static {ctype} *{fn}(FILE *f, long byte_offset, size_t n,
                     const char *what) {{
  {ctype} *p = ({ctype} *)malloc(n * sizeof({ctype}));
  if (!p) {{
    fprintf(stderr, "out of memory reading %s (%zu)\\n", what, n);
    exit(1);
  }}
  if (fseek(f, byte_offset, SEEK_SET) != 0) {{
    fprintf(stderr, "seek to offset %ld for %s failed\\n", byte_offset, what);
    exit(1);
  }}
  size_t got = fread(p, sizeof({ctype}), n, f);
  if (got != n) {{
    fprintf(stderr, "short read on %s at offset %ld: wanted %zu, got %zu\\n",
                    what, byte_offset, n, got);
    exit(1);
  }}
  return p;
}}

"""
        )
    return "\n".join(lines)


def c_timing_report_helper():
    """C helpers for the run id (argv[5], default: UTC timestamp) and the JSON
    timing report (argv[6]) that reporter.py reads."""
    return """static void resolve_run_id(int argc, char **argv, char *buf, size_t buflen) {
  if (argc > 5) {
    snprintf(buf, buflen, "%s", argv[5]);
    return;
  }
  time_t now = time(NULL);
  struct tm tm_utc;
  gmtime_r(&now, &tm_utc);
  strftime(buf, buflen, "%Y%m%d_%H%M%S", &tm_utc);
}

static void print_json_string(FILE *f, const char *s) {
  fputc('"', f);
  for (const char *p = s; *p; p++) {
    if (*p == '"' || *p == '\\\\')
      fputc('\\\\', f);
    fputc(*p, f);
  }
  fputc('"', f);
}

static void write_timing_report(const char *json_path, const char *model,
                                 const char *pipeline, const char *run_id,
                                 const char *weights_path, const char *input_path,
                                 const char *output_path, int reps,
                                 double median_ns, double mean_ns, double min_ns,
                                 double max_ns) {
  FILE *jf = fopen(json_path, "w");
  if (!jf) {
    fprintf(stderr, "warning: failed to write timing report '%s': ", json_path);
    perror(NULL);
    return;
  }
  fprintf(jf, "{\\n  \\"model\\": ");
  print_json_string(jf, model);
  fprintf(jf, ",\\n  \\"pipeline\\": ");
  print_json_string(jf, pipeline);
  fprintf(jf, ",\\n  \\"run_id\\": ");
  print_json_string(jf, run_id);
  fprintf(jf, ",\\n  \\"weights_path\\": ");
  print_json_string(jf, weights_path);
  fprintf(jf, ",\\n  \\"input_path\\": ");
  print_json_string(jf, input_path);
  fprintf(jf, ",\\n  \\"output_path\\": ");
  print_json_string(jf, output_path);
  fprintf(jf, ",\\n  \\"reps\\": %d", reps);
  fprintf(jf, ",\\n  \\"median_ns\\": %.1f", median_ns);
  fprintf(jf, ",\\n  \\"mean_ns\\": %.1f", mean_ns);
  fprintf(jf, ",\\n  \\"min_ns\\": %.1f", min_ns);
  fprintf(jf, ",\\n  \\"max_ns\\": %.1f", max_ns);
  fprintf(jf, "\\n}\\n");
  fclose(jf);
  printf("wrote timing report to %s\\n", json_path);
  fflush(stdout);
}
"""


def c_fork_helpers():
    """C helpers that run each model call in a forked child: the mim code never
    frees its intermediates, and forking keeps that from accumulating across
    reps."""
    return """/* --- per-call process isolation; see c_fork_helpers() in gen_drivers.py ---
 * Each model call runs in a forked child (weights/input shared copy-on-write);
 * times come back through shared memory, and the last child writes the output.
 * DRIVER_NO_FORK=1 runs every call in-process instead.
 */
static int driver_fork_enabled(void) {
  const char *e = getenv("DRIVER_NO_FORK");
  return !(e && e[0] && e[0] != '0');
}

static double *alloc_shared_times(int reps) {
  size_t n = (size_t)reps * sizeof(double);
  void *p = mmap(NULL, n, PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    perror("mmap for shared timing buffer");
    exit(1);
  }
  return (double *)p;
}

static void await_child(pid_t pid, const char *what) {
  int st = 0;
  if (waitpid(pid, &st, 0) < 0) {
    perror("waitpid");
    exit(1);
  }
  if (WIFSIGNALED(st)) {
    int s = WTERMSIG(st);
    fprintf(stderr, "%s: child killed by signal %d\\n", what, s);
    if (s == SIGKILL)
      fprintf(stderr,
              "  SIGKILL almost always means the OOM killer. This model "
              "leaks several GiB\\n  per call inside the generated code; a "
              "single call already exceeds free RAM.\\n  Forking bounds the "
              "leak to one call, so it cannot help beyond that.\\n");
    exit(1);
  }
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
    fprintf(stderr, "%s: child exited with status %d\\n", what,
            WIFEXITED(st) ? WEXITSTATUS(st) : -1);
    exit(1);
  }
}

static void release_times(double *times, int reps, int forked) {
  if (forked)
    munmap(times, (size_t)reps * sizeof(double));
  else
    free(times);
}
"""


def generate_mim_driver(sig, input_idx, variant_name):
    name = sig.name
    params = sig.params
    out_dims = sig.return_dims
    out_count = total_elems(out_dims)

    extern_args = ", ".join(
        f"{_C_BASE[p.base][0]} *{p.cname}" for p in params
    )
    call_args = ", ".join(p.cname for p in params)

    count_decls = "\n".join(
        f"static const size_t {p.cname}_count = {total_elems(p.dims)};" for p in params
    )

    read_calls = []
    for p in params:
        if p.role == "input":
            continue
        ctype, reader = _C_BASE.get(p.base, _C_BASE["float"])
        read_calls.append(
            f'  {ctype} *{p.cname} = {reader}(wf, {p.byte_offset}, '
            f'{p.cname}_count, "{p.cname}");'
        )
    read_weights_block = "\n".join(read_calls)

    shape_comment_lines = "\n".join(
        f" * {p.cname}: shape {p.dims} ({p.role})"
        + (
            ""
            if p.role == "input"
            else f" <- {p.canonical or '?'} @ byte {p.byte_offset}"
        )
        for p in params
    )

    src = f"""#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* Auto-generated driver for `{name}` ("mim" raw-pointer calling convention).
 * Generated by gen_drivers.py -- do not hand-edit.
 *
 * Parameter shapes (in declaration order):
{shape_comment_lines}
 * output: shape {out_dims}
 *
 * The LLVM signature may declare this as a single struct-of-pointers
 * argument or as separate pointer arguments -- either way the SysV x86-64
 * ABI (for a non-`byval` aggregate, or for separate scalar pointer args)
 * flattens it into one pointer per calling-convention slot in order, so a
 * flat `float *` extern declaration reproduces the real calling convention
 * with no translation layer.
 *
 * Usage: {name}_driver_mim <weights> <input> <reps> <output> [run_id] [json_report]
 * run_id/json_report default to a generated UTC timestamp / a filename
 * derived from it if omitted -- pass the same run_id across every model
 * and pipeline in one batch (e.g. from reporter.py) to correlate their
 * timing reports.
 */
extern float *{name}({extern_args});

{count_decls}
static const size_t OUT_COUNT = {out_count};

{c_read_floats_helper()}
{c_typed_read_helpers({p.base for p in params})}
{c_timing_report_helper()}
{c_fork_helpers()}
static void write_result(float *out, const char *output_path) {{
  printf("writing output to %s...\\n", output_path);
  fflush(stdout);
  FILE *of = fopen(output_path, "wb");
  if (!of) {{
    fprintf(stderr, "failed to open output file '%s': ", output_path);
    perror(NULL);
    exit(1);
  }}
  size_t wrote = fwrite(out, sizeof(float), OUT_COUNT, of);
  fclose(of);
  if (wrote != OUT_COUNT) {{
    fprintf(stderr, "short write on output: wanted %zu floats, wrote %zu\\n",
            OUT_COUNT, wrote);
    exit(1);
  }}
  free(out);
}}

int main(int argc, char **argv) {{
  printf("reading weights...\\n");
  fflush(stdout);
  const char *weights_path = argc > 1 ? argv[1] : "weights.bin";
  const char *input_path = argc > 2 ? argv[2] : "input.bin";
  int reps = argc > 3 ? atoi(argv[3]) : 5;
  const char *output_path = argc > 4 ? argv[4] : "output_mim.bin";

  char run_id_buf[64];
  resolve_run_id(argc, argv, run_id_buf, sizeof(run_id_buf));
  const char *run_id = run_id_buf;

  char json_path_buf[512];
  const char *json_report_path;
  if (argc > 6) {{
    json_report_path = argv[6];
  }} else {{
    snprintf(json_path_buf, sizeof(json_path_buf), "{name}_{variant_name}_%s_timing.json", run_id);
    json_report_path = json_path_buf;
  }}

  FILE *wf = fopen(weights_path, "rb");
  if (!wf) {{
    fprintf(stderr, "failed to open weights file '%s': ", weights_path);
    perror(NULL);
    return 1;
  }}

{read_weights_block}
  fclose(wf);

  printf("weights loaded.\\n");
  fflush(stdout);

  printf("reading input...\\n");
  fflush(stdout);
  FILE *xf = fopen(input_path, "rb");
  if (!xf) {{
    fprintf(stderr, "failed to open input file '%s': ", input_path);
    perror(NULL);
    return 1;
  }}
  float *x = read_floats(xf, x_count, "x");
  fclose(xf);

  printf("input loaded.\\n");
  fflush(stdout);
  const int use_fork = driver_fork_enabled();
  double *times = use_fork ? alloc_shared_times(reps)
                           : (double *)malloc(reps * sizeof(double));
  printf("call isolation: %s\\n", use_fork ? "fork per call" : "in-process (DRIVER_NO_FORK)");

  printf("start warm-up (1 call)...\\n");
  fflush(stdout);
  if (use_fork) {{
    fflush(NULL);
    pid_t p = fork();
    if (p < 0) {{ perror("fork"); return 1; }}
    if (p == 0) {{
      float *out = {name}({call_args});
      free(out);
      _exit(0);
    }}
    await_child(p, "warm-up");
  }} else {{
    float *out = {name}({call_args});
    free(out);
  }}
  printf("warm-up done.\\n");
  fflush(stdout);
  printf("start timed reps...\\n");
  fflush(stdout);
  float *out = NULL;
  for (int i = 0; i < reps; i++) {{
    const int is_last = (i == reps - 1);
    if (use_fork) {{
      fflush(NULL);
      pid_t p = fork();
      if (p < 0) {{ perror("fork"); return 1; }}
      if (p == 0) {{
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        float *cur = {name}({call_args});
        clock_gettime(CLOCK_MONOTONIC, &t1);
        times[i] = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                   (double)(t1.tv_nsec - t0.tv_nsec);
        /* The last rep's child owns writing the result -- that keeps the
           output buffer from having to cross the process boundary.
           write_result() frees `cur`. */
        if (is_last)
          write_result(cur, output_path);
        _exit(0);
      }}
      await_child(p, "timed rep");
    }} else {{
      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      float *cur = {name}({call_args});
      clock_gettime(CLOCK_MONOTONIC, &t1);
      times[i] = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                 (double)(t1.tv_nsec - t0.tv_nsec);
      if (out)
        free(out);
      out = cur;
    }}
  }}
  printf("finish timed reps...\\n");
  fflush(stdout);

  if (!use_fork)
    write_result(out, output_path);

  qsort(times, reps, sizeof(double), cmp_double);
  double sum = 0;
  for (int i = 0; i < reps; i++)
    sum += times[i];

  printf("pipeline=mim->llvm weights=%s input=%s reps=%d median_ns=%.1f "
         "mean_ns=%.1f min_ns=%.1f max_ns=%.1f\\n",
         weights_path, input_path, reps, times[reps / 2], sum / reps, times[0],
         times[reps - 1]);

  write_timing_report(json_report_path, "{name}", "{variant_name}", run_id,
                       weights_path, input_path, output_path, reps,
                       times[reps / 2], sum / reps, times[0], times[reps - 1]);

  free(x);
  release_times(times, reps, use_fork);
  return 0;
}}
"""
    return src


# ---------------------------------------------------------------------------
# C code generation: mlir style (expanded memref descriptors)
# ---------------------------------------------------------------------------


def generate_mlir_driver(sig, shape_sig, input_idx, variant_name, source_desc=None):
    """sig: Signature of this .ll (memref layout, no dims); shape_sig: the
    sibling .mlir's Signature (dims, names), matched by position."""
    name = sig.name
    if len(sig.params) != len(shape_sig.params):
        raise ValueError(
            f"{name}: mlir signature has {len(sig.params)} memref groups but "
            f"shape source has {len(shape_sig.params)} params -- cannot match"
        )

    # Copy dims/role/cname from the canonical shape source onto our params,
    # positionally, and sanity-check the rank agrees.
    for p, sp, (start, end, rank) in zip(sig.params, shape_sig.params, sig.mlir_groups):
        if len(sp.dims) != rank:
            raise ValueError(
                f"{name}: param rank mismatch (mlir group rank {rank} vs "
                f"shape source rank {len(sp.dims)})"
            )
        p.dims = sp.dims
        p.role = sp.role
        p.cname = sp.cname
        p.arg_name = sp.arg_name
        p.canonical = sp.canonical
        p.byte_offset = sp.byte_offset
        p.base = sp.base

    if len(shape_sig.results) != 1 or (sig.results and len(sig.results) != 1):
        raise ValueError(f"{name}: only single-result models are supported")
    if sig.results and len(sig.results[0][0]) != len(shape_sig.results[0][0]):
        raise ValueError(f"{name}: result rank mismatch between the .ll and its .mlir")
    out_dims = shape_sig.results[0][0]
    out_rank = len(out_dims)
    result_t = f"memref{out_rank}d"
    result_typedef = ""
    free_out_4 = "    free(out.alloc);"
    free_out_6 = "      free(out.alloc);"
    free_out_8 = "        free(out.alloc);"
    free_cur_10 = "          free(cur.alloc);"
    write_body = (
        "  size_t n_out = 1;\n"
        f"  for (int i = 0; i < {out_rank}; i++)\n"
        "    n_out *= (size_t)out.sizes[i];\n"
        "  size_t wrote = fwrite(out.aligned, sizeof(float), n_out, of);"
    )
    out_comment = f" * output: shape {out_dims}"

    input_p = next(p for p in sig.params if p.role == "input")

    # One memrefRd typedef per (rank, element type) used.
    combos = {}
    for p in sig.params:
        combos.setdefault((len(p.dims), p.base), None)
    combos.setdefault((out_rank, "float"), None)


    def _mtype(r, base):
        return f"memref{r}d" if base == "float" else f"memref{r}d_{base}"


    typedefs = "\n".join(
        f"typedef struct {{\n"
        f"  {_C_BASE[base][0]} *alloc, *aligned;\n"
        f"  int64_t offset;\n"
        f"  int64_t sizes[{r}];\n"
        f"  int64_t strides[{r}];\n"
        f"}} {_mtype(r, base)};"
        for r, base in sorted(combos)
    )

    # Extern parameter list -- must exactly mirror this .ll's own scalarized
    # signature (types/order), using our derived names for readability.
    extern_parts = []
    for p, (start, end, rank) in zip(sig.params, sig.mlir_groups):
        ctype = _C_BASE.get(p.base, _C_BASE["float"])[0]
        extern_parts.append(f"{ctype} *{p.cname}_a")
        extern_parts.append(f"{ctype} *{p.cname}_al")
        extern_parts.append(f"int64_t {p.cname}_off")
        for k in range(rank):
            extern_parts.append(f"int64_t {p.cname}_s{k}")
        for k in range(rank):
            extern_parts.append(f"int64_t {p.cname}_st{k}")
    extern_args = ",\n    ".join(extern_parts)

    # Call-site literal arguments (alloc==aligned==the loaded buffer, offset
    # 0, sizes from the canonical shape, row-major contiguous strides).
    call_parts = []
    for p in sig.params:
        strides = row_major_strides(p.dims)
        call_parts.append(f"{p.cname}, {p.cname}, 0")
        call_parts.append(", ".join(str(d) for d in p.dims))
        call_parts.append(", ".join(str(s) for s in strides))
    call_args = ",\n        ".join(call_parts)

    count_decls = "\n".join(
        f"static const size_t {p.cname}_count = {total_elems(p.dims)};"
        for p in sig.params
    )

    read_calls = "\n".join(
        f'  {_C_BASE.get(p.base, _C_BASE["float"])[0]} *{p.cname} = '
        f'{_C_BASE.get(p.base, _C_BASE["float"])[1]}(wf, {p.byte_offset}, '
        f'{p.cname}_count, "{p.cname}");'
        for p in sig.params
        if p.role != "input"
    )

    shape_comment_lines = "\n".join(
        f" * {p.cname}: shape {p.dims} ({p.role})"
        + (
            ""
            if p.role == "input"
            else f" <- {p.canonical or '?'} @ byte {p.byte_offset}"
        )
        for p in sig.params
    )

    src = f"""#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* Auto-generated driver for `{name}` matching MLIR's expanded memref
 * descriptor calling convention (convert-func-to-llvm output). Generated
 * by gen_drivers.py -- do not hand-edit.
 *
 * Shapes are not recoverable from this .ll's own (fully scalarized)
 * signature, so they were borrowed from {source_desc or shape_sig.name}
 * -- which is a file for THIS pipeline, so its argument order matches
 * one-for-one what convert-func-to-llvm scalarized. Each weight's location
 * in weights.bin is its byte offset from {shape_sig.name}_params.json,
 * looked up by canonical torch parameter name:
{shape_comment_lines}
{out_comment}
 *
 * Usage: {name}_driver_mlir <weights> <input> <reps> <output> [run_id] [json_report]
 * run_id/json_report default to a generated UTC timestamp / a filename
 * derived from it if omitted -- pass the same run_id across every model
 * and pipeline in one batch (e.g. from reporter.py) to correlate their
 * timing reports.
 */

{typedefs}{result_typedef}

extern {result_t} {name}(
    {extern_args});

{count_decls}

{c_read_floats_helper()}
{c_typed_read_helpers({p.base for p in sig.params})}
{c_timing_report_helper()}
{c_fork_helpers()}
static void write_result({result_t} out, const char *output_path) {{
  printf("writing output to %s...\\n", output_path);
  fflush(stdout);
  FILE *of = fopen(output_path, "wb");
  if (!of) {{
    fprintf(stderr, "failed to open output file '%s': ", output_path);
    perror(NULL);
    exit(1);
  }}
{write_body}
  fclose(of);
  if (wrote != n_out) {{
    fprintf(stderr, "short write on output: wanted %zu floats, wrote %zu\\n",
            n_out, wrote);
    exit(1);
  }}
}}

int main(int argc, char **argv) {{
  printf("reading weights...\\n");
  fflush(stdout);
  const char *weights_path = argc > 1 ? argv[1] : "weights.bin";
  const char *input_path = argc > 2 ? argv[2] : "input.bin";
  int reps = argc > 3 ? atoi(argv[3]) : 5;
  const char *output_path = argc > 4 ? argv[4] : "output_torchmlir.bin";

  char run_id_buf[64];
  resolve_run_id(argc, argv, run_id_buf, sizeof(run_id_buf));
  const char *run_id = run_id_buf;

  char json_path_buf[512];
  const char *json_report_path;
  if (argc > 6) {{
    json_report_path = argv[6];
  }} else {{
    snprintf(json_path_buf, sizeof(json_path_buf), "{name}_{variant_name}_%s_timing.json", run_id);
    json_report_path = json_path_buf;
  }}

  FILE *wf = fopen(weights_path, "rb");
  if (!wf) {{
    fprintf(stderr, "failed to open weights file '%s': ", weights_path);
    perror(NULL);
    return 1;
  }}

{read_calls}
  fclose(wf);

  printf("weights loaded.\\n");
  fflush(stdout);

  printf("reading input...\\n");
  fflush(stdout);
  FILE *xf = fopen(input_path, "rb");
  if (!xf) {{
    fprintf(stderr, "failed to open input file '%s': ", input_path);
    perror(NULL);
    return 1;
  }}
  float *{input_p.cname} = read_floats(xf, {input_p.cname}_count, "{input_p.cname}");
  fclose(xf);

  printf("input loaded.\\n");
  fflush(stdout);
  const int use_fork = driver_fork_enabled();
  double *times = use_fork ? alloc_shared_times(reps)
                           : (double *)malloc(reps * sizeof(double));
  printf("call isolation: %s\\n", use_fork ? "fork per call" : "in-process (DRIVER_NO_FORK)");

  printf("start warm-up (1 call)...\\n");
  fflush(stdout);
  if (use_fork) {{
    fflush(NULL);
    pid_t p = fork();
    if (p < 0) {{ perror("fork"); return 1; }}
    if (p == 0) {{
      {result_t} out = {name}(
          {call_args});
{free_out_6}
      _exit(0);
    }}
    await_child(p, "warm-up");
  }} else {{
    {result_t} out = {name}(
        {call_args});
{free_out_4}
  }}
  printf("warm-up done.\\n");
  fflush(stdout);
  printf("start timed reps...\\n");
  fflush(stdout);
  int keep_out = 0;
  {result_t} out;
  for (int i = 0; i < reps; i++) {{
    const int is_last = (i == reps - 1);
    if (use_fork) {{
      fflush(NULL);
      pid_t p = fork();
      if (p < 0) {{ perror("fork"); return 1; }}
      if (p == 0) {{
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        {result_t} cur = {name}(
            {call_args});
        clock_gettime(CLOCK_MONOTONIC, &t1);
        times[i] = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                   (double)(t1.tv_nsec - t0.tv_nsec);
        /* The last rep's child owns writing the result, so the output
           buffer never has to cross the process boundary. */
        if (is_last) {{
          write_result(cur, output_path);
{free_cur_10}
        }}
        _exit(0);
      }}
      await_child(p, "timed rep");
    }} else {{
      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      {result_t} cur = {name}(
          {call_args});
      clock_gettime(CLOCK_MONOTONIC, &t1);
      times[i] = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                 (double)(t1.tv_nsec - t0.tv_nsec);
      if (keep_out)
{free_out_8}
      out = cur;
      keep_out = 1;
    }}
  }}
  printf("finish timed reps...\\n");
  fflush(stdout);

  if (!use_fork) {{
    write_result(out, output_path);
{free_out_4}
  }}

  qsort(times, reps, sizeof(double), cmp_double);
  double sum = 0;
  for (int i = 0; i < reps; i++)
    sum += times[i];

  printf("pipeline=mlir->llvm weights=%s input=%s reps=%d median_ns=%.1f "
         "mean_ns=%.1f min_ns=%.1f max_ns=%.1f\\n",
         weights_path, input_path, reps, times[reps / 2], sum / reps, times[0],
         times[reps - 1]);

  write_timing_report(json_report_path, "{name}", "{variant_name}", run_id,
                       weights_path, input_path, output_path, reps,
                       times[reps / 2], sum / reps, times[0], times[reps - 1]);

  free({input_p.cname});
  release_times(times, reps, use_fork);
  return 0;
}}
"""
    return src


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def resolve_blob_layout(ll_path, params, model_name, name_source=None):
    """Bind `params` to the model's weight manifest. Returns (ok, note); not ok
    means no driver may be written."""
    model_root = find_model_root(ll_path)
    try:
        manifest = param_manifest.find_manifest(model_root, model_name)
    except Exception as e:
        print(f"    ERROR: {e}")
        return False, None
    if manifest is None:
        print(
            f"    ERROR: no {model_name}_params.json in {model_root} -- run "
            f"gen_data.py first (it writes the manifest alongside the blob)"
        )
        return False, None

    problem = validate_input_against_manifest(params, manifest, model_name)
    if problem:
        print(f"    ERROR: {problem}")
        return False, None

    mode, message = bind_to_manifest(params, manifest, model_name)
    if mode == "error":
        print(f"    ERROR: {message}")
        return False, None
    if mode in ("positional", "by_name_unread"):
        print(f"    WARNING: {message}")
    if name_source:
        message = f"{message} (names from {name_source})"
    return True, message


def warn_stale_driver(out_path):
    """Warn that a driver left on disk does not match the current data."""
    if os.path.exists(out_path):
        print(
            f"    *** STALE DRIVER LEFT IN PLACE: {out_path}\n"
            f"    *** It was generated against a different weights.bin layout "
            f"and will read the wrong weights.\n"
            f"    *** compile_all.sh will still build it -- delete it, or fix "
            f"the cause above and re-run with --force."
        )


def process_file(ll_path, dry_run=False, force=False):
    with open(ll_path) as f:
        text = f.read()

    sig = parse_signature(text, filename_hint=ll_path)
    dirpath = os.path.dirname(ll_path)
    variant_name = os.path.basename(dirpath)

    if sig.style == "unknown":
        print(f"[SKIP] {ll_path}: unrecognized calling convention")
        return

    if sig.style == "raw_flat":
        if None in [d for p in sig.params for d in p.dims]:
            print(f"[SKIP] {ll_path}: could not fully parse parameter shapes")
            return
        input_idx = assign_roles(sig.params, sig.return_dims)
        style_tag = "mim"
        out_path = os.path.join(dirpath, f"{sig.name}_driver_{style_tag}.c")
        print(f"[{sig.name}] {ll_path}  (style={sig.style})")
        ok, note = resolve_blob_layout(ll_path, sig.params, sig.name)
        if not ok:
            warn_stale_driver(out_path)
            print(f"[SKIP] {ll_path}: cannot bind weights.bin")
            return
        print(f"    weights: {note}")
        for p in sig.params:
            loc = "input.bin" if p.role == "input" else f"@{p.byte_offset}"
            print(
                f"    {p.cname:>5} [{p.role:>6}] : {str(p.dims):<22} "
                f"{loc:>12}  {p.canonical or ''}"
            )
        if dry_run:
            print(f"    -> (dry-run) would write {out_path}")
            return
        if not force and os.path.exists(out_path):
            print(f"    -> SKIP (already exists; use --force to overwrite)")
            return
        src = generate_mim_driver(sig, input_idx, variant_name)

    elif sig.style == "mlir_memref":
        shape_sig = find_sibling_mlir_shape_source(ll_path, sig.name)
        source_kind = "sibling .mlir"
        if shape_sig is None:
            print(f"[SKIP] {ll_path}: no sibling .mlir declaring @{sig.name}")
            return
        style_tag = "mlir"
        out_path = os.path.join(dirpath, f"{sig.name}_driver_{style_tag}.c")
        try:
            if any(p.role is None for p in shape_sig.params):
                assign_roles(shape_sig.params, shape_sig.return_dims)
            input_idx = next(
                i for i, p in enumerate(shape_sig.params) if p.role == "input"
            )
        except StopIteration:
            print(f"[SKIP] {ll_path}: could not determine input parameter")
            return
        print(
            f"[{sig.name}] {ll_path}  (style={sig.style}, shapes from {source_kind}: "
            f"{getattr(shape_sig, 'source_file', shape_sig.name)})"
        )
        if getattr(shape_sig, "unsupported", None):
            print(f"    ERROR: {shape_sig.unsupported}")
            warn_stale_driver(out_path)
            print(f"[SKIP] {ll_path}: unsupported model shape")
            return
        # Bind the shape source's params -- generate_mlir_driver copies dims,
        # names and offsets from them onto this .ll's scalarized groups.
        ok, note = resolve_blob_layout(
            ll_path,
            shape_sig.params,
            sig.name,
            name_source=getattr(shape_sig, "name_source", None),
        )
        if not ok:
            warn_stale_driver(out_path)
            print(f"[SKIP] {ll_path}: cannot bind weights.bin")
            return
        print(f"    weights: {note}")
        for sp in shape_sig.params:
            loc = "input.bin" if sp.role == "input" else f"@{sp.byte_offset}"
            print(
                f"    {sp.cname:>5} [{sp.role:>6}] : {str(sp.dims):<22} "
                f"{loc:>12}  {sp.canonical or ''}"
            )
        if dry_run:
            print(f"    -> (dry-run) would write {out_path}")
            return
        if not force and os.path.exists(out_path):
            print(f"    -> SKIP (already exists; use --force to overwrite)")
            return
        try:
            src = generate_mlir_driver(
                sig,
                shape_sig,
                input_idx,
                variant_name,
                source_desc=f"{source_kind} ({getattr(shape_sig, 'source_file', shape_sig.name)})",
            )
        except ValueError as e:
            print(f"[SKIP] {ll_path}: {e}")
            return

    else:
        return

    with open(out_path, "w") as f:
        f.write(src)
    print(f"    -> wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", help="Root directory to scan for model variant folders")
    ap.add_argument(
        "--dry-run", action="store_true", help="Only print what would be generated"
    )
    ap.add_argument(
        "--force", action="store_true", help="Overwrite existing driver .c files"
    )
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"Not a directory: {args.root}", file=sys.stderr)
        sys.exit(1)

    found_any = False
    for ll_path in find_all_ll_files(args.root):
        found_any = True
        try:
            process_file(ll_path, dry_run=args.dry_run, force=args.force)
        except Exception as e:
            print(f"[ERROR] {ll_path}: {e}", file=sys.stderr)

    if not found_any:
        print(f"No .ll files found under {args.root}", file=sys.stderr)


if __name__ == "__main__":
    main()
