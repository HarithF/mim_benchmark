#!/usr/bin/env python3
"""
produce_refs.py -- compute each model's PyTorch reference output once and cache
it next to the model's data, so reporter.py does not rerun PyTorch every time.

    scripts/produce_refs.py .                # fill in whatever is missing
    scripts/produce_refs.py . --force        # recompute everything
    scripts/produce_refs.py . --check        # report status, write nothing
    scripts/produce_refs.py 30_SwinTransformerV2

Writes <Name>_ref.bin (the output, raw float32) and <Name>_ref.json (what it
was computed from). The cache is used only while that signature still matches:
the reference .py and <Name>_params.json by hash, weights.bin and input.bin by
size and mtime (hash if the mtime moved).

Each model runs in its own subprocess, which keeps memory bounded: a large
model's PyTorch forward pass (tens of GB) is released when its process exits
instead of accumulating across models. Caching also takes this load off reporter.py,
which handles every model in one process and would otherwise run each forward pass there.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import param_manifest  # noqa: E402
import reporter  # noqa: E402

CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Cache signature
# ---------------------------------------------------------------------------


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sig(path):
    """Size, mtime and sha256 of one blob."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def blob_matches(recorded, path):
    """Is `path` still the recorded file? Size must match; a matching mtime
    accepts, otherwise the sha256 decides (copies move mtimes, not bytes)."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    if recorded is None or recorded.get("size") != st.st_size:
        return False
    if recorded.get("mtime_ns") == st.st_mtime_ns:
        return True
    digest = recorded.get("sha256")
    return bool(digest) and digest == sha256_file(path)


def build_signature(model_dir, name, ref_module, hash_blobs=False):
    """Everything the reference depends on, or None if anything is missing.
    Blobs are hashed only when producing (hash_blobs=True), not when checking."""
    manifest_path = param_manifest.manifest_path(model_dir, name)
    weights = os.path.join(model_dir, f"{name}_weights.bin")
    inputs = os.path.join(model_dir, f"{name}_input.bin")
    if not (
        ref_module
        and os.path.exists(ref_module)
        and os.path.exists(manifest_path)
        and os.path.exists(weights)
        and os.path.exists(inputs)
    ):
        return None
    sig = {
        "ref_module": os.path.relpath(ref_module, model_dir),
        "ref_module_sha256": sha256_file(ref_module),
        "manifest_sha256": sha256_file(manifest_path),
    }
    if hash_blobs:
        sig["weights"] = file_sig(weights)
        sig["input"] = file_sig(inputs)
    return sig


def cache_paths(model_dir, name):
    return (
        os.path.join(model_dir, f"{name}_ref.bin"),
        os.path.join(model_dir, f"{name}_ref.json"),
    )


def load_cache_meta(model_dir, name):
    _bin_path, meta_path = cache_paths(model_dir, name)
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def cache_state(model_dir, name, ref_module):
    """('valid'|'stale'|'absent'|'unusable', message); 'unusable' means there
    is nothing to compute (no reference .py, manifest or data)."""
    sig = build_signature(model_dir, name, ref_module)
    if sig is None:
        return "unusable", "no reference .py, manifest, weights.bin or input.bin"
    meta = load_cache_meta(model_dir, name)
    if meta is None:
        return "absent", "no cached reference"
    if meta.get("version") != CACHE_VERSION:
        return "stale", f"cache version {meta.get('version')} != {CACHE_VERSION}"

    bin_path, _ = cache_paths(model_dir, name)
    try:
        on_disk = os.path.getsize(bin_path)
    except OSError:
        return "stale", f"{os.path.basename(bin_path)} is missing"
    want = meta.get("count", -1) * 4
    if on_disk != want:
        return (
            "stale",
            f"{os.path.basename(bin_path)} is {on_disk} bytes, expected {want}",
        )

    for key, what in (
        ("ref_module_sha256", "the reference .py changed"),
        ("manifest_sha256", f"{name}_params.json changed"),
    ):
        if meta.get(key) != sig[key]:
            return "stale", what

    for key, path, what in (
        ("weights", os.path.join(model_dir, f"{name}_weights.bin"), "weights.bin"),
        ("input", os.path.join(model_dir, f"{name}_input.bin"), "input.bin"),
    ):
        if not blob_matches(meta.get(key), path):
            return "stale", f"{what} changed"
    return "valid", "up to date"


# ---------------------------------------------------------------------------
# Producing one reference (runs in the child process)
# ---------------------------------------------------------------------------


def produce_one(model_dir):
    """Compute and write one model's reference. Returns an exit status int."""
    import numpy as np

    found = reporter.find_weights_and_input(model_dir)
    if found is None:
        print(f"[{model_dir}] SKIP: no <Name>_weights.bin", file=sys.stderr)
        return 2
    name, weights_path, input_path = found

    ref_module = reporter.find_torch_ref_module(model_dir, name)
    state, why = cache_state(model_dir, name, ref_module)
    if state == "unusable":
        print(f"[{name}] SKIP: {why}", file=sys.stderr)
        return 2

    try:
        manifest = param_manifest.find_manifest(model_dir, name)
    except Exception as e:
        print(f"[{name}] SKIP: {e}", file=sys.stderr)
        return 2

    shape = reporter.find_shape_source(model_dir, expected_name=name)

    ref = reporter.run_torch_reference(
        ref_module, weights_path, input_path, shape, manifest
    )
    if ref is None:
        # run_torch_reference already explained itself on stderr.
        print(f"[{name}] FAILED: no reference produced", file=sys.stderr)
        return 1

    bin_path, meta_path = cache_paths(model_dir, name)
    # Blobs are hashed here, once, so checking the cache never has to.
    sig = build_signature(model_dir, name, ref_module, hash_blobs=True)
    ref = np.ascontiguousarray(ref, dtype=np.float32)
    ref.tofile(bin_path)

    meta = {
        "version": CACHE_VERSION,
        "model": name,
        "count": int(ref.size),
        "dtype": "float32",
        "output_dims": list(shape.output_dims) if shape else None,
        "mean_abs": float(np.abs(ref).mean()),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **sig,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")

    print(
        f"[{name}] wrote {os.path.basename(bin_path)} "
        f"({ref.size} floats, {ref.nbytes / 1e9:.2f} GB, mean|ref|={meta['mean_abs']:.6g})"
    )
    return 0


# ---------------------------------------------------------------------------
# Driving the children
# ---------------------------------------------------------------------------


def run_child(model_dir, force):
    """Compute one model's reference in a subprocess (an OOM kill or crash
    costs that model only)."""
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--one", model_dir]
    if force:
        cmd.append("--force")
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "root", nargs="?", default=".", help="A model directory, or a root of many"
    )
    ap.add_argument(
        "--force", action="store_true", help="recompute even where the cache is valid"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="report each model's cache state and write nothing",
    )
    ap.add_argument(
        "--one",
        metavar="MODEL_DIR",
        help="internal: compute exactly this model, in this process",
    )
    args = ap.parse_args()

    # Child mode.
    if args.one:
        model_dir = args.one
        found = reporter.find_weights_and_input(model_dir)
        if found is not None and not args.force:
            name = found[0]
            ref_module = reporter.find_torch_ref_module(model_dir, name)
            state, why = cache_state(model_dir, name, ref_module)
            if state == "valid":
                print(f"[{name}] cached reference is up to date")
                return 0
        return produce_one(model_dir)

    model_dirs = reporter.find_model_dirs(args.root)
    if not model_dirs:
        print(f"No model directories found under {args.root}", file=sys.stderr)
        return 1

    if args.check:
        widths = max(len(d) for d in model_dirs)
        counts = {}
        for d in model_dirs:
            found = reporter.find_weights_and_input(d)
            if found is None:
                state, why = "unusable", "no <Name>_weights.bin"
            else:
                name = found[0]
                state, why = cache_state(
                    d, name, reporter.find_torch_ref_module(d, name)
                )
            counts[state] = counts.get(state, 0) + 1
            print(f"{d:<{widths}}  {state:<9} {why}")
        print()
        print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
        return 0 if counts.get("stale", 0) == 0 else 1

    ok, skipped, failed, killed = [], [], [], []
    for d in model_dirs:
        rc = run_child(d, args.force)
        if rc == 0:
            ok.append(d)
        elif rc == 2:
            skipped.append(d)
        elif rc < 0 or rc >= 128:
            # Died on a signal (-9/137: OOM killer); report it, the child could not.
            sig = -rc if rc < 0 else rc - 128
            killed.append((d, sig))
            print(
                f"!! {d}: reference process died on signal {sig}"
                + (" (out of memory)" if sig == 9 else ""),
                file=sys.stderr,
            )
        else:
            failed.append(d)

    print()
    print(f"==> {len(ok)} reference(s) written or already current")
    if skipped:
        print(f"    {len(skipped)} skipped (nothing to compute)")
    for d in skipped:
        print(f"      - {d}")
    if failed:
        print(f"    {len(failed)} failed")
    for d in failed:
        print(f"      - {d}")
    if killed:
        print(f"    {len(killed)} killed by a signal")
    for d, sig in killed:
        print(f"      - {d} (signal {sig})")
    return 0 if not failed and not killed else 1


if __name__ == "__main__":
    sys.exit(main())
