#!/usr/bin/env python3
"""
Export KernelBench-style model files to Linalg-on-Tensors MLIR with torch-mlir.

Each file defines `Model`, `get_inputs()` and `get_init_inputs()`. Parameters and
floating-point buffers become explicit function arguments instead of constants,
and <stem>.params.txt names every argument with its shape.

Usage:
    python export_mlir.py --repo-dir ./models --out-dir ./mlir_out
    python export_mlir.py --repo-dir . --out-dir ./out --pattern "*_VGG*.py" --verbose

Output, mirroring the input tree under --out-dir:
    <stem>.mlir   <stem>.params.txt   export_report.txt
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import signal
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.func import functional_call

from torch_mlir import fx

THIS_FILE = Path(__file__).resolve()


_DECOMPOSITION_TABLE = None


def build_decomposition_table() -> dict:
    """torch-mlir's decomposition table (built once)."""
    global _DECOMPOSITION_TABLE
    if _DECOMPOSITION_TABLE is None:
        from torch_mlir.extras.fx_decomp_util import get_decomposition_table

        _DECOMPOSITION_TABLE = dict(get_decomposition_table())
    return _DECOMPOSITION_TABLE


class FunctionalWrapper(nn.Module):
    """forward(*inputs, *weights): the model with every parameter and buffer as
    an explicit argument, so the export is parametric over the weights."""

    def __init__(self, base_model: nn.Module, param_names: list[str], num_inputs: int):
        super().__init__()
        object.__setattr__(self, "_base_model", base_model)
        self.param_names = list(param_names)
        self.num_inputs = num_inputs

    def forward(self, *args):
        inputs = args[: self.num_inputs]
        params = args[self.num_inputs :]
        params_dict = dict(zip(self.param_names, params))
        return functional_call(self._base_model, params_dict, tuple(inputs))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@dataclass
class ExportResult:
    path: Path
    ok: bool
    message: str = ""
    mlir_path: Optional[Path] = None
    notes: list[str] = field(default_factory=list)


def load_module_from_path(path: Path, unique_name: str):
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


def find_model_files(repo_dir: Path, pattern: str):
    yield from sorted(repo_dir.rglob(pattern))


def sanitize_func_name(stem: str) -> str:
    name = "".join(c if (c.isalnum() or c == "_") else "_" for c in stem)
    if name and name[0].isdigit():
        name = f"m_{name}"
    return name or "model"


def as_list(x: Any) -> list:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def tensors_close(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> bool:
    return a.shape == b.shape and torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)


def outputs_close(a: Any, b: Any, atol=1e-4, rtol=1e-4) -> bool:
    """Loose comparator that handles tensors, tuples/lists, and dicts."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return tensors_close(a, b, atol, rtol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            outputs_close(x, y, atol, rtol) for x, y in zip(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            outputs_close(a[k], b[k], atol, rtol) for k in a
        )
    return a == b


# --------------------------------------------------------------------------
# Core export logic for a single file
# --------------------------------------------------------------------------
def attempt_export(
    module,
    path: Path,
    seed: int,
    verbose: bool,
    skip_verify: bool,
    strict_verify: bool,
):
    """
    One end-to-end attempt for a single already-imported model file.

    Returns (mlir_text_or_None, manifest_lines, error_message, notes).
    """
    notes: list[str] = []

    # 1. Build the model + inputs, sanity check plain forward
    try:
        torch.manual_seed(seed)
        init_args = as_list(module.get_init_inputs())
        model = module.Model(*init_args)
        model.eval()

        inputs = as_list(module.get_inputs())

        state_names = [name for name, _ in model.named_parameters()]
        num_params = len(state_names)

        # Floating-point buffers become arguments; integer ones stay constants.
        buffer_names = [
            name for name, buf in model.named_buffers() if buf.is_floating_point()
        ]
        skipped = [
            name for name, buf in model.named_buffers() if not buf.is_floating_point()
        ]
        state_names += buffer_names
        notes.append(
            f"buffers as args: {len(buffer_names)}"
            + (f"; non-float buffers left as constants: {len(skipped)}" if skipped else "")
        )

        state_dict = dict(model.named_parameters())
        state_dict.update(dict(model.named_buffers()))
        state_values = tuple(state_dict[name].detach() for name in state_names)
    except Exception as e:
        tb = traceback.format_exc() if verbose else str(e)
        return None, [], f"model construction/forward failed: {tb}", notes

    # 2. Check the wrapper reproduces the model.
    wrapper = FunctionalWrapper(model, state_names, num_inputs=len(inputs))
    wrapper.eval()

    if not skip_verify:
        try:
            with torch.no_grad():
                want = model(*inputs)
                got = wrapper(*inputs, *state_values)
            matched = outputs_close(want, got)
        except Exception as e:
            tb = traceback.format_exc() if verbose else str(e)
            return None, [], f"functional wrapper forward failed: {tb}", notes
        finally:
            # Free the reference outputs (can be GBs) before export.
            want = got = None
            gc.collect()

        if not matched:
            msg = "functional wrapper output mismatch vs. original model"
            if strict_verify:
                return None, [], msg, notes
            notes.append(
                f"WARNING: {msg} (exported anyway; use --strict-verify to fail)"
            )

    # 3. Export to Torch-MLIR / Linalg-on-Tensors (parametric over weights)
    func_name = sanitize_func_name(path.stem)
    try:
        linalg_module = fx.export_and_import(
            wrapper,
            *inputs,
            *state_values,
            func_name=func_name,
            output_type=fx.OutputType.LINALG_ON_TENSORS,
            decomposition_table=build_decomposition_table(),
        )
        mlir_text = str(linalg_module)
    except Exception as e:
        tb = traceback.format_exc() if verbose else str(e)
        return None, [], f"torch-mlir export failed: {tb}", notes

    # 4. Argument manifest
    manifest = [f"func_name: {func_name}"]
    for i, tensor in enumerate(inputs):
        shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else "?"
        manifest.append(f"arg{i}: <model input {i}>  shape={shape}")
    for i, name in enumerate(state_names):
        kind = "param" if i < num_params else "buffer"
        manifest.append(
            f"arg{i + len(inputs)}: {name}  shape={tuple(state_dict[name].shape)}  kind={kind}"
        )

    return mlir_text, manifest, "", notes


def export_one(
    path: Path,
    repo_dir: Path,
    out_dir: Path,
    seed: int,
    verbose: bool,
    skip_verify: bool = False,
    strict_verify: bool = False,
) -> ExportResult:
    rel = path.relative_to(repo_dir)
    unique_mod_name = "repo_model__" + "_".join(rel.with_suffix("").parts)

    # 1. Import the file as a module
    try:
        module = load_module_from_path(path, unique_mod_name)
    except Exception as e:
        return ExportResult(path, False, f"import failed: {e}")

    if not hasattr(module, "Model"):
        return ExportResult(path, False, "no `Model` class found, skipping")
    if not hasattr(module, "get_inputs") or not hasattr(module, "get_init_inputs"):
        return ExportResult(
            path, False, "missing get_inputs()/get_init_inputs(), skipping"
        )

    # 2. Export (parameters and float buffers as arguments).
    mlir_text, manifest, error, notes = attempt_export(
        module, path, seed, verbose, skip_verify, strict_verify
    )
    if mlir_text is None:
        return ExportResult(path, False, error, notes=notes)

    # 3. Write outputs
    out_subdir = out_dir / rel.parent
    out_subdir.mkdir(parents=True, exist_ok=True)

    mlir_path = out_subdir / f"{path.stem}.mlir"
    with open(mlir_path, "w") as f:
        f.write(mlir_text)

    manifest_path = out_subdir / f"{path.stem}.params.txt"
    with open(manifest_path, "w") as f:
        for line in manifest:
            f.write(line + "\n")
        for note in notes:
            f.write(f"# {note}\n")

    return ExportResult(path, True, "ok", mlir_path, notes=notes)


RESULT_OK_PREFIX = "RESULT_OK "
RESULT_FAIL_PREFIX = "RESULT_FAIL "
RESULT_NOTE_PREFIX = "RESULT_NOTE "


def _encode(message: str) -> str:
    """Flatten a possibly-multiline message onto a single result line."""
    return message.replace("\\", "\\\\").replace("\n", "\\n")


def _decode(message: str) -> str:
    out, i = [], 0
    while i < len(message):
        if message[i] == "\\" and i + 1 < len(message):
            nxt = message[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
        out.append(message[i])
        i += 1
    return "".join(out)


# Child mode: export one file in this process. Each file gets its own process,
# so memory cannot accumulate and one OOM does not stop the run.
def run_child(
    path: Path,
    repo_dir: Path,
    out_dir: Path,
    seed: int,
    verbose: bool,
    skip_verify: bool,
    strict_verify: bool,
) -> int:
    result = export_one(
        path, repo_dir, out_dir, seed, verbose, skip_verify, strict_verify
    )
    for note in result.notes:
        print(f"{RESULT_NOTE_PREFIX}{_encode(note)}")
    if result.ok:
        print(f"{RESULT_OK_PREFIX}{result.mlir_path}")
        return 0
    else:
        print(f"{RESULT_FAIL_PREFIX}{_encode(result.message)}")
        return 1


# Driver mode: one child process per file.
def run_driver(args) -> None:
    repo_dir: Path = args.repo_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(find_model_files(repo_dir, args.pattern))
    if not files:
        print(f"No files matching {args.pattern!r} found under {repo_dir}")
        return

    results: list[ExportResult] = []
    for path in files:
        rel = path.relative_to(repo_dir)
        expected_mlir = out_dir / rel.parent / f"{path.stem}.mlir"

        if args.skip_existing and expected_mlir.exists():
            print(f"[{rel}] already exported, skipping (--skip-existing)")
            results.append(
                ExportResult(path, True, "skipped (already exists)", expected_mlir)
            )
            continue

        print(f"[{rel}] exporting (subprocess)...", flush=True)
        cmd = [
            sys.executable,
            str(THIS_FILE),
            "--repo-dir",
            str(repo_dir),
            "--out-dir",
            str(out_dir),
            "--seed",
            str(args.seed),
            "--single-file",
            str(path),
        ]
        if args.verbose:
            cmd.append("--verbose")
        if args.skip_verify:
            cmd.append("--skip-verify")
        if args.strict_verify:
            cmd.append("--strict-verify")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout,  # None = no timeout
            )
        except subprocess.TimeoutExpired:
            msg = f"child process timed out after {args.timeout}s"
            results.append(ExportResult(path, False, msg))
            print(f"  FAILED: {msg}")
            if args.stop_on_error:
                break
            continue

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        notes = [
            _decode(line[len(RESULT_NOTE_PREFIX) :])
            for line in stdout.splitlines()
            if line.startswith(RESULT_NOTE_PREFIX)
        ]

        if proc.returncode < 0:
            # Negative returncode from subprocess.run means the child was
            # killed by a signal (e.g. -9 == SIGKILL, almost always OOM).
            sig = -proc.returncode
            try:
                sig_name = signal.Signals(sig).name
            except ValueError:
                sig_name = str(sig)
            msg = f"child process killed by signal {sig_name} (likely out-of-memory)"
            results.append(ExportResult(path, False, msg, notes=notes))
            print(f"  FAILED: {msg}")
        elif proc.returncode == 0 and RESULT_OK_PREFIX in stdout:
            mlir_path_str = ""
            for line in reversed(stdout.strip().splitlines()):
                if line.startswith(RESULT_OK_PREFIX):
                    mlir_path_str = line[len(RESULT_OK_PREFIX) :]
                    break
            results.append(
                ExportResult(path, True, "ok", Path(mlir_path_str), notes=notes)
            )
            print(f"  ok -> {mlir_path_str}")
            for note in notes:
                print(f"  note: {note}")
        else:
            fail_msg = None
            for line in reversed(stdout.strip().splitlines()):
                if line.startswith(RESULT_FAIL_PREFIX):
                    fail_msg = _decode(line[len(RESULT_FAIL_PREFIX) :])
                    break
            if fail_msg is None:
                stderr_tail = stderr.strip().splitlines()[-1] if stderr.strip() else ""
                fail_msg = stderr_tail or f"child exited with code {proc.returncode}"
            results.append(ExportResult(path, False, fail_msg, notes=notes))
            print(f"  FAILED: {fail_msg.splitlines()[0] if fail_msg else fail_msg}")
            if args.verbose and stderr:
                print(stderr)

        if not results[-1].ok and args.stop_on_error:
            break

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print("\n==================== SUMMARY ====================")
    print(f"Total:  {len(results)}")
    print(f"OK:     {len(ok)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("\nFailures:")
        for r in failed:
            first = r.message.splitlines()[0] if r.message else ""
            print(f"  {r.path}: {first}")

    report_path = out_dir / "export_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Total: {len(results)}  OK: {len(ok)}  Failed: {len(failed)}\n\n")
        for r in results:
            status = "OK" if r.ok else "FAILED"
            f.write(f"[{status}] {r.path}\n")
            if r.ok:
                f.write(f"    -> {r.mlir_path}\n")
            else:
                for line in r.message.splitlines() or [""]:
                    f.write(f"    {line}\n")
            for note in r.notes:
                f.write(f"    # {note}\n")
    print(f"\nWrote report to {report_path}")

    if failed:
        print(
            f"\n{len(failed)} file(s) failed. Fix/skip as needed and re-run with "
            f"--skip-existing to resume without redoing completed work."
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        required=True,
        help="Root directory to walk for model files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write .mlir/.params.txt files into",
    )
    parser.add_argument(
        "--pattern", default="*.py", help="Glob pattern for model files (default: *.py)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="torch.manual_seed value before init/inputs"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the whole run on the first failure (default: keep going)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose .mlir output already exists in --out-dir (use to resume a crashed/killed run)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-file timeout in seconds for the export subprocess (default: no timeout)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the eager reference forward passes. Cuts peak memory a lot for "
        "models with huge activations (use if a file keeps getting OOM-killed)",
    )
    parser.add_argument(
        "--strict-verify",
        action="store_true",
        help="Treat a wrapper/model output mismatch as a hard failure instead of "
        "a warning recorded alongside the export",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print full tracebacks on failure"
    )
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,  # internal: used when the driver re-invokes this script per file
    )
    args = parser.parse_args()

    if args.single_file is not None:
        # Child mode (see run_child).
        repo_dir = args.repo_dir.resolve()
        out_dir = args.out_dir.resolve()
        rc = run_child(
            args.single_file.resolve(),
            repo_dir,
            out_dir,
            args.seed,
            args.verbose,
            args.skip_verify,
            args.strict_verify,
        )
        sys.exit(rc)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
