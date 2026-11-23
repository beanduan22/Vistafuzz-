from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vistafuzz.native import LIBRARIES, get_runtime
from vistafuzz.native_base import resolve_object
from vistafuzz.collector import collect_api, has_runtime_signature
from vistafuzz.materialize import materialize
from vistafuzz.models import Finding, ParamSpec, Value
from vistafuzz.oracles import _iter_numeric
from vistafuzz.reproducer import render


DTYPES = ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
          "uint64", "float16", "float32", "float64", "complex64", "complex128")
APIS = {"numpy": "numpy.negative", "scipy": "scipy.linalg.norm",
        "sklearn": "sklearn.preprocessing.normalize", "cv2": "cv2.sqrt",
        "torch": "torch.neg", "tensorflow": "tensorflow.negative",
        "jax": "jax.numpy.negative", "keras": "keras.ops.negative",
        "paddle": "paddle.neg", "mindspore": "mindspore.ops.neg",
        "oneflow": "oneflow.neg", "chainer": "chainer.functions.absolute"}


def readback(value):
    arrays = list(_iter_numeric(value, include_bool=True))
    if len(arrays) != 1:
        raise AssertionError(f"expected one numerical array, got {len(arrays)}")
    return arrays[0]


def check_conversion(runtime, label, value, dtype):
    expected = np.asarray(value, dtype=dtype)
    row = {"case": label, "dtype": str(expected.dtype), "shape": list(expected.shape)}
    try:
        native = runtime.array(value, dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        return {**row, "status": "rejected", "reason": type(exc).__name__ + ": " + str(exc)}
    try:
        actual = readback(native)
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise AssertionError(f"expected {expected.dtype}/{expected.shape}, got {actual.dtype}/{actual.shape}")
        np.testing.assert_array_equal(actual, expected)
        if expected.dtype.kind == "f":
            np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    except (AssertionError, TypeError, ValueError) as exc:
        return {**row, "status": "failed", "reason": str(exc)}
    return {**row, "status": "passed"}


def check(library):
    module = importlib.import_module(library)
    runtime = get_runtime(library)
    rows = []
    rows.append(check_conversion(runtime, "inferred_float64", np.array([0.1]), None))
    for dtype in DTYPES:
        for shape in ((), (3,), (2, 3), (0, 3)):
            rows.append(check_conversion(runtime, "shape_dtype", np.ones(shape, dtype=dtype), dtype))
    strided = np.arange(6, dtype="float64").reshape(2, 3)[:, ::-1]
    strided.flags.writeable = False
    rows.append(check_conversion(runtime, "readonly_negative_stride", strided, "float64"))
    rows.append(check_conversion(runtime, "fortran_layout", np.asfortranarray(strided), "float64"))
    rows.append(check_conversion(runtime, "non_native_byteorder", strided.astype(">f8"), "float64"))
    rows.append(check_conversion(runtime, "nonfinite_and_signed_zero",
                                 np.array([np.nan, np.inf, -np.inf, -0.0, 0.0]), "float64"))
    rows.append(check_conversion(runtime, "uint64_boundaries",
                                 np.array([0, 2**63, 2**64 - 1], dtype="uint64"), "uint64"))
    api = APIS[library]
    values = np.array([[1.0, 4.0], [9.0, 16.0]])
    spec = ParamSpec(name="input", kind="array", position=0, call_kind="positional")
    recipe = {"form": "literal_array", "dtype": "float64", "values": values.tolist()}
    value = Value(kind="array", payload=values, dtype="float64", shape=values.shape, recipe=recipe)
    positional, keyword = materialize(api, [spec], {"input": value})
    actual = readback(resolve_object(api)(*positional, **keyword))
    expected = {"scipy": lambda: np.linalg.norm(values),
                "sklearn": lambda: values / np.linalg.norm(values, axis=1, keepdims=True),
                "cv2": lambda: np.sqrt(values),
                "chainer": lambda: np.abs(values)}.get(library, lambda: -values)()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    record = collect_api(api)
    finding = Finding(api=api, oracle="nan", kind="non_finite_output", order=["input"],
                      args={"input": {"recipe": recipe, "by_keyword": False}})
    with tempfile.TemporaryDirectory(prefix="vistafuzz-runtime-reproducer-") as temp:
        path = Path(temp) / "repro.py"
        path.write_text(render(finding))
        child = subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
                               timeout=30, env={**os.environ, "PYTHONPATH": ""})
        if child.returncode != 0 or "calling " not in child.stdout or "OK:" not in child.stdout:
            raise AssertionError("standalone reproducer failed: " + child.stderr[-1500:] + child.stdout[-500:])
    return {"library": library, "version": str(getattr(module, "__version__", "unknown")),
            "backend": module.backend.backend() if library == "keras" else library,
            "python": sys.version.split()[0], "conversions": rows,
            "counts": {status: sum(row["status"] == status for row in rows)
                       for status in ("passed", "rejected", "failed")},
            "api": api, "api_call": "passed", "standalone_reproducer": "passed",
            "runtime_signature": record is not None and has_runtime_signature(record)}


def main():
    parser = argparse.ArgumentParser(description="Check one library runtime in its own Python environment")
    parser.add_argument("--lib", choices=LIBRARIES, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        report = check(args.lib)
        code = int(report["counts"]["failed"] > 0)
    except Exception as exc:
        report = {"library": args.lib, "error": type(exc).__name__ + ": " + str(exc),
                  "traceback": traceback.format_exc()}
        code = 1
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "conversions"}))
    return code


if __name__ == "__main__":
    sys.exit(main())
