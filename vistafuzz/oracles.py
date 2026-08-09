from __future__ import annotations

import re
from typing import Any

import numpy as np

REJECTION_EXCEPTIONS = (
    "ValueError", "TypeError", "IndexError", "KeyError", "AttributeError",
    "NotImplementedError", "ArithmeticError", "ZeroDivisionError",
    "OverflowError", "UnboundLocalError", "StopIteration", "LookupError",
    "ModuleNotFoundError", "ImportError", "FileNotFoundError", "OSError",
    "PermissionError", "MemoryError", "InvalidArgumentError",
    "UnimplementedError", "FailedPreconditionError", "OutOfRangeError",
    "ResourceExhaustedError", "OutOfMemoryError", "XlaRuntimeError",
    "TracerArrayConversionError", "ConcretizationTypeError", "UFuncTypeError",
    "AxisError", "LinAlgError", "NotFittedError", "ConvergenceWarning",
    "DTypePromotionError", "ComplexWarning",
)

INTERNAL_MARKERS = (
    "internal assert", "internal error", "please report a bug",
    "report a bug to", "check failed", "segmentation fault",
    "core dumped", "unreachable code", "invariant violated",
    "should not happen", "memory corrupt", "double free",
)

RESOURCE_MARKERS = (
    "out of memory", "cannot allocate", "std::bad_alloc", "oom",
    "insufficient memory", "allocation failed", "unable to allocate",
    "resource exhausted", "too large", "maximum allowed dimension",
)
_RESOURCE_RE = re.compile("|".join(re.escape(m) for m in RESOURCE_MARKERS), re.I)


_MARKER_RE = re.compile("|".join(re.escape(m) for m in INTERNAL_MARKERS), re.I)


def is_resource_exhaustion(text: str) -> bool:
    return bool(_RESOURCE_RE.search(text or ""))


SUSPICIOUS_EXCEPTIONS = frozenset({
    "SystemError", "RecursionError", "BufferError",
    "ReferenceError", "UnicodeDecodeError", "UnicodeEncodeError",
    "StopIteration", "GeneratorExit", "SystemExit", "KeyboardInterrupt",
    "FatalError", "InternalError", "Aborted",
})


def classify_exception(exc_type: str, message: str) -> str:
    if is_resource_exhaustion(message):
        return "rejected"
    if _MARKER_RE.search(message or ""):
        return "unexpected"
    if exc_type in REJECTION_EXCEPTIONS:
        return "rejected"
    if exc_type in SUSPICIOUS_EXCEPTIONS:
        return "unexpected"
    if exc_type == "AssertionError":
        return "rejected" if (message or "").strip() else "unexpected"
    return "rejected"


def _iter_numeric(obj: Any, depth: int = 0):
    if depth > 3 or obj is None:
        return
    if isinstance(obj, (bool, np.bool_)):
        return
    if isinstance(obj, (int, float, np.integer, np.floating, complex, np.complexfloating)):
        yield np.asarray(obj)
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fc":
            yield obj
        return
    if isinstance(obj, (list, tuple, set)):
        for item in list(obj)[:8]:
            yield from _iter_numeric(item, depth + 1)
        return
    if isinstance(obj, dict):
        for item in list(obj.values())[:8]:
            yield from _iter_numeric(item, depth + 1)
        return
    for attr in ("detach", "cpu", "numpy", "asnumpy", "to_numpy", "__array__"):
        fn = getattr(obj, attr, None)
        if not callable(fn):
            continue
        try:
            converted = fn()
        except Exception:
            continue
        if isinstance(converted, np.ndarray):
            if converted.dtype.kind in "fc":
                yield converted
            return
        if converted is not obj and depth < 3:
            yield from _iter_numeric(converted, depth + 1)
            return


def inputs_are_finite(natives: list[Any]) -> bool:
    for item in natives:
        for arr in _iter_numeric(item):
            if arr.size and not np.all(np.isfinite(arr)):
                return False
    return True


def check_nan(result: Any) -> tuple[bool, str]:
    for arr in _iter_numeric(result):
        if not arr.size:
            continue
        finite = np.isfinite(arr)
        if np.all(finite):
            continue
        flat = np.asarray(arr).ravel()
        bad = flat[~np.isfinite(flat)]
        n_nan = int(np.sum(np.isnan(flat)))
        n_inf = int(bad.size - n_nan)
        sample = ", ".join(repr(float(v)) for v in bad[:3])
        return True, (f"output contains {n_nan} NaN and {n_inf} inf value(s) "
                      f"(e.g. {sample}) for finite inputs")
    return False, ""


def crash_verdict(exit_code: int, signal_number: int | None,
                  timed_out: bool) -> tuple[str, str] | None:
    if timed_out:
        return "hang", "invocation exceeded the per-case timeout"
    if signal_number:
        return "signal", f"process terminated by signal {signal_number}"
    if exit_code not in (0, None):
        return "exit_code", f"process exited with code {exit_code}"
    return None
