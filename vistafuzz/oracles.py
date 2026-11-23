from __future__ import annotations

import ast
import doctest
import re
from functools import lru_cache
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


def classify_exception(exc_type: str, message: str, constraints_satisfied: bool = False) -> str:
    if is_resource_exhaustion(message):
        return "rejected"
    if constraints_satisfied and (exc_type == "AssertionError" or "(-215:Assertion failed)" in message):
        return "unexpected"
    if _MARKER_RE.search(message or ""):
        return "unexpected"
    if exc_type in REJECTION_EXCEPTIONS:
        return "rejected"
    if exc_type in SUSPICIOUS_EXCEPTIONS:
        return "unexpected"
    if exc_type == "AssertionError":
        return "rejected" if (message or "").strip() else "unexpected"
    return "rejected"


def _iter_numeric(obj: Any, depth: int = 0, include_bool: bool = False, _seen=None):
    if obj is None or depth > 32:
        return
    if _seen is None:
        _seen = {}
    if id(obj) in _seen:
        return
    _seen[id(obj)] = obj
    if isinstance(obj, (bool, np.bool_)):
        if include_bool:
            yield np.asarray(obj)
        return
    if isinstance(obj, (int, float, np.number, complex)):
        yield np.asarray(obj)
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("biufc" if include_bool else "iufc"):
            yield obj
        return
    if isinstance(obj, (list, tuple, set, dict)):
        values = obj.values() if isinstance(obj, dict) else obj
        for item in values:
            yield from _iter_numeric(item, depth + 1, include_bool, _seen)
        return
    for attr in ("numpy", "asnumpy", "to_numpy", "array", "__array__", "cpu", "detach"):
        try:
            fn = getattr(obj, attr, None)
            converted = fn() if callable(fn) else fn
        except Exception:
            continue
        if converted is None or converted is obj:
            continue
        arrays = list(_iter_numeric(converted, depth + 1, include_bool, _seen))
        if arrays:
            yield from arrays
            return


def inputs_are_finite(natives: list[Any]) -> bool:
    return all(np.all(np.isfinite(arr)) for item in natives for arr in _iter_numeric(item))


def _nan_propagates(doc):
    for sentence in re.split(r"[\n.;]", doc.lower()):
        if re.search(r"\b(?:not|never|ignore|ignores|ignored)\b", sentence):
            continue
        if re.search(r"nan.{0,100}propagat|propagat.{0,100}nan", sentence):
            return True
        if re.search(r"(?:return|output|result).{0,40}nan.{0,50}(?:input|argument).{0,30}nan", sentence):
            return True
        if re.search(r"(?:input|argument).{0,30}nan.{0,50}(?:return|output|result).{0,30}nan", sentence):
            return True
    return False


def _matching_mask(output, inputs, nan_only=False):
    covered = np.zeros(output.shape, dtype=bool)
    for source in inputs:
        try:
            source = np.broadcast_to(source, output.shape)
        except ValueError:
            continue
        matching = np.isnan(output) & np.isnan(source)
        if not nan_only and output.dtype.kind != "c" and source.dtype.kind != "c":
            matching |= np.isposinf(output) & np.isposinf(source)
            matching |= np.isneginf(output) & np.isneginf(source)
        covered |= matching
    return covered


def _example_literal(node, variables):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex, bool)):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_example_literal(item, variables) for item in node.elts]
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        constants = {"inf": np.inf, "nan": np.nan, "pi": np.pi, "e": np.e}
        if node.id in constants:
            return constants[node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id in ("np", "numpy", "math", "jnp"):
            return _example_literal(ast.Name(id=node.attr), {})
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _example_literal(node.operand, variables)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _example_literal(node.left, variables)
        right = _example_literal(node.right, variables)
        if not isinstance(left, (int, float, complex)) or not isinstance(right, (int, float, complex)):
            raise ValueError("non-scalar example arithmetic")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow) and abs(right) <= 1024 and abs(left) <= 1e308:
            return left ** right
    if isinstance(node, ast.Call):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        allowed = {"array", "asarray", "tensor", "constant"}
        if name in allowed and len(node.args) == 1 and not node.keywords:
            return np.asarray(_example_literal(node.args[0], variables))
    raise ValueError("example requires execution")


@lru_cache(maxsize=128)
def _documented_nonfinite_cases(doc, api):
    if not doc or not api:
        return ()
    name = api.rsplit(".", 1)[-1]
    elementwise = bool(re.search(r"element[- ]?wise|point[- ]?wise", doc, re.I))
    variables, cases = {}, []
    try:
        examples = doctest.DocTestParser().get_examples(doc)
    except ValueError:
        return ()
    for example in examples[:64]:
        try:
            statements = ast.parse(example.source).body
            for statement in statements:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target = statement.targets[0]
                    if isinstance(target, ast.Name):
                        variables.pop(target.id, None)
                        try:
                            variables[target.id] = _example_literal(statement.value, variables)
                        except (ValueError, TypeError, ArithmeticError):
                            pass
            if len(statements) != 1 or not isinstance(statements[0], ast.Expr):
                continue
            call = statements[0].value
            if not isinstance(call, ast.Call) or call.keywords or not call.args:
                continue
            called = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
            if called != name:
                continue
            inputs = [np.asarray(_example_literal(arg, variables)) for arg in call.args]
            expected = np.asarray(_example_literal(ast.parse(example.want.strip(), mode="eval").body, {}))
            if any(value.dtype.kind not in "biufc" for value in [*inputs, expected]):
                continue
            if not elementwise and any(value.ndim for value in [*inputs, expected]):
                continue
            broadcast = [np.broadcast_to(value, expected.shape) for value in inputs]
            if expected.size > 4096:
                continue
            for index in np.flatnonzero(~np.isfinite(expected))[:64]:
                arguments = tuple(value.flat[index].item() for value in broadcast)
                cases.append((arguments, expected.flat[index].item(), elementwise))
        except (SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError):
            continue
    return tuple(cases)


def _documented_masks(output, inputs, doc, api):
    allowed = np.zeros(output.shape, dtype=bool)
    missing = np.zeros(output.shape, dtype=bool)
    for arguments, expected, elementwise in _documented_nonfinite_cases(doc, api):
        if len(arguments) != len(inputs) or not elementwise and any(a.ndim for a in inputs):
            continue
        matches = np.ones(output.shape, dtype=bool)
        for source, value in zip(inputs, arguments):
            try:
                source = np.broadcast_to(source, output.shape)
            except ValueError:
                matches[...] = False
                break
            matches &= (source == value) | (np.isnan(source) & np.isnan(value))
        if np.isnan(expected):
            correct = np.isnan(output)
        else:
            correct = output == expected
        allowed |= matches & correct
        missing |= matches & ~correct
    return allowed, missing & ~allowed


def check_nan(result: Any, inputs=None, doc: str = "", api: str = "") -> tuple[bool, str]:
    sources = [a for value in (inputs or []) for a in _iter_numeric(value)]
    outputs = list(_iter_numeric(result, include_bool=True))
    expects_nan = _nan_propagates(doc) and any(np.any(np.isnan(a)) for a in sources)
    for arr in outputs:
        if not arr.size:
            continue
        documented, missing = _documented_masks(arr, sources, doc, api)
        if np.any(missing):
            return True, "output contradicts a documented non-finite input/output example"
        if expects_nan:
            for source in sources:
                try:
                    expected = np.broadcast_to(np.isnan(source), arr.shape)
                except ValueError:
                    continue
                if np.any(expected & ~np.isnan(arr)):
                    return True, "documented NaN propagation is absent from the output"
        bad_mask = ~np.isfinite(arr)
        if not np.any(bad_mask):
            continue
        unexpected = bad_mask & ~_matching_mask(arr, sources) & ~documented
        if np.any(unexpected):
            flat = np.asarray(arr)[unexpected]
            n_nan = int(np.count_nonzero(np.isnan(flat)))
            n_inf = int(flat.size - n_nan)
            sample = ", ".join(repr(v.item()) for v in flat[:3])
            return True, (f"output contains {n_nan} NaN and {n_inf} infinity value(s) "
                          f"not explained by input propagation (e.g. {sample}); "
                          "check the documented numerical domain")
    if expects_nan and outputs and all(np.all(np.isfinite(a)) for a in outputs):
        return True, "documented NaN propagation produced a finite or Boolean output"
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
