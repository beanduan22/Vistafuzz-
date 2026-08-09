from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .models import CALL_KEYWORD, CALL_POSITIONAL, CALL_POSITIONAL_OR_KEYWORD, ParamSpec, Value

_ARRAY_CTORS = ("asarray", "array", "tensor", "as_tensor", "to_tensor",
                "constant", "Tensor", "from_numpy")
_DTYPE_ALIASES = {"float64": ("float64", "double", "float_"),
                  "float32": ("float32", "float", "single"),
                  "float16": ("float16", "half"),
                  "int64": ("int64", "long", "int_"),
                  "int32": ("int32", "int"),
                  "int16": ("int16", "short"),
                  "uint8": ("uint8", "ubyte"),
                  "bool": ("bool_", "bool", "boolean")}


def library_root(api: str) -> str:
    return api.split(".", 1)[0]


def _module(root: str):
    try:
        return importlib.import_module(root)
    except Exception:
        return None


def native_dtype(root: str, name: str) -> Any:
    mod = _module(root)
    for alias in _DTYPE_ALIASES.get(name, (name,)):
        candidate = getattr(mod, alias, None) if mod is not None else None
        if candidate is not None and not callable(candidate) or _is_dtype_like(candidate):
            return candidate
    return getattr(np, name, None)


def _is_dtype_like(obj: Any) -> bool:
    if obj is None:
        return False
    text = f"{type(obj).__module__}.{type(obj).__name__}".lower()
    return "dtype" in text or "type" in type(obj).__name__.lower()


def to_native_array(root: str, arr: np.ndarray, dtype_name: str | None) -> Any:
    mod = _module(root)
    if mod is None or root == "numpy":
        return arr
    dtype = native_dtype(root, dtype_name) if dtype_name else None
    for name in _ARRAY_CTORS:
        ctor = getattr(mod, name, None)
        if not callable(ctor):
            continue
        for kwargs in ([{"dtype": dtype}] if dtype is not None else []) + [{}]:
            try:
                out = ctor(arr, **kwargs)
            except Exception:
                continue
            if out is not None:
                return out
    return arr


def materialize_value(root: str, value: Value) -> Any:
    if value.kind == "array":
        payload = value.payload
        if not isinstance(payload, np.ndarray):
            payload = np.asarray(payload)
        return to_native_array(root, payload, value.dtype)
    if value.kind == "dtype":
        native = native_dtype(root, str(value.payload))
        return native if native is not None else value.payload
    if value.kind == "receiver":
        return value.payload
    if value.kind in ("shape",):
        return tuple(value.payload)
    if value.kind in ("sequence",):
        return list(value.payload)
    return value.payload


def materialize(api: str, specs: list[ParamSpec],
                args: dict[str, Value]) -> tuple[list[Any], dict[str, Any]]:
    root = library_root(api)
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    ordered = sorted(specs, key=lambda s: (s.position if s.position is not None else 1 << 20))

    keyword_only_from_here = False
    for spec in ordered:
        value = args.get(spec.name)
        if value is None or value.omitted:
            if spec.call_kind in (CALL_POSITIONAL, CALL_POSITIONAL_OR_KEYWORD):
                keyword_only_from_here = True
            continue
        native = materialize_value(root, value)
        if spec.call_kind == CALL_KEYWORD or keyword_only_from_here:
            if spec.call_kind == CALL_POSITIONAL:
                continue
            keyword[spec.name] = native
        else:
            positional.append(native)
    return positional, keyword
