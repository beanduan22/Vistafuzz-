from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import get_adapter
from .models import CALL_KEYWORD, CALL_POSITIONAL, CALL_POSITIONAL_OR_KEYWORD, ParamSpec, Value

def library_root(api: str) -> str:
    return api.split(".", 1)[0]


def native_dtype(root: str, name: str) -> Any:
    return get_adapter(root).dtype(name)


def to_native_array(root: str, arr: np.ndarray, dtype_name: str | None) -> Any:
    return get_adapter(root).array(arr, dtype_name)


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
