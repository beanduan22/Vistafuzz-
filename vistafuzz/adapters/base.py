from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass

import numpy as np


def resolve_object(path):
    parts = path.split(".")
    for size in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:size]))
            break
        except ModuleNotFoundError as exc:
            if exc.name and not ".".join(parts[:size]).startswith(exc.name):
                raise
    else:
        raise ImportError(path)
    for name in parts[size:]:
        obj = getattr(obj, name)
    return obj


@dataclass(frozen=True)
class Adapter:
    root: str
    array_path: str
    dtype_path: str = "numpy"
    dtype_as_string: bool = False
    accepts_dtype: bool = True

    def documentation(self, obj):
        return inspect.getdoc(obj) or ""

    def signature(self, obj):
        try:
            return inspect.signature(obj)
        except (TypeError, ValueError):
            return None

    def dtype(self, name):
        if self.dtype_as_string:
            return name
        namespace = importlib.import_module(self.dtype_path)
        if self.dtype_path == "numpy":
            return np.dtype(name)
        candidate = getattr(namespace, name, None)
        if candidate is None:
            raise TypeError(f"{self.root} does not support dtype {name}")
        return candidate

    def array(self, value, dtype_name):
        ctor = resolve_object(self.array_path)
        dtype = self.dtype(dtype_name) if dtype_name else None
        array = np.asarray(value, dtype=dtype_name)
        out = ctor(array, dtype=dtype) if self.accepts_dtype else ctor(array)
        if tuple(getattr(out, "shape", array.shape)) != array.shape:
            raise ValueError(f"{self.root} changed the generated shape")
        actual = getattr(out, "dtype", None)
        if actual is not None and dtype_name and dtype_name not in str(actual):
            try:
                matches = np.dtype(actual) == np.dtype(dtype_name)
            except TypeError:
                matches = False
            if not matches:
                raise ValueError(f"{self.root} changed {dtype_name} to {actual}")
        return out
