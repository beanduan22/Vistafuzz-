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
class Runtime:
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
        return self.check_array(out, array, dtype_name)

    def check_array(self, out, array, dtype_name):
        shape = getattr(out, "shape", None)
        if shape is None:
            raise TypeError(f"{self.root} did not expose the materialized shape")
        if tuple(shape) != array.shape:
            raise ValueError(f"{self.root} changed the generated shape")
        actual = getattr(out, "dtype", None)
        if actual is None:
            raise TypeError(f"{self.root} did not expose the materialized dtype")
        expected = np.dtype(dtype_name if dtype_name is not None else array.dtype)
        named_dtype = getattr(actual, "name", None)
        if isinstance(named_dtype, str):
            named_dtype = named_dtype.lower()
        candidates = (getattr(actual, "as_numpy_dtype", None), named_dtype,
                      actual, str(actual).rsplit(".", 1)[-1].lower())
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                observed = np.dtype(candidate)
            except (TypeError, ValueError):
                continue
            if observed != expected:
                raise ValueError(f"{self.root} changed {expected.name} to {actual}")
            return out
        raise TypeError(f"{self.root} exposed an unrecognized dtype: {actual}")
