from __future__ import annotations

import numpy as np

ORDINARY = "ordinary"
BOUNDARY = "boundary"
EXTREME = "extreme"
NONFINITE = "nonfinite"
SENSITIVE = "sensitive"
NOISE = "noise"
MASK = "mask"
DIVISION = "division"

FLOAT_BOUNDARY = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 1e-8, -1e-8]
FLOAT_EXTREME = [1e300, -1e300, 1e-300, -1e-300, 5e-324, 1.7976931348623157e308,
                 2.2250738585072014e-308, 1e38, -1e38]
FLOAT_NONFINITE = [float("nan"), float("inf"), float("-inf")]
FLOAT_SENSITIVE = [np.pi, np.pi / 2, -np.pi, 2 * np.pi, 709.0, -709.0, 710.0,
                   88.7, -88.7, 1e-16, 0.9999999999999999, 1.0000000000000002]
INT_BOUNDARY = [0, 1, -1, 2, 127, 128, -128, 255, 32767, -32768]
INT_EXTREME = [2147483647, -2147483648, 9223372036854775807, -9223372036854775808]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _pool(pattern: str, is_int: bool) -> list:
    if is_int:
        return {BOUNDARY: INT_BOUNDARY, EXTREME: INT_EXTREME,
                SENSITIVE: INT_BOUNDARY, NONFINITE: INT_EXTREME}.get(pattern, [])
    return {BOUNDARY: FLOAT_BOUNDARY, EXTREME: FLOAT_EXTREME,
            NONFINITE: FLOAT_NONFINITE, SENSITIVE: FLOAT_SENSITIVE}.get(pattern, [])


def build_scalar(pattern: str, dtype: str, seed: int,
                 low: float | None = None, high: float | None = None):
    rng = _rng(seed)
    is_int = dtype.startswith(("int", "uint"))
    if dtype == "bool":
        return bool(rng.integers(0, 2))
    pool = _pool(pattern, is_int)
    if pool:
        value = pool[int(rng.integers(0, len(pool)))]
    elif is_int:
        lo = int(low) if low is not None else -16
        hi = int(high) if high is not None else 16
        value = int(rng.integers(lo, max(lo + 1, hi + 1)))
    else:
        lo = -8.0 if low is None else float(low)
        hi = 8.0 if high is None else float(high)
        value = float(rng.uniform(lo, hi))
    if low is not None and value < low:
        value = low
    if high is not None and value > high:
        value = high
    return np.array(value).astype(dtype).item() if dtype != "bool" else bool(value)


def build_array(pattern: str, dtype: str, shape, seed: int,
                low: float | None = None, high: float | None = None,
                params: dict | None = None) -> np.ndarray:
    params = params or {}
    shape = tuple(int(d) for d in (shape or ()))
    rng = _rng(seed)
    is_int = dtype.startswith(("int", "uint"))
    size = int(np.prod(shape)) if shape else 1

    if dtype == "bool":
        base = rng.integers(0, 2, size=size).astype("bool")
        return base.reshape(shape)
    if is_int:
        lo = int(low) if low is not None else 0
        hi = int(high) if high is not None else 8
        base = rng.integers(lo, max(lo + 1, hi + 1), size=size).astype(dtype)
    else:
        lo = -4.0 if low is None else float(low)
        hi = 4.0 if high is None else float(high)
        base = rng.uniform(lo, hi, size=size).astype(dtype)

    pool = _pool(pattern, is_int)
    if pool:
        picks = rng.integers(0, len(pool), size=size)
        base = np.array([pool[i] for i in picks], dtype=dtype)
    elif pattern == NOISE:
        scale = float(params.get("scale", 0.1))
        base = base + rng.normal(0.0, scale, size=size).astype(base.dtype)
    elif pattern == MASK:
        fill = params.get("fill", 0)
        ratio = float(params.get("ratio", 0.3))
        mask = rng.random(size=size) < ratio
        base = base.copy()
        base[mask] = np.array(fill).astype(base.dtype)
    elif pattern == DIVISION:
        factor = float(params.get("factor", 2.0)) or 1.0
        base = (base / factor).astype(base.dtype)

    if low is not None:
        base = np.maximum(base, np.array(low).astype(base.dtype))
    if high is not None:
        base = np.minimum(base, np.array(high).astype(base.dtype))
    return base.reshape(shape)


def build_value(recipe: dict):
    form = recipe.get("form", "array")
    if form == "const":
        return recipe.get("value")
    if form == "scalar":
        return build_scalar(recipe.get("pattern", ORDINARY), recipe.get("dtype", "float64"),
                            int(recipe.get("seed", 0)), recipe.get("low"), recipe.get("high"))
    if form == "array":
        return build_array(recipe.get("pattern", ORDINARY), recipe.get("dtype", "float64"),
                           recipe.get("shape", ()), int(recipe.get("seed", 0)),
                           recipe.get("low"), recipe.get("high"), recipe.get("params"))
    if form == "sequence":
        items = [build_value(r) for r in recipe.get("items", [])]
        return tuple(items) if recipe.get("as_tuple") else items
    raise ValueError(f"unknown recipe form: {form!r}")
