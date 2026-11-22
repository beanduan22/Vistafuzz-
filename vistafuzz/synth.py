from __future__ import annotations

import math

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
FLOAT_EXTREME = [1e300, -1e300, 1e-300, -1e-300, 5e-324,
                 1.7976931348623157e308, 2.2250738585072014e-308, 1e38, -1e38]
FLOAT_NONFINITE = [float("nan"), float("inf"), float("-inf")]
FLOAT_SENSITIVE = [np.pi, np.pi / 2, -np.pi, 2 * np.pi, 709.0, -709.0,
                   710.0, 713.0, 88.7, -88.7, 1e-16,
                   0.9999999999999999, 1.0000000000000002]
INT_BOUNDARY = [0, 1, -1, 2, 127, 128, -128, 255, 32767, -32768]
INT_EXTREME = [2147483647, -2147483648, 9223372036854775807,
               -9223372036854775808]


def numeric_bounds(dtype, low=None, high=None, exclusive_min=False,
                   exclusive_max=False):
    dt = np.dtype(dtype)
    if dt.kind in "iu":
        info = np.iinfo(dt)
        lo = info.min if low is None else max(info.min, math.ceil(low))
        hi = info.max if high is None else min(info.max, math.floor(high))
        if low is not None and exclusive_min and lo == low:
            lo += 1
        if high is not None and exclusive_max and hi == high:
            hi -= 1
    elif dt.kind in "fc":
        real_dt = np.empty((), dtype=dt).real.dtype
        info = np.finfo(real_dt)
        lo = -float(info.max) if low is None else max(-float(info.max), low)
        hi = float(info.max) if high is None else min(float(info.max), high)
        with np.errstate(over="ignore", invalid="ignore"):
            lo_cast = np.asarray(lo, dtype=real_dt)[()]
            hi_cast = np.asarray(hi, dtype=real_dt)[()]
            if low is not None and (lo_cast < low or exclusive_min and lo_cast == low):
                lo_cast = np.nextafter(lo_cast, real_dt.type(np.inf))
            if high is not None and (hi_cast > high or exclusive_max and hi_cast == high):
                hi_cast = np.nextafter(hi_cast, real_dt.type(-np.inf))
        lo, hi = float(lo_cast), float(hi_cast)
    else:
        raise ValueError(f"unsupported numerical dtype: {dtype}")
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        raise ValueError(f"no {dtype} value satisfies the numerical bounds")
    return lo, hi


def _pool(pattern, is_int):
    if is_int:
        return {BOUNDARY: INT_BOUNDARY, EXTREME: INT_EXTREME,
                SENSITIVE: INT_BOUNDARY, NONFINITE: INT_EXTREME}.get(pattern, [])
    return {BOUNDARY: FLOAT_BOUNDARY, EXTREME: FLOAT_EXTREME,
            NONFINITE: FLOAT_NONFINITE, SENSITIVE: FLOAT_SENSITIVE}.get(pattern, [])


def build_array(pattern, dtype, shape, seed, low=None, high=None,
                params=None, allow_nonfinite=True, exclusive_min=False,
                exclusive_max=False):
    params = params or {}
    dt = np.dtype(dtype)
    shape = tuple(int(d) for d in (() if shape is None else shape))
    rng = np.random.default_rng(seed)
    size = math.prod(shape)
    if dt.kind == "b":
        return rng.integers(0, 2, size=size).astype(dt).reshape(shape)
    lo, hi = numeric_bounds(dt, low, high, exclusive_min, exclusive_max)
    is_int = dt.kind in "iu"
    real_dt = np.empty((), dtype=dt).real.dtype
    if pattern == NONFINITE and (is_int or not allow_nonfinite):
        pattern = EXTREME
    pool = list(_pool(pattern, is_int))
    if pattern == BOUNDARY:
        if low is not None:
            pool.append(lo)
        if high is not None:
            pool.append(hi)
    if pattern == EXTREME:
        pool.extend([lo, hi])
    if pool:
        if pattern == NONFINITE:
            values = [pool[int(i)] for i in rng.integers(0, len(pool), size=size)]
        else:
            pool = [min(hi, max(lo, v)) for v in pool if math.isfinite(v)]
            values = [pool[int(i)] for i in rng.integers(0, len(pool), size=size)]
        base = np.asarray(values, dtype=dt)
    else:
        a = max(lo, -4 if low is None else lo)
        b = min(hi, 4 if high is None else hi)
        if a > b:
            a = b = lo if low is not None else hi
        if is_int:
            base = np.asarray([rng.integers(int(a), int(b), endpoint=True, dtype=dt)
                               for _ in range(size)], dtype=dt)
        else:
            weights = rng.random(size=size)
            base = np.asarray((1 - weights) * a + weights * b, dtype=real_dt)
            if dt.kind == "c":
                imag = rng.uniform(-1, 1, size=size).astype(real_dt)
                base = base.astype(dt) + 1j * imag
            if pattern == NOISE:
                base = base + rng.normal(0, params.get("scale", 0.1), size=size)
            elif pattern == MASK:
                base[rng.random(size=size) < params.get("ratio", 0.3)] = params.get("fill", 0)
            elif pattern == DIVISION:
                factor = float(params.get("factor", 2.0))
                if not math.isfinite(factor) or factor == 0:
                    raise ValueError("division requires a finite, non-zero factor")
                base = base / factor
    if pattern != NONFINITE:
        if dt.kind == "c":
            base = (np.clip(np.real(base), lo, hi)
                    + 1j * np.clip(np.imag(base), lo, hi)).astype(dt)
        else:
            base = np.clip(base, lo, hi).astype(dt)
        if not np.all(np.isfinite(base)):
            raise ValueError(f"finite generation overflowed {dtype}")
    return base.reshape(shape)


def build_scalar(pattern, dtype, seed, low=None, high=None,
                 allow_nonfinite=True, exclusive_min=False, exclusive_max=False):
    return build_array(pattern, dtype, (), seed, low, high,
                       allow_nonfinite=allow_nonfinite,
                       exclusive_min=exclusive_min, exclusive_max=exclusive_max).item()


def build_value(recipe):
    form = recipe.get("form", "array")
    if form == "const":
        return recipe.get("value")
    if form == "literal_array":
        return np.asarray(recipe["values"], dtype=recipe["dtype"])
    if form in ("array", "scalar"):
        value = build_array(
            recipe.get("pattern", ORDINARY), recipe.get("dtype", "float64"),
            recipe.get("shape", ()) if form == "array" else (),
            int(recipe.get("seed", 0)), recipe.get("low"), recipe.get("high"),
            recipe.get("params"), recipe.get("allow_nonfinite", True),
            recipe.get("exclusive_min", False), recipe.get("exclusive_max", False))
        return value.item() if form == "scalar" else value
    if form == "sequence":
        items = [build_value(r) for r in recipe.get("items", [])]
        return tuple(items) if recipe.get("as_tuple") else items
    if form == "dict":
        return {k: build_value(v) for k, v in recipe.get("items", {}).items()}
    raise ValueError(f"unknown recipe form: {form!r}")
