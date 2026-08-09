from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import synth
from .models import ParamSpec, Value

FLOAT_DTYPES = ("float64", "float32", "float16")
INT_DTYPES = ("int64", "int32", "int16", "uint8")
DEFAULT_SHAPES = ((), (1,), (3,), (2, 3), (1, 4), (2, 2, 2), (2, 3, 4))
VALUE_PATTERNS = (synth.ORDINARY, synth.BOUNDARY, synth.SENSITIVE,
                  synth.EXTREME, synth.NONFINITE)
IMAGE_PATTERNS = (synth.NOISE, synth.MASK, synth.DIVISION)

_IMAGE_HINTS = ("image", "img", "src", "dst", "frame", "picture", "photo",
                "mask", "canvas", "pixels", "rgb", "bgr", "gray")

NEVER_SYNTHESIZED = frozenset({"out", "output", "device", "stream", "session",
                               "graph", "ctx", "context", "handle", "sess", "where"})


def keeps_default(spec: ParamSpec) -> bool:
    return not spec.required and spec.name.lower() in NEVER_SYNTHESIZED


@dataclass
class Envelope:

    kinds: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    shapes: list[tuple[int, ...]] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    enum_values: list[Any] = field(default_factory=list)
    low: float | None = None
    high: float | None = None
    allow_nonfinite: bool = True
    image_like: bool = False
    omittable: bool = False

    def is_empty(self) -> bool:
        return not (self.kinds or self.enum_values)


def _bounds(spec: ParamSpec) -> tuple[float | None, float | None, bool]:
    con = spec.constraints or {}
    low = con.get("min")
    high = con.get("max")
    low = float(low) if isinstance(low, (int, float)) and not isinstance(low, bool) else None
    high = float(high) if isinstance(high, (int, float)) and not isinstance(high, bool) else None
    if low is not None and con.get("exclusive_min"):
        low = math.nextafter(low, math.inf)
    if high is not None and con.get("exclusive_max"):
        high = math.nextafter(high, -math.inf)
    allow_nonfinite = con.get("allow_nonfinite")
    allow = True if allow_nonfinite is None else bool(allow_nonfinite)
    if low is not None or high is not None:
        allow = False
    return low, high, allow


def _shape_candidates(spec: ParamSpec) -> list[tuple[int, ...]]:
    shape = spec.shape or {}
    dims = shape.get("dims")
    if isinstance(dims, (list, tuple)) and dims and all(
            isinstance(d, int) and 0 < d <= 64 for d in dims):
        return [tuple(int(d) for d in dims)]
    rank_min = shape.get("rank_min")
    rank_max = shape.get("rank_max")
    rank_min = int(rank_min) if isinstance(rank_min, int) else 0
    rank_max = int(rank_max) if isinstance(rank_max, int) else 3
    rank_min, rank_max = max(0, min(rank_min, 4)), max(0, min(rank_max, 4))
    if rank_max < rank_min:
        rank_min, rank_max = rank_max, rank_min
    picks = [s for s in DEFAULT_SHAPES if rank_min <= len(s) <= rank_max]
    return picks or [(2, 3)]


def envelope_for(spec: ParamSpec) -> Envelope:
    low, high, allow_nonfinite = _bounds(spec)
    env = Envelope(low=low, high=high, allow_nonfinite=allow_nonfinite,
                   omittable=not spec.required and spec.has_default)
    kind = spec.kind if spec.kind != "unknown" else _guess_kind(spec)
    env.kinds = [kind]
    env.enum_values = list(spec.enum_values or [])

    if kind == "array":
        env.dtypes = [d for d in (spec.dtype_candidates or FLOAT_DTYPES[:2])
                      if d in FLOAT_DTYPES + INT_DTYPES + ("bool",)] or ["float64"]
        env.shapes = _shape_candidates(spec)
        env.image_like = any(h in spec.name.lower() for h in _IMAGE_HINTS) or \
            any(h in (spec.doc or "").lower()[:120] for h in _IMAGE_HINTS)
        env.patterns = list(VALUE_PATTERNS)
        if env.image_like:
            env.patterns = list(IMAGE_PATTERNS) + list(VALUE_PATTERNS)
        if not allow_nonfinite:
            env.patterns = [p for p in env.patterns if p != synth.NONFINITE]
    elif kind in ("number", "int", "axis"):
        env.dtypes = ["float64"] if kind == "number" else ["int64"]
        if spec.dtype_candidates:
            env.dtypes = [d for d in spec.dtype_candidates
                          if d in FLOAT_DTYPES + INT_DTYPES] or env.dtypes
        env.shapes = [()]
        env.patterns = [p for p in VALUE_PATTERNS
                        if allow_nonfinite or p != synth.NONFINITE]
        if kind in ("int", "axis"):
            env.patterns = [p for p in env.patterns if p != synth.NONFINITE]
    elif kind == "bool":
        env.enum_values = env.enum_values or [True, False]
    elif kind == "str":
        env.enum_values = env.enum_values or []
    elif kind == "dtype":
        env.dtypes = [d for d in (spec.dtype_candidates or FLOAT_DTYPES[:2])] or ["float64"]
    elif kind in ("shape", "sequence"):
        env.dtypes = ["int64"]
        env.shapes = [(1,), (2,), (3,)]
        env.patterns = [synth.ORDINARY, synth.BOUNDARY]
    return env


def _guess_kind(spec: ParamSpec) -> str:
    default = spec.default
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, str):
        return "str"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "number"
    name = spec.name.lower()
    if name in ("axis", "dim", "axes"):
        return "axis"
    if name in ("dtype", "type"):
        return "dtype"
    return "array"


def initialize(specs: list[ParamSpec], seed: int = 0) -> dict[str, Value]:
    from .generation import make_value

    args: dict[str, Value] = {}
    for idx, spec in enumerate(specs):
        if spec.call_kind in ("var_positional", "var_keyword"):
            continue
        env = envelope_for(spec)
        if not spec.required and spec.has_default:
            args[spec.name] = Value(kind="default", payload=spec.default,
                                    recipe={"form": "const", "value": _safe_const(spec.default)},
                                    strategy="init", omitted=True)
            continue
        value = make_value(spec, env, args, seed=seed * 1009 + idx,
                           dtype=env.dtypes[0] if env.dtypes else None,
                           shape=env.shapes[0] if env.shapes else None,
                           pattern=synth.ORDINARY)
        if value is None:
            return {}
        args[spec.name] = value
    return args


def _safe_const(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(
            isinstance(v, (bool, int, float, str, type(None))) for v in value):
        return list(value)
    return repr(value)
