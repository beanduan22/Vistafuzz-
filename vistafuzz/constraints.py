from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import relationships, synth
from .models import INT_NAMES, ParamSpec, Value

FLOAT_DTYPES = ("float64", "float32", "float16", "complex64", "complex128")
INT_DTYPES = ("int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8")
DEFAULT_SHAPES = ((), (1,), (3,), (2, 3), (1, 4), (2, 2, 2), (2, 3, 4))
MAX_ELEMENTS = 262144
MAX_RANK = 32
VALUE_PATTERNS = (synth.ORDINARY, synth.BOUNDARY, synth.SENSITIVE,
                  synth.EXTREME, synth.NONFINITE)
IMAGE_PATTERNS = (synth.NOISE, synth.MASK, synth.DIVISION)
_IMAGE_HINTS = ("image", "img", "src", "dst", "frame", "picture", "photo",
                "mask", "canvas", "pixels", "rgb", "bgr", "gray")
NEVER_SYNTHESIZED = frozenset({"out", "output", "device", "stream", "session",
                               "graph", "ctx", "context", "handle", "sess", "where"})
UNKNOWN_KINDS = ("array", "int", "number", "str", "bool")
DATA_LIKE_NAMES = frozenset({"data", "values", "value", "x", "a", "arr", "array",
                             "input", "obj", "object", "seq", "iterable", "numerator",
                             "buffer", "content", "items", "elements"})


def keeps_default(spec):
    return not spec.required and spec.name.lower() in NEVER_SYNTHESIZED


@dataclass
class Envelope:
    kinds: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    shapes: list[tuple[int, ...]] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    enum_values: list[Any] = field(default_factory=list)
    low: int | float | None = None
    high: int | float | None = None
    exclusive_min: bool = False
    exclusive_max: bool = False
    allow_nonfinite: bool = True
    image_like: bool = False
    omittable: bool = False

    def is_empty(self):
        if not self.kinds:
            return True
        if self.kinds[0] == "array":
            return not self.dtypes or not self.shapes
        return False


def _bounds(spec):
    con = spec.constraints or {}
    low, high = con.get("min"), con.get("max")
    for value in (low, high):
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, (int, float))
                                  or not math.isfinite(value)):
            raise ValueError(f"invalid numerical bound on {spec.name}")
    if low is not None and high is not None and low > high:
        raise ValueError(f"inverted numerical bounds on {spec.name}")
    allow = con.get("allow_nonfinite") is not False and low is None and high is None
    return low, high, allow


def _shape_candidates(spec):
    shape = spec.shape or {}
    dims = shape.get("dims")
    if dims is not None:
        if not isinstance(dims, (list, tuple)) or any(
                not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in dims):
            return []
        candidate = tuple(dims)
        return [candidate] if len(candidate) <= MAX_RANK and math.prod(candidate) <= MAX_ELEMENTS else []
    low = shape.get("rank_min", 0)
    high = shape.get("rank_max", max(3, low))
    if not isinstance(low, int) or not isinstance(high, int) or low < 0 or high < low:
        return []
    if low > MAX_RANK:
        return []
    high = min(high, MAX_RANK)
    picks = [s for s in DEFAULT_SHAPES if low <= len(s) <= high]
    for rank in range(low, high + 1):
        if not any(len(s) == rank for s in picks):
            picks.append((1,) * rank)
    return picks


def envelope_for(spec):
    low, high, allow = _bounds(spec)
    con = spec.constraints or {}
    env = Envelope(low=low, high=high, allow_nonfinite=allow,
                   exclusive_min=bool(con.get("exclusive_min")),
                   exclusive_max=bool(con.get("exclusive_max")),
                   omittable=not spec.required and spec.has_default)
    kind = spec.kind if spec.kind != "unknown" else _guess_kind(spec)
    env.kinds = [kind] if kind != "unknown" else list(UNKNOWN_KINDS)
    kind = env.kinds[0]
    env.enum_values = list(spec.enum_values or [])
    if kind in ("array", "number", "int", "axis", "dtype"):
        defaults = ["int64"] if kind in ("int", "axis") else list(FLOAT_DTYPES[:2])
        env.dtypes = [d for d in (spec.dtype_candidates or defaults)
                      if d in FLOAT_DTYPES + INT_DTYPES + ("bool",)]
        if kind in ("int", "axis"):
            env.dtypes = [d for d in env.dtypes if d in INT_DTYPES]
        env.patterns = [p for p in VALUE_PATTERNS
                        if p != synth.NONFINITE or allow and kind not in ("int", "axis")]
        env.shapes = _shape_candidates(spec) if kind == "array" else [()]
        if kind == "array":
            env.image_like = (any(h in spec.name.lower() for h in _IMAGE_HINTS)
                              or any(h in spec.doc.lower()[:120] for h in _IMAGE_HINTS))
            if env.image_like:
                env.patterns = list(IMAGE_PATTERNS) + env.patterns
    elif kind == "bool":
        env.enum_values = env.enum_values or [True, False]
    elif kind in ("shape", "sequence"):
        a, b = con.get("len_min", 1), con.get("len_max", max(3, con.get("len_min", 1)))
        if not isinstance(a, int) or not isinstance(b, int) or a < 0 or b < a or a > 1024:
            env.kinds = []
        else:
            env.shapes = [(n,) for n in sorted({a, min(b, 1024), (a + min(b, 1024)) // 2})]
        env.dtypes = [d for d in (spec.dtype_candidates or ["int64"])
                      if d in FLOAT_DTYPES + INT_DTYPES]
        env.patterns = [synth.ORDINARY, synth.BOUNDARY]
    elif kind in ("callable", "device", "receiver"):
        env.kinds = []
    return env


def _guess_kind(spec):
    default = spec.default
    for cls, name in ((bool, "bool"), (str, "str"), (int, "int"), (float, "number"),
                      (dict, "dict"), (list, "sequence"), (tuple, "sequence")):
        if isinstance(default, cls):
            return name
    if spec.name.lower() in ("axis", "dim", "axes"):
        return "axis"
    if spec.name.lower() in ("dtype", "type"):
        return "dtype"
    return "int" if spec.name.lower() in INT_NAMES else "unknown"


def initialize(specs, seed=0, resolve_relationships=True):
    from .generation import make_value

    args = {}
    ordered = relationships.dependency_order(specs) if resolve_relationships else specs
    required_sources = relationships.required_sources(specs) if resolve_relationships else set()
    for idx, spec in enumerate(ordered):
        if spec.call_kind in ("var_positional", "var_keyword"):
            continue
        if not spec.required and spec.has_default and spec.name not in required_sources:
            args[spec.name] = Value(kind="default", payload=spec.default, omitted=True,
                                    recipe={"form": "const", "value": _safe_const(spec.default)})
            continue
        env = envelope_for(spec)
        value = make_value(spec, env, args, seed=seed * 1009 + idx,
                           dtype=env.dtypes[0] if env.dtypes else None,
                           shape=env.shapes[0] if env.shapes else None,
                           pattern=synth.ORDINARY, resolve_relationships=resolve_relationships)
        if value is None:
            return {}
        args[spec.name] = value
    return args


def _safe_const(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(
            isinstance(v, (bool, int, float, str, type(None))) for v in value):
        return list(value)
    return repr(value)
