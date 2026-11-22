from __future__ import annotations

import random

import numpy as np

from . import relationships, synth
from .constraints import Envelope, MAX_ELEMENTS, MAX_RANK, envelope_for, keeps_default
from .models import ParamSpec, Value

STRATEGIES = ("type", "size", "value")
VARIANTS = {
    "full": frozenset(),
    "noT": frozenset({"type"}),
    "noS": frozenset({"size"}),
    "noRand": frozenset({"ordinary", "noise"}),
    "noBound": frozenset({"boundary", "sensitive", "extreme"}),
    "noNF": frozenset({"nonfinite"}),
    "noRel": frozenset({"relationships"}),
    "noVal": frozenset({"validation"}),
}
_STR_FALLBACKS = ("", "a", "valid", "0")
_UNSET = object()


def make_value(spec, env, args, *, seed, dtype=None, shape=None, pattern=None,
               const=_UNSET, kind=None, strategy="init", resolve_relationships=True):
    if env.is_empty():
        return None
    if resolve_relationships:
        if any(source not in args or args[source].omitted
               for source in relationships.referenced(spec)):
            return None
        overrides = relationships.apply(spec, args)
    else:
        overrides = {}
    if kind is None or kind not in env.kinds:
        kind = env.kinds[0]
    if const is not _UNSET or kind in ("bool", "str") or (env.enum_values and kind != "dtype"):
        value = const
        if value is _UNSET:
            pool = env.enum_values or list(_STR_FALLBACKS if kind == "str" else [True, False])
            value = pool[seed % len(pool)]
        return Value(kind=kind, payload=value, recipe={"form": "const", "value": value},
                     strategy=strategy)
    if kind == "dtype":
        name = overrides.get("dtype") or dtype or (env.dtypes[0] if env.dtypes else None)
        if name is None:
            return None
        return Value(kind="dtype", payload=name, dtype=name,
                     recipe={"form": "const", "value": name, "as_dtype": True}, strategy=strategy)
    if kind == "dict":
        return Value(kind=kind, payload={}, recipe={"form": "dict", "items": {}}, strategy=strategy)
    if kind == "axis" and "axis_choices" in overrides:
        choices = overrides["axis_choices"]
        choices = [x for x in choices
                   if (env.low is None or x > env.low if env.exclusive_min else env.low is None or x >= env.low)
                   and (env.high is None or x < env.high if env.exclusive_max else env.high is None or x <= env.high)]
        if not choices:
            return None
        axis = choices[seed % len(choices)]
        return Value(kind="axis", payload=axis, dtype="int64",
                     recipe={"form": "const", "value": axis}, strategy=strategy)
    dt = overrides.get("dtype") or dtype or (env.dtypes[0] if env.dtypes else None)
    if dt is None:
        return None
    if kind in ("int", "axis") and np.dtype(dt).kind not in "iu":
        return None
    bounds = {"low": env.low, "high": env.high, "exclusive_min": env.exclusive_min,
              "exclusive_max": env.exclusive_max, "allow_nonfinite": env.allow_nonfinite}
    if kind in ("int", "number", "axis"):
        recipe = {"form": "scalar", "pattern": pattern or synth.ORDINARY,
                  "dtype": dt, "seed": seed, **bounds}
    elif kind in ("shape", "sequence"):
        selected = shape if shape is not None else env.shapes[seed % len(env.shapes)]
        length = selected[0]
        items = []
        for idx in range(length):
            item = {"form": "scalar", "pattern": pattern or synth.ORDINARY,
                    "dtype": dt, "seed": seed + idx, **bounds}
            if kind == "shape":
                item.update({"low": max(1, env.low or 1), "high": env.high or 4})
            items.append(item)
        recipe = {"form": "sequence", "items": items, "as_tuple": kind == "shape"}
    elif kind == "array":
        shp = overrides.get("shape")
        if shp is None:
            shp = shape if shape is not None else env.shapes[0]
        if len(shp) > MAX_RANK or np.prod(shp, dtype=object) > MAX_ELEMENTS:
            return None
        pat = pattern or synth.ORDINARY
        params = {}
        if pat == synth.NOISE:
            params = {"scale": 0.1}
        elif pat == synth.MASK:
            params = {"fill": 0, "ratio": 0.3}
        elif pat == synth.DIVISION:
            params = {"factor": float(2 ** (1 + seed % 4))}
        recipe = {"form": "array", "pattern": pat, "dtype": dt, "shape": list(shp),
                  "seed": seed, "params": params, **bounds}
    else:
        return None
    try:
        payload = synth.build_value(recipe)
    except (ValueError, TypeError, OverflowError):
        return None
    shp = tuple(payload.shape) if kind == "array" else (len(payload),) if kind in ("shape", "sequence") else ()
    return Value(kind=kind, payload=payload, dtype=dt, shape=shp,
                 recipe=recipe, strategy=strategy)


def derive_variants(spec, args, disabled=frozenset()):
    if keeps_default(spec):
        return []
    env = envelope_for(spec)
    if env.is_empty():
        return []
    overrides = relationships.apply(spec, args) if "relationships" not in disabled else {}
    variants = [{"strategy": "value", "const": v} for v in env.enum_values]
    if len(env.kinds) > 1:
        variants += [{"strategy": "type", "kind": k} for k in env.kinds]
    if "dtype" not in overrides:
        variants += [{"strategy": "type", "dtype": d} for d in env.dtypes]
    if "shape" not in overrides and "axis_choices" not in overrides:
        variants += [{"strategy": "size", "shape": s} for s in env.shapes]
    variants += [{"strategy": "value", "pattern": p} for p in env.patterns if p not in disabled]
    if env.omittable:
        variants.append({"strategy": "value", "omit": True})
    return [v for v in variants if v["strategy"] not in disabled]


def select_strategy(rng, variants):
    available = sorted({v["strategy"] for v in variants})
    return rng.choice(available) if available else "value"


def generate(spec, args, variants, strategy, rng, resolve_relationships=True):
    pool = [v for v in variants if v["strategy"] == strategy] or variants
    if not pool:
        return None
    choice = rng.choice(pool)
    if choice.get("omit"):
        return Value(kind="default", payload=spec.default, omitted=True,
                     recipe={"form": "const", "value": spec.default}, strategy=strategy)
    default_pattern = synth.BOUNDARY if all(v.get("pattern") != synth.ORDINARY for v in variants) else None
    return make_value(spec, envelope_for(spec), args, seed=rng.randrange(1 << 30),
                      dtype=choice.get("dtype"), shape=choice.get("shape"),
                      pattern=choice.get("pattern", default_pattern), const=choice.get("const", _UNSET),
                      kind=choice.get("kind"), strategy=strategy,
                      resolve_relationships=resolve_relationships)


def refresh_dependents(specs, args, rng):
    for spec in relationships.dependency_order(specs):
        current = args.get(spec.name)
        if current is None or current.omitted or not spec.relationships:
            continue
        if any(source not in args or args[source].omitted for source in relationships.referenced(spec)):
            raise ValueError(f"relationship source unavailable for {spec.name}")
        overrides = relationships.apply(spec, args)
        needs_shape = "shape" in overrides and tuple(overrides["shape"]) != current.shape
        needs_dtype = "dtype" in overrides and overrides["dtype"] != current.dtype
        axis_bad = ("axis_choices" in overrides and current.kind == "axis"
                    and current.payload not in overrides["axis_choices"])
        if needs_shape or needs_dtype or axis_bad:
            value = make_value(spec, envelope_for(spec), args, seed=rng.randrange(1 << 30),
                               pattern=current.recipe.get("pattern"), strategy=current.strategy)
            if value is None:
                raise ValueError(f"cannot resolve relationships for {spec.name}")
            args[spec.name] = value


def constraints_satisfied(specs, args, resolve_relationships=True):
    for spec in specs:
        value = args.get(spec.name)
        if value is None or value.omitted:
            if spec.required and spec.call_kind not in ("var_positional", "var_keyword"):
                return False
            continue
        env = envelope_for(spec)
        payload = value.payload
        if spec.enum_values and not any(_equal(payload, choice) for choice in spec.enum_values):
            return False
        if value.kind in ("sequence", "shape"):
            length = len(payload)
            if length < spec.constraints.get("len_min", 0) or length > spec.constraints.get("len_max", length):
                return False
        if value.kind == "array":
            shape = spec.shape
            if shape.get("dims") is not None and tuple(shape["dims"]) != value.shape:
                return False
            if len(value.shape) < shape.get("rank_min", 0) or len(value.shape) > shape.get("rank_max", MAX_RANK):
                return False
        if value.kind in ("array", "number", "int", "axis", "sequence", "shape"):
            arr = np.asarray(payload)
            if arr.dtype.kind in "biufc":
                if not env.allow_nonfinite and not np.all(np.isfinite(arr)):
                    return False
                for bound, exclusive, lower in ((env.low, env.exclusive_min, True),
                                                 (env.high, env.exclusive_max, False)):
                    if bound is not None:
                        if arr.dtype.kind == "c":
                            return False
                        valid = (arr > bound if exclusive else arr >= bound) if lower else (
                            arr < bound if exclusive else arr <= bound)
                        if not np.all(valid):
                            return False
        if resolve_relationships and spec.relationships:
            if any(source not in args or args[source].omitted for source in relationships.referenced(spec)):
                return False
            overrides = relationships.apply(spec, args)
            if "shape" in overrides and value.shape != tuple(overrides["shape"]):
                return False
            if "dtype" in overrides and value.dtype != overrides["dtype"]:
                return False
            if "axis_choices" in overrides and value.payload not in overrides["axis_choices"]:
                return False
    return True


def _equal(a, b):
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except (TypeError, ValueError):
        try:
            return bool(a == b)
        except (TypeError, ValueError):
            return False
