from __future__ import annotations

import random
from typing import Any

from . import relationships, synth
from .constraints import Envelope, envelope_for, keeps_default
from .models import ParamSpec, Value

STRATEGIES = ("type", "size", "value")

_STR_FALLBACKS = ("", "a", "valid", "0")


def make_value(spec: ParamSpec, env: Envelope, args: dict[str, Value], *,
               seed: int, dtype: str | None = None,
               shape: tuple[int, ...] | None = None,
               pattern: str | None = None, const: Any = None,
               kind: str | None = None,
               strategy: str = "init") -> Value | None:
    overrides = relationships.apply(spec, args)
    if kind is None or kind not in env.kinds:
        kind = env.kinds[0] if env.kinds else "array"

    if const is not None or kind in ("bool", "str") or (env.enum_values and kind != "dtype"):
        value = const
        if value is None:
            pool = env.enum_values or list(_STR_FALLBACKS if kind == "str" else [True, False])
            value = pool[seed % len(pool)] if pool else None
        return Value(kind=kind, payload=value, recipe={"form": "const", "value": value},
                     strategy=strategy)

    if kind == "dtype":
        name = dtype or overrides.get("dtype") or (env.dtypes[0] if env.dtypes else "float64")
        return Value(kind="dtype", payload=name, dtype=name,
                     recipe={"form": "const", "value": name, "as_dtype": True},
                     strategy=strategy)

    if kind == "axis":
        choices = overrides.get("axis_choices")
        if choices:
            axis = choices[seed % len(choices)]
        else:
            axis = int(synth.build_scalar(pattern or synth.ORDINARY, "int64", seed,
                                          env.low, env.high))
            axis = max(-4, min(4, axis))
        return Value(kind="axis", payload=axis, dtype="int64",
                     recipe={"form": "const", "value": axis}, strategy=strategy)

    if kind in ("int", "number"):
        dt = dtype or overrides.get("dtype") or (env.dtypes[0] if env.dtypes else "float64")
        if kind == "int" and dt.startswith("float"):
            dt = "int64"
        recipe = {"form": "scalar", "pattern": pattern or synth.ORDINARY, "dtype": dt,
                  "seed": seed, "low": env.low, "high": env.high}
        return Value(kind=kind, payload=synth.build_value(recipe), dtype=dt,
                     recipe=recipe, strategy=strategy)

    if kind in ("shape", "sequence"):
        length = len(shape) if shape else (1 + seed % 3)
        items = [{"form": "const", "value": int(1 + (seed + i) % 4)} for i in range(length)]
        recipe = {"form": "sequence", "items": items, "as_tuple": kind == "shape"}
        return Value(kind=kind, payload=synth.build_value(recipe),
                     shape=(length,), recipe=recipe, strategy=strategy)

    dt = overrides.get("dtype") or dtype or (env.dtypes[0] if env.dtypes else "float64")
    shp = overrides.get("shape") or shape or (env.shapes[0] if env.shapes else (2, 3))
    pat = pattern or synth.ORDINARY
    params: dict[str, Any] = {}
    if pat == synth.NOISE:
        params = {"scale": 0.1}
    elif pat == synth.MASK:
        params = {"fill": 0, "ratio": 0.3}
    elif pat == synth.DIVISION:
        params = {"factor": float(2 ** (1 + seed % 4))}
    if dt.startswith(("int", "uint")) and pat == synth.NONFINITE:
        pat = synth.EXTREME
    recipe = {"form": "array", "pattern": pat, "dtype": dt, "shape": list(shp),
              "seed": seed, "low": env.low, "high": env.high, "params": params}
    try:
        payload = synth.build_value(recipe)
    except Exception:
        return None
    return Value(kind="array", payload=payload, dtype=dt, shape=tuple(shp),
                 recipe=recipe, strategy=strategy)


def derive_variants(spec: ParamSpec, args: dict[str, Value]) -> list[dict[str, Any]]:
    if keeps_default(spec):
        return []
    env = envelope_for(spec)
    overrides = relationships.apply(spec, args)
    variants: list[dict[str, Any]] = []

    if env.enum_values:
        variants += [{"strategy": "value", "const": v} for v in env.enum_values[:8]]

    if len(env.kinds) > 1:
        variants += [{"strategy": "type", "kind": k} for k in env.kinds]

    if "dtype" not in overrides:
        for dtype in env.dtypes[:4]:
            variants.append({"strategy": "type", "dtype": dtype})
    if env.kinds and env.kinds[0] == "dtype":
        for dtype in env.dtypes[:4]:
            variants.append({"strategy": "type", "dtype": dtype})

    if "shape" not in overrides and "axis_choices" not in overrides:
        for shape in env.shapes[:6]:
            variants.append({"strategy": "size", "shape": tuple(shape)})

    if "axis_choices" in overrides:
        variants += [{"strategy": "value", "const": a}
                     for a in overrides["axis_choices"][:6]]

    for pattern in env.patterns[:6]:
        variants.append({"strategy": "value", "pattern": pattern})

    if env.omittable:
        variants.append({"strategy": "value", "omit": True})
    return variants


def select_strategy(rng: random.Random, variants: list[dict[str, Any]]) -> str:
    available = sorted({v["strategy"] for v in variants})
    return rng.choice(available) if available else "value"


def generate(spec: ParamSpec, args: dict[str, Value], variants: list[dict[str, Any]],
             strategy: str, rng: random.Random) -> Value | None:
    pool = [v for v in variants if v["strategy"] == strategy] or variants
    if not pool:
        return None
    choice = rng.choice(pool)
    env = envelope_for(spec)
    seed = rng.randrange(1 << 30)

    if choice.get("omit"):
        return Value(kind="default", payload=spec.default, omitted=True,
                     recipe={"form": "const", "value": spec.default}, strategy=strategy)
    return make_value(spec, env, args, seed=seed,
                      dtype=choice.get("dtype"), shape=choice.get("shape"),
                      pattern=choice.get("pattern"), const=choice.get("const"),
                      kind=choice.get("kind"), strategy=strategy)


def refresh_dependents(specs: list[ParamSpec], args: dict[str, Value],
                       rng: random.Random) -> None:
    for spec in specs:
        if not spec.relationships or spec.name not in args:
            continue
        current = args[spec.name]
        if current.omitted:
            continue
        overrides = relationships.apply(spec, args)
        needs_shape = "shape" in overrides and tuple(overrides["shape"]) != (current.shape or ())
        needs_dtype = "dtype" in overrides and overrides["dtype"] != current.dtype
        axis_bad = ("axis_choices" in overrides
                    and current.kind == "axis"
                    and current.payload not in overrides["axis_choices"])
        if not (needs_shape or needs_dtype or axis_bad):
            continue
        env = envelope_for(spec)
        value = make_value(spec, env, args, seed=rng.randrange(1 << 30),
                           pattern=current.recipe.get("pattern"),
                           strategy=current.strategy)
        if value is not None:
            args[spec.name] = value
