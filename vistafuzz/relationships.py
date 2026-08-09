from __future__ import annotations

from typing import Any

from .models import ParamSpec, Value


def referenced(spec: ParamSpec) -> set[str]:
    return {rel.source for rel in spec.relationships}


def resolve_shape(spec: ParamSpec, args: dict[str, Value]) -> tuple[int, ...] | None:
    source = spec.follows_shape_of
    if not source:
        return None
    value = args.get(source)
    if value is None or value.shape is None:
        return None
    return tuple(value.shape)


def resolve_dtype(spec: ParamSpec, args: dict[str, Value]) -> str | None:
    source = spec.follows_dtype_of
    if not source:
        return None
    value = args.get(source)
    if value is None:
        return None
    return value.dtype


def resolve_axis_bounds(spec: ParamSpec, args: dict[str, Value]) -> tuple[int, int] | None:
    source = spec.axis_of
    if not source:
        return None
    value = args.get(source)
    if value is None or value.shape is None:
        return None
    rank = len(value.shape)
    if rank == 0:
        return (0, 0)
    return (-rank, rank - 1)


def axis_candidates(spec: ParamSpec, args: dict[str, Value]) -> list[int]:
    bounds = resolve_axis_bounds(spec, args)
    if bounds is None:
        return []
    low, high = bounds
    return list(range(low, high + 1))


def unresolved(spec: ParamSpec, args: dict[str, Value]) -> list[str]:
    return [rel.source for rel in spec.relationships if rel.source not in args]


def apply(spec: ParamSpec, args: dict[str, Value]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    shape = resolve_shape(spec, args)
    if shape is not None:
        out["shape"] = shape
    dtype = resolve_dtype(spec, args)
    if dtype is not None:
        out["dtype"] = dtype
    axes = axis_candidates(spec, args)
    if axes:
        out["axis_choices"] = axes
    return out
