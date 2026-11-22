from __future__ import annotations

from typing import Any

from .models import ParamSpec, Value


def dependency_order(specs: list[ParamSpec]) -> list[ParamSpec]:
    by_name = {p.name: p for p in specs}
    if len(by_name) != len(specs):
        raise ValueError("duplicate parameter names")
    ordered, active, done = [], set(), set()

    def visit(name):
        if name in active:
            raise ValueError(f"cyclic relationship involving {name}")
        if name in done:
            return
        if name not in by_name:
            raise ValueError(f"relationship source {name!r} is absent from the signature")
        active.add(name)
        for source in sorted(referenced(by_name[name])):
            visit(source)
        active.remove(name)
        done.add(name)
        ordered.append(by_name[name])

    for param in specs:
        visit(param.name)
    return ordered


def required_sources(specs: list[ParamSpec]) -> set[str]:
    needed = set()
    for spec in reversed(dependency_order(specs)):
        if spec.required or spec.name in needed:
            needed.update(referenced(spec))
    return needed


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
