from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

KINDS = (
    "array",
    "number",
    "int",
    "bool",
    "str",
    "dtype",
    "axis",
    "shape",
    "sequence",
    "dict",
    "callable",
    "device",
    "receiver",
    "unknown",
)

REL_SHAPE_FOLLOWS = "shape_follows"
REL_AXIS_OF = "axis_of"
REL_DTYPE_FOLLOWS = "dtype_follows"
SUPPORTED_RELATIONSHIPS = (REL_SHAPE_FOLLOWS, REL_AXIS_OF, REL_DTYPE_FOLLOWS)

INT_NAMES = frozenset({
    "year", "month", "day", "hour", "minute", "second", "microsecond",
    "millisecond", "week", "weekday", "n", "k", "m", "size", "count", "num",
    "number", "index", "idx", "length", "width", "height", "rows", "cols",
    "columns", "ndim", "digits", "precision", "denominator", "numerator",
    "repeats", "times", "steps", "iterations", "bins", "degree", "order",
    "radix", "base", "offset", "start", "stop", "step", "maxlen", "capacity",
    "max_denominator", "decimals", "nbytes", "period", "periods", "window",
})

CALL_POSITIONAL = "positional"
CALL_POSITIONAL_OR_KEYWORD = "positional_or_keyword"
CALL_KEYWORD = "keyword"
CALL_VAR_POSITIONAL = "var_positional"
CALL_VAR_KEYWORD = "var_keyword"


@dataclass
class Relationship:

    kind: str
    source: str

    def is_supported(self) -> bool:
        return self.kind in SUPPORTED_RELATIONSHIPS


@dataclass
class ParamSpec:

    name: str
    kind: str = "unknown"
    required: bool = True
    default: Any = None
    has_default: bool = False
    position: int | None = None
    call_kind: str = CALL_POSITIONAL_OR_KEYWORD
    constraints: dict[str, Any] = field(default_factory=dict)
    enum_values: list[Any] = field(default_factory=list)
    dtype_candidates: list[str] = field(default_factory=list)
    shape: dict[str, Any] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)
    doc: str = ""

    def relationship(self, kind: str) -> Relationship | None:
        for rel in self.relationships:
            if rel.kind == kind:
                return rel
        return None

    @property
    def follows_shape_of(self) -> str | None:
        rel = self.relationship(REL_SHAPE_FOLLOWS)
        return rel.source if rel else None

    @property
    def axis_of(self) -> str | None:
        rel = self.relationship(REL_AXIS_OF)
        return rel.source if rel else None

    @property
    def follows_dtype_of(self) -> str | None:
        rel = self.relationship(REL_DTYPE_FOLLOWS)
        return rel.source if rel else None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["relationships"] = [asdict(r) for r in self.relationships]
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ParamSpec":
        data = dict(data)
        rels = [Relationship(**r) for r in data.pop("relationships", []) or []]
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in data.items() if k in known}
        return cls(relationships=rels, **data)


@dataclass
class ApiSpec:

    api: str
    signature_text: str = ""
    doc: str = ""
    params: list[ParamSpec] = field(default_factory=list)
    extractor: str = ""
    notes: list[str] = field(default_factory=list)
    is_method: bool = False
    receiver_path: str = ""
    receiver_params: list[ParamSpec] = field(default_factory=list)

    def param(self, name: str) -> ParamSpec | None:
        for p in self.params:
            if p.name == name:
                return p
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "api": self.api,
            "signature_text": self.signature_text,
            "doc": self.doc,
            "extractor": self.extractor,
            "notes": list(self.notes),
            "params": [p.to_json() for p in self.params],
            "is_method": self.is_method,
            "receiver_path": self.receiver_path,
            "receiver_params": [p.to_json() for p in self.receiver_params],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ApiSpec":
        return cls(
            api=data["api"],
            signature_text=data.get("signature_text", ""),
            doc=data.get("doc", ""),
            extractor=data.get("extractor", ""),
            notes=list(data.get("notes", [])),
            params=[ParamSpec.from_json(p) for p in data.get("params", [])],
            is_method=bool(data.get("is_method", False)),
            receiver_path=data.get("receiver_path", ""),
            receiver_params=[ParamSpec.from_json(p)
                             for p in data.get("receiver_params", [])],
        )


@dataclass
class Value:

    kind: str
    payload: Any = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    recipe: dict[str, Any] = field(default_factory=dict)
    strategy: str = "init"
    omitted: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape is not None else None,
            "recipe": self.recipe,
            "strategy": self.strategy,
            "omitted": self.omitted,
        }


@dataclass
class Finding:

    api: str
    oracle: str
    kind: str
    detail: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    strategy: str = ""
    receiver_path: str = ""
    receiver_args: dict[str, Any] = field(default_factory=dict)
    receiver_order: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
