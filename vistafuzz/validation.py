from __future__ import annotations

from dataclasses import dataclass

from .collector import ApiRecord, signature_params
from .models import ApiSpec, ParamSpec, SUPPORTED_RELATIONSHIPS


@dataclass
class ValidationResult:
    spec: ApiSpec | None
    ok: bool
    reason: str = ""
    dropped: list[str] = None
    corrected: list[str] = None

    def __post_init__(self) -> None:
        self.dropped = self.dropped or []
        self.corrected = self.corrected or []


_UNSUPPORTED_NAMES = {"out", "output", "device", "stream", "session", "graph",
                      "ctx", "context", "handle", "sess"}


def validate(spec: ApiSpec, record: ApiRecord) -> ValidationResult:
    rows = signature_params(record)
    if not rows:
        for idx, param in enumerate(spec.params):
            param.position = idx
        if not spec.params:
            return ValidationResult(None, False, "no runtime signature and no parameters")
        spec.notes.append("runtime signature unavailable; keyword invocation only")
        for param in spec.params:
            param.call_kind = "keyword"
        return ValidationResult(spec, True)

    by_name = {row["name"]: row for row in rows}
    validated: list[ParamSpec] = []
    dropped: list[str] = []
    corrected: list[str] = []

    extracted = {p.name: p for p in spec.params}
    for row in rows:
        name = row["name"]
        param = extracted.get(name)
        if param is None:
            param = ParamSpec(name=name, kind="unknown", doc="(not returned by extractor)")
            spec.notes.append(f"parameter {name!r} missing from extraction; signature-only")
        if param.required != bool(row["required"]) or param.call_kind != row["call_kind"]:
            corrected.append(name)
        param.required = bool(row["required"])
        param.has_default = bool(row["has_default"])
        param.default = row["default"] if row["has_default"] else None
        param.position = row["position"]
        param.call_kind = row["call_kind"]
        validated.append(param)

    for param in spec.params:
        if param.name not in by_name:
            dropped.append(param.name)

    surviving = {p.name for p in validated}
    for param in validated:
        kept = [r for r in param.relationships
                if r.kind in SUPPORTED_RELATIONSHIPS and r.source in surviving
                and r.source != param.name]
        if len(kept) != len(param.relationships):
            spec.notes.append(f"dropped unresolvable relationship(s) on {param.name!r}")
        param.relationships = kept

    spec.params = validated

    essential = [p for p in validated
                 if p.required and p.call_kind not in ("var_positional", "var_keyword")]
    if not essential and not validated:
        return ValidationResult(None, False, "no parameters to generate")
    blocked = [p.name for p in essential if p.name in _UNSUPPORTED_NAMES]
    if blocked:
        return ValidationResult(None, False,
                                f"required unsupported parameter(s): {', '.join(blocked)}")
    if len(essential) > 8:
        return ValidationResult(None, False,
                                f"{len(essential)} required parameters exceed the arity limit")
    return ValidationResult(spec, True, dropped=dropped, corrected=corrected)
