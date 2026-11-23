from __future__ import annotations

from dataclasses import dataclass

from .collector import ApiRecord, has_runtime_signature, receiver_record, signature_params
from .models import ApiSpec, ParamSpec, SUPPORTED_RELATIONSHIPS
from .relationships import dependency_order


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


def _align(params: list[ParamSpec], rows: list[dict], notes: list[str],
           label: str) -> tuple[list[ParamSpec], list[str], list[str]]:
    by_name = {row["name"]: row for row in rows}
    extracted = {p.name: p for p in params}
    validated: list[ParamSpec] = []
    corrected: list[str] = []
    for row in rows:
        name = row["name"]
        param = extracted.get(name)
        if param is None:
            param = ParamSpec(name=name, kind="unknown")
            notes.append(f"{label} parameter {name!r} missing from extraction")
        if param.required != bool(row["required"]) or param.call_kind != row["call_kind"]:
            corrected.append(name)
        param.required = bool(row["required"])
        param.has_default = bool(row["has_default"])
        param.default = row["default"] if row["has_default"] else None
        param.position = row["position"]
        param.call_kind = row["call_kind"]
        validated.append(param)
    dropped = [p.name for p in params if p.name not in by_name]
    surviving = {p.name for p in validated}
    for param in validated:
        param.relationships = [r for r in param.relationships
                               if r.kind in SUPPORTED_RELATIONSHIPS
                               and r.source in surviving and r.source != param.name]
    return validated, dropped, corrected


def _validate_receiver(spec: ApiSpec, record: ApiRecord) -> str:
    base = receiver_record(record)
    if base is None:
        return "method without a resolvable class"
    rows = signature_params(base)
    if rows:
        spec.receiver_params, _, _ = _align(spec.receiver_params, rows,
                                            spec.notes, "receiver")
    else:
        spec.receiver_params = []
        spec.notes.append("receiver constructor signature unavailable")
    essential = [p for p in spec.receiver_params
                 if p.required and p.call_kind not in ("var_positional", "var_keyword")]
    if len(essential) > 6:
        return f"receiver needs {len(essential)} required constructor arguments"
    blocked = [p.name for p in essential if p.name in _UNSUPPORTED_NAMES]
    if blocked:
        return f"receiver needs unsupported argument(s): {', '.join(blocked)}"
    return ""


def validate(spec: ApiSpec, record: ApiRecord, *, signature_policy: str = "runtime") -> ValidationResult:
    if signature_policy not in ("runtime", "documented"):
        raise ValueError(f"unknown signature policy: {signature_policy}")
    if not record.doc.strip():
        return ValidationResult(None, False, "no usable documentation")
    if signature_policy == "runtime" and not has_runtime_signature(record):
        return ValidationResult(None, False, "no usable runtime signature; documentation is not an independent signature")
    if len({p.name for p in spec.params}) != len(spec.params):
        return ValidationResult(None, False, "duplicate parameter names in extraction")
    if spec.is_method:
        reason = _validate_receiver(spec, record)
        if reason:
            return ValidationResult(None, False, reason)
    rows = signature_params(record)
    if not rows:
        if spec.is_method and record.signature is not None:
            remaining = [p for p in record.signature.parameters if p not in ("self", "cls")]
            if not remaining:
                spec.params = []
                return ValidationResult(spec, True)
        return ValidationResult(None, False, "no usable runtime or binding signature")

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
            return ValidationResult(None, False,
                                    f"unresolvable relationship(s) on {param.name!r}")
        param.relationships = kept

    spec.params = validated
    try:
        dependency_order(validated)
    except ValueError as exc:
        return ValidationResult(None, False, str(exc))

    essential = [p for p in validated
                 if p.required and p.call_kind not in ("var_positional", "var_keyword")]
    if not validated and not spec.is_method:
        return ValidationResult(None, False, "no parameters to generate")
    blocked = [p.name for p in essential if p.name in _UNSUPPORTED_NAMES]
    if blocked:
        return ValidationResult(None, False,
                                f"required unsupported parameter(s): {', '.join(blocked)}")
    if len(essential) > 8:
        return ValidationResult(None, False,
                                f"{len(essential)} required parameters exceed the arity limit")
    return ValidationResult(spec, True, dropped=dropped, corrected=corrected)
