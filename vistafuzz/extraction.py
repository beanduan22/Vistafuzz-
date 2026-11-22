from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from . import llm as llm_mod
from .collector import ApiRecord, receiver_record, signature_params
from .models import (ApiSpec, INT_NAMES, KINDS, ParamSpec, Relationship,
                     REL_AXIS_OF, REL_DTYPE_FOLLOWS, REL_SHAPE_FOLLOWS,
                     SUPPORTED_RELATIONSHIPS)
from .prompt import MAX_DOC_CHARS, build_prompt

DEFAULT_FLOAT_DTYPES = ["float32", "float64"]
DEFAULT_INT_DTYPES = ["int32", "int64"]

_ARRAY_WORDS = ("array", "ndarray", "tensor", "matrix", "array_like", "array-like",
                "sequence of", "image", "img", "vector")
_INT_WORDS = ("int", "integer", "index", "count", "number of", "size", "length", "n_")
_FLOAT_WORDS = ("float", "double", "scalar", "real", "number", "value", "rate",
                "tolerance", "epsilon", "alpha", "beta", "gamma", "threshold")
_BOOL_WORDS = ("bool", "boolean", "flag", "whether", "if true", "if `true`")
_STR_WORDS = ("str", "string", "name", "mode", "method", "kind", "order", "norm")
_SEQ_WORDS = ("tuple", "list", "sequence", "iterable", "shape")


def _coerce_rel(entry: Any, self_name: str) -> Relationship | None:
    if not isinstance(entry, dict):
        return None
    kind = str(entry.get("kind") or entry.get("type") or "").strip()
    source = str(entry.get("source") or entry.get("ref") or "").strip()
    alias = {"shape_following": REL_SHAPE_FOLLOWS, "same_shape": REL_SHAPE_FOLLOWS,
             "rank_bounded_axis": REL_AXIS_OF, "axis": REL_AXIS_OF,
             "type_following": REL_DTYPE_FOLLOWS, "same_dtype": REL_DTYPE_FOLLOWS}
    kind = alias.get(kind, kind)
    if kind not in SUPPORTED_RELATIONSHIPS or not source or source == self_name:
        return None
    return Relationship(kind=kind, source=source)


def _clean_map(raw: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key in keys:
        value = raw.get(key)
        if value in (None, "", "null", []):
            continue
        out[key] = value
    return out


def parse_llm_params(payload: dict[str, Any]) -> list[ParamSpec]:
    rows = payload.get("params")
    if not isinstance(rows, list):
        raise llm_mod.LLMError("response has no 'params' list")
    specs: list[ParamSpec] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        name = str(row["name"])
        kind = str(row.get("kind") or "unknown").strip().lower()
        if kind not in KINDS:
            kind = {"tensor": "array", "ndarray": "array", "float": "number",
                    "integer": "int", "boolean": "bool", "string": "str",
                    "tuple": "sequence", "list": "sequence"}.get(kind, "unknown")
        dtypes = [str(d).rsplit(".", 1)[-1] for d in (row.get("dtype_candidates") or []) if d]
        rels = [r for r in (_coerce_rel(e, name) for e in row.get("relationships") or [])
                if r is not None]
        specs.append(ParamSpec(
            name=name,
            kind=kind,
            required=bool(row.get("required", True)),
            default=row.get("default"),
            has_default=not bool(row.get("required", True)),
            position=row.get("position") if isinstance(row.get("position"), int) else len(specs),
            call_kind=str(row.get("call_kind") or "positional_or_keyword"),
            dtype_candidates=dtypes,
            shape=_clean_map(row.get("shape"), ("rank_min", "rank_max", "dims")),
            constraints=_clean_map(row.get("constraints"),
                                   ("min", "max", "exclusive_min", "exclusive_max",
                                    "allow_nonfinite", "len_min", "len_max")),
            enum_values=list(row.get("enum_values") or []),
            relationships=rels,
            doc=str(row.get("doc") or "")[:400],
        ))
    if not specs:
        raise llm_mod.LLMError("response contained no usable parameter entries")
    return specs


def extract_with_llm(record: ApiRecord, config: llm_mod.LLMConfig,
                     prompt_input: str = "doc+sig") -> list[ParamSpec]:
    rows = signature_params(record)
    prompt = build_prompt(record.api, record.signature_text, record.doc, rows, prompt_input)
    text = llm_mod.complete(prompt, config)
    return parse_llm_params(llm_mod.extract_json_object(text))


def _doc_blocks(doc: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    header = re.compile(r"^\s*[*`]*([A-Za-z_][A-Za-z0-9_]*)[*`]*\s*(?:\(([^)]*)\))?\s*[:：]\s*(.*)$")
    for line in (doc or "").splitlines():
        match = header.match(line)
        if match and (len(line) - len(line.lstrip())) <= 8:
            if current:
                blocks[current] = "\n".join(buf)
            current = match.group(1)
            buf = [f"{match.group(2) or ''} {match.group(3) or ''}"]
        elif current:
            buf.append(line.strip())
    if current:
        blocks[current] = "\n".join(buf)
    return blocks


def _kind_from_text(name: str, text: str, default: Any) -> str:
    blob = f"{name} {text}".lower()
    if name.lower() in ("axis", "dim", "axes", "dims") or "axis" in blob[:40]:
        return "axis"
    if name.lower() in ("dtype", "out_dtype", "type") or "data-type" in blob or "data type" in blob:
        return "dtype"
    if isinstance(default, bool) or any(w in blob for w in _BOOL_WORDS):
        return "bool"
    if any(w in blob for w in _ARRAY_WORDS):
        return "array"
    if name.lower() in ("shape", "size", "newshape", "output_shape"):
        return "shape"
    if any(w in blob for w in _SEQ_WORDS):
        return "sequence"
    if isinstance(default, str) or any(w in blob for w in _STR_WORDS):
        return "str"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int) or any(w in blob for w in _INT_WORDS):
        return "int"
    if isinstance(default, float) or any(w in blob for w in _FLOAT_WORDS):
        return "number"
    if name.lower() in INT_NAMES:
        return "int"
    return "array" if text.strip() else "unknown"


_RANGE_RE = re.compile(r"(?:in|within|between)\s*[\[\(]\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*[\]\)]")
_NONNEG_RE = re.compile(r"non-?negative|must be positive|greater than or equal to 0|>=\s*0")
_POSITIVE_RE = re.compile(r"must be positive|strictly positive|greater than 0|>\s*0")
_SAME_SHAPE_RE = re.compile(r"same shape (?:as|with)\s*[`'\"]?([A-Za-z_][A-Za-z0-9_]*)")
_SAME_TYPE_RE = re.compile(r"same (?:data[- ]?type|dtype|type) (?:as|with)\s*[`'\"]?([A-Za-z_][A-Za-z0-9_]*)")
_ENUM_RE = re.compile(r"one of\s*[:\s]*\{?([^.}\n]+)\}?")


def _constraints_from_text(text: str) -> dict[str, Any]:
    low = text.lower()
    out: dict[str, Any] = {}
    match = _RANGE_RE.search(low)
    if match:
        try:
            out["min"], out["max"] = float(match.group(1)), float(match.group(2))
        except ValueError:
            pass
    if _POSITIVE_RE.search(low):
        out["min"], out["exclusive_min"] = 0.0, True
    elif _NONNEG_RE.search(low):
        out["min"] = 0.0
    if "must be finite" in low or "not nan" in low:
        out["allow_nonfinite"] = False
    return out


def _relationships_from_text(name: str, text: str, kind: str,
                             known: set[str]) -> list[Relationship]:
    rels: list[Relationship] = []
    low = text.lower()
    shape_hit = _SAME_SHAPE_RE.search(low)
    if shape_hit and shape_hit.group(1) in known and shape_hit.group(1) != name:
        rels.append(Relationship(REL_SHAPE_FOLLOWS, shape_hit.group(1)))
    type_hit = _SAME_TYPE_RE.search(low)
    if type_hit and type_hit.group(1) in known and type_hit.group(1) != name:
        rels.append(Relationship(REL_DTYPE_FOLLOWS, type_hit.group(1)))
    if kind == "axis":
        for ref in ("input", "x", "a", "arr", "array", "tensor", "data"):
            if ref in known and ref != name:
                rels.append(Relationship(REL_AXIS_OF, ref))
                break
    return rels


def extract_offline(record: ApiRecord) -> list[ParamSpec]:
    rows = signature_params(record)
    blocks = _doc_blocks(record.doc)
    names = {row["name"] for row in rows}
    specs: list[ParamSpec] = []
    for row in rows:
        name = row["name"]
        text = blocks.get(name, "")
        kind = _kind_from_text(name, text, row.get("default"))
        dtypes: list[str] = []
        if kind in ("array", "number", "dtype"):
            dtypes = list(DEFAULT_FLOAT_DTYPES)
        elif kind in ("int", "axis"):
            dtypes = list(DEFAULT_INT_DTYPES)
        enum: list[Any] = []
        if kind == "bool":
            enum = [True, False]
        elif kind == "str":
            hit = _ENUM_RE.search(text.lower())
            if hit:
                enum = [w.strip(" '\"`") for w in hit.group(1).split(",")][:6]
                enum = [w for w in enum if w and " " not in w][:6]
        specs.append(ParamSpec(
            name=name,
            kind=kind,
            required=bool(row["required"]),
            default=row.get("default"),
            has_default=bool(row["has_default"]),
            position=row["position"],
            call_kind=row["call_kind"],
            dtype_candidates=dtypes,
            shape={"rank_min": 1, "rank_max": 3} if kind == "array" else {},
            constraints=_constraints_from_text(text),
            enum_values=enum,
            relationships=_relationships_from_text(name, text, kind, names),
            doc=text.strip()[:400],
        ))
    return specs


_RECEIVER_CACHE: dict[tuple[str, str], list[ParamSpec]] = {}


def _extract_params(record: ApiRecord, extractor: str,
                    config: llm_mod.LLMConfig | None,
                    notes: list[str], prompt_input: str = "doc+sig") -> tuple[list[ParamSpec], str]:
    if extractor == "llm":
        config = config or llm_mod.LLMConfig.from_env()
        return extract_with_llm(record, config, prompt_input), f"llm:{config.backend}:{config.model}"
    if extractor == "signature":
        return extract_offline(record), "signature"
    raise ValueError(f"unknown extractor {extractor!r}")


def extract(record: ApiRecord, *, extractor: str = "llm",
            config: llm_mod.LLMConfig | None = None, cache_dir: str = "",
            prompt_input: str = "doc+sig") -> ApiSpec:
    config = config or llm_mod.LLMConfig.from_env()
    identity = {"api": record.api, "doc": record.doc, "signature": record.signature_text,
                "extractor": extractor, "backend": config.backend, "model": config.model,
                "temperature": config.temperature, "num_ctx": config.num_ctx,
                "max_tokens": config.max_tokens, "prompt_input": prompt_input,
                "prompt": build_prompt(record.api, record.signature_text, record.doc,
                                       signature_params(record), prompt_input)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_path = os.path.join(cache_dir, record.api + "_" + key[:20] + ".json")
    if cache_dir and os.path.exists(cache_path):
        with open(cache_path) as handle:
            return ApiSpec.from_json(json.load(handle)["spec"])
    notes: list[str] = []
    params, used = _extract_params(record, extractor, config, notes, prompt_input)

    receiver_params: list[ParamSpec] = []
    receiver_path = ""
    if record.is_method:
        receiver_path = record.class_path
        receiver_key = (receiver_path, key)
        if receiver_key in _RECEIVER_CACHE:
            receiver_params = [ParamSpec.from_json(p.to_json())
                               for p in _RECEIVER_CACHE[receiver_key]]
        else:
            base = receiver_record(record)
            if base is not None:
                receiver_params, _ = _extract_params(base, extractor, config, notes, prompt_input)
                _RECEIVER_CACHE[receiver_key] = [ParamSpec.from_json(p.to_json())
                                        for p in receiver_params]

    spec = ApiSpec(api=record.api, signature_text=record.signature_text,
                   doc=record.doc, params=params, extractor=used, notes=notes,
                   is_method=record.is_method, receiver_path=receiver_path,
                   receiver_params=receiver_params, signature_source=record.signature_source,
                   prompt_input=prompt_input,
                   documentation_truncated=prompt_input != "sig-only" and len(record.doc) > MAX_DOC_CHARS)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w") as handle:
            json.dump({"cache_key": key, "spec": spec.to_json()}, handle, indent=2, default=repr)
    return spec
