from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
from dataclasses import dataclass
from typing import Any, Iterator

from .models import (CALL_KEYWORD, CALL_POSITIONAL, CALL_POSITIONAL_OR_KEYWORD,
                     CALL_VAR_KEYWORD, CALL_VAR_POSITIONAL)

_EXCLUDE_TOKENS = frozenset({
    "save", "load", "read", "write", "download", "fetch", "dump", "export",
    "checkpoint", "serialize", "deserialize", "open", "close", "file", "path",
    "url", "dataset", "datasets", "cache", "pickle", "csv", "json", "npz",
    "cuda", "gpu", "tpu", "npu", "xla", "device", "devices", "distributed",
    "rpc", "nccl", "spawn", "daemon", "thread", "threads", "process",
    "processes", "server", "cluster", "socket", "stream",
    "session", "graph", "compile", "jit", "context", "scope", "handle",
    "register", "hook", "callback", "state",
    "print", "show", "plot", "draw", "display", "summary", "repr", "logger",
    "logging", "verbosity", "warn", "warning", "assert",
    "delete", "remove", "destroy", "reset", "clear", "free", "shutdown",
    "kill", "exit", "quit", "seed",
})
_EXCLUDE_SUBSTRINGS = ("imread", "imwrite", "imshow", "savetxt", "loadtxt",
                       "fromfile", "tofile", "memmap", "printoptions")
_EXCLUDE_MODULE_PATTERNS = (
    "test", "tests", "testing", "benchmark", "_pytest", "conftest",
    "distributed", "rpc", "onnx", "serialization", "utils.data", "datasets",
    "io", "compiler", "profiler", "debug", "deprecat",
)
_PRIVATE = re.compile(r"(^|\.)_")

_EXCLUDE_EXACT_NAMES = frozenset({
    "test", "tester", "bench", "benchmark", "main", "setup", "configuration",
    "info", "who", "source", "lookfor", "help", "deprecate", "get_include",
})

_NUMERIC_HINTS = (
    "array", "ndarray", "tensor", "matrix", "vector", "scalar", "dtype",
    "float", "int", "axis", "shape", "elementwise", "element-wise", "compute",
    "returns", "numeric", "numerical", "value", "input",
)


@dataclass
class ApiRecord:

    api: str
    obj: Any
    doc: str
    signature_text: str
    signature: inspect.Signature | None
    module: str


def _iter_modules(root: str, max_depth: int) -> Iterator[str]:
    yield root
    try:
        mod = importlib.import_module(root)
    except Exception:
        return
    paths = getattr(mod, "__path__", None)
    if not paths:
        return
    for info in pkgutil.walk_packages(paths, prefix=root + "."):
        name = info.name
        depth = name.count(".") - root.count(".")
        if depth > max_depth:
            continue
        tail = name[len(root) + 1:]
        if _PRIVATE.search("." + tail):
            continue
        if any(p in tail.lower() for p in _EXCLUDE_MODULE_PATTERNS):
            continue
        yield name


def _call_kind(param: inspect.Parameter) -> str:
    return {
        inspect.Parameter.POSITIONAL_ONLY: CALL_POSITIONAL,
        inspect.Parameter.POSITIONAL_OR_KEYWORD: CALL_POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY: CALL_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL: CALL_VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD: CALL_VAR_KEYWORD,
    }[param.kind]


_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def name_tokens(name: str) -> set[str]:
    parts: list[str] = []
    for chunk in name.split("_"):
        parts.extend(p for p in _CAMEL_SPLIT.split(chunk) if p)
    return {p.lower() for p in parts}


def _is_excluded_name(qualname: str) -> bool:
    tail = qualname.rsplit(".", 1)[-1]
    lowered = tail.lower()
    if lowered in _EXCLUDE_EXACT_NAMES:
        return True
    if any(sub in lowered for sub in _EXCLUDE_SUBSTRINGS):
        return True
    return bool(name_tokens(tail) & _EXCLUDE_TOKENS)


def _looks_numeric(doc: str, signature_text: str) -> bool:
    blob = (doc[:1500] + " " + signature_text).lower()
    return any(h in blob for h in _NUMERIC_HINTS)


def _split_top_level(text: str) -> list[str]:
    parts, depth, quote, buf = [], 0, "", []
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


_DOC_SIG_RE = re.compile(r"^\s*[\w.]*\s*\((.*)\)\s*$")


def parse_doc_signature(first_line: str) -> list[dict[str, Any]]:
    raw = (first_line or "").strip()
    bracketed: set[str] = set()
    if "[" in raw:
        tail = raw[raw.index("["):]
        bracketed = {t.split("=")[0].strip(" ,[]()\t")
                     for t in tail.replace("[", " ").replace("]", " ").split(",")}
        bracketed.discard("")
    text = raw.replace("[, ", ", ").replace("[", "").replace("]", "")
    match = _DOC_SIG_RE.match(text)
    if not match:
        return []
    rows: list[dict[str, Any]] = []
    positional_only_upto: int | None = None
    keyword_only_from: int | None = None
    for token in _split_top_level(match.group(1)):
        if token == "/":
            positional_only_upto = len(rows)
            continue
        if token == "*" or token.startswith("**") or token.startswith("*"):
            if token in ("*",) or token.startswith("*") and not token.startswith("**"):
                keyword_only_from = len(rows)
            continue
        name, _, default_text = token.partition("=")
        name, _, annotation = name.partition(":")
        name, annotation = name.strip(), annotation.strip()
        if not name.isidentifier():
            continue
        has_default = bool(default_text.strip())
        default: Any = None
        if has_default:
            try:
                default = ast.literal_eval(default_text.strip())
            except (ValueError, SyntaxError):
                default = default_text.strip()
        annotated_optional = "optional" in annotation.lower() or "none" in annotation.lower()
        optional = has_default or annotated_optional or name in bracketed
        rows.append({"name": name, "position": len(rows), "required": not optional,
                     "has_default": optional, "default": default,
                     "call_kind": CALL_POSITIONAL_OR_KEYWORD,
                     "annotation": annotation})
    for idx, row in enumerate(rows):
        if positional_only_upto is not None and idx < positional_only_upto:
            row["call_kind"] = CALL_POSITIONAL
        elif keyword_only_from is not None and idx >= keyword_only_from:
            row["call_kind"] = CALL_KEYWORD
    return rows


def _is_opaque(sig: inspect.Signature | None) -> bool:
    if sig is None:
        return True
    kinds = {p.kind for p in sig.parameters.values()}
    return not kinds or kinds <= {inspect.Parameter.VAR_POSITIONAL,
                                  inspect.Parameter.VAR_KEYWORD}


_DOC_SIG_LINE_RE = re.compile(r"^\s*[\w.]+\s*\(.*\)\s*$")


def doc_signature_line(doc: str) -> str:
    for raw in (doc or "").strip().splitlines()[:6]:
        line = raw.strip().rstrip(":")
        if "->" in line:
            line = line.split("->")[0].strip()
        cleaned = line.replace("[, ", ", ").replace("[", "").replace("]", "")
        if "(" in cleaned and _DOC_SIG_LINE_RE.match(cleaned):
            return line
    return ""


def describe(obj: Any) -> tuple[str, inspect.Signature | None]:
    doc = inspect.getdoc(obj) or ""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        sig = None
    if not _is_opaque(sig):
        return str(sig), sig
    line = doc_signature_line(doc)
    if line:
        return line[line.index("("):line.rindex(")") + 1], None
    return (str(sig), sig) if sig is not None else ("", None)


def collect_api(api: str) -> ApiRecord | None:
    obj = resolve(api)
    if obj is None or not callable(obj):
        return None
    doc = inspect.getdoc(obj) or ""
    sig_text, sig = describe(obj)
    module = api.rsplit(".", 1)[0]
    return ApiRecord(api=api, obj=obj, doc=doc, signature_text=sig_text,
                     signature=sig, module=module)


def resolve(path: str) -> Any:
    parts = path.split(".")
    try:
        obj = importlib.import_module(parts[0])
    except Exception:
        return None
    idx = 1
    while idx < len(parts):
        candidate = ".".join(parts[: idx + 1])
        try:
            spec = importlib.util.find_spec(candidate)
        except (ImportError, ValueError, AttributeError):
            spec = None
        if spec is None:
            break
        try:
            obj = importlib.import_module(candidate)
        except Exception:
            break
        idx += 1
    for name in parts[idx:]:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def collect(lib: str, *, max_depth: int = 3, limit: int | None = None,
            numeric_only: bool = True, include: str = "",
            exclude_filters: bool = True) -> list[ApiRecord]:
    seen: set[int] = set()
    out: list[ApiRecord] = []
    pattern = re.compile(include) if include else None

    for mod_name in _iter_modules(lib, max_depth):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name, obj in sorted(vars(mod).items()):
            if name.startswith("_") or not callable(obj):
                continue
            if isinstance(obj, type):
                continue
            owner = getattr(obj, "__module__", "") or ""
            if owner and not owner.split(".")[0] == lib.split(".")[0]:
                continue
            qualname = f"{mod_name}.{name}"
            if pattern and not pattern.search(qualname):
                continue
            if exclude_filters and _is_excluded_name(qualname):
                continue
            if id(obj) in seen:
                continue
            doc = inspect.getdoc(obj) or ""
            sig_text, sig = describe(obj)
            if not doc or not sig_text:
                continue
            if sig is not None and not sig.parameters:
                continue
            if numeric_only and not _looks_numeric(doc, sig_text):
                continue
            seen.add(id(obj))
            out.append(ApiRecord(api=qualname, obj=obj, doc=doc,
                                 signature_text=sig_text, signature=sig,
                                 module=mod_name))
            if limit and len(out) >= limit:
                return out
    return out


def signature_params(record: ApiRecord) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if _is_opaque(record.signature):
        return parse_doc_signature(record.signature_text
                                   or doc_signature_line(record.doc))
    if record.signature is None:
        return rows
    for pos, (name, par) in enumerate(record.signature.parameters.items()):
        if name in ("self", "cls"):
            continue
        has_default = par.default is not inspect.Parameter.empty
        rows.append({
            "name": name,
            "position": pos,
            "required": not has_default and par.kind not in (
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD),
            "has_default": has_default,
            "default": par.default if has_default else None,
            "call_kind": _call_kind(par),
            "annotation": "" if par.annotation is inspect.Parameter.empty
                          else str(par.annotation),
        })
    return rows
