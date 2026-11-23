from __future__ import annotations

import inspect
import json
from dataclasses import asdict

from . import oracles, synth
from . import native_base as base
from .native import get_runtime
from .models import Finding

_RUNTIME = """
def _native(recipe, root):
    form = recipe.get("form")
    if form == "receiver":
        values = recipe.get("args", {})
        entries = [dict(v, name=k) for k, v in values.items()]
        return _build_receiver(recipe["path"], entries, root)
    if form == "sequence":
        items = [_native(item, root) for item in recipe.get("items", [])]
        return tuple(items) if recipe.get("as_tuple") else items
    if form == "dict":
        return {k: _native(v, root) for k, v in recipe.get("items", {}).items()}
    if recipe.get("as_dtype"):
        return RUNTIME.dtype(recipe["value"])
    value = build_value(recipe)
    if form in ("array", "literal_array"):
        return RUNTIME.array(value, recipe.get("dtype"))
    return value


def _build_args(entries, root):
    positional, keyword = [], {}
    for entry in entries:
        if entry.get("omitted"):
            continue
        value = _native(entry.get("recipe", {}), root)
        if entry.get("by_keyword"):
            keyword[entry["name"]] = value
        else:
            positional.append(value)
    return positional, keyword


def _build_receiver(path, entries, root):
    cls = resolve_object(path)
    positional, keyword = _build_args(entries, root)
    return cls(*positional, **keyword)


def _timeout(_signum, _frame):
    raise TimeoutError("invocation exceeded its timeout")


def main():
    import signal

    root = API.split(".", 1)[0]
    positional, keyword = _build_args(ARGS, root)
    if RECEIVER_PATH:
        receiver = _build_receiver(RECEIVER_PATH, RECEIVER_ARGS, root)
        print(f"receiver: {RECEIVER_PATH} -> {receiver!r}"[:200])
        fn = getattr(receiver, API.rsplit(".", 1)[1])
    else:
        fn = resolve_object(API)
    print(f"calling {API} with {len(positional)} positional and {sorted(keyword)} keyword args")
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(30)
    try:
        result = fn(*positional, **keyword)
    except BaseException as exc:
        verdict = classify_exception(type(exc).__name__, str(exc), CONSTRAINTS_SATISFIED)
        if ORACLE == "crash" and (verdict == "unexpected" or KIND == "hang" and isinstance(exc, TimeoutError)):
            print(f"BUG REPRODUCES: {type(exc).__name__}: {exc}")
            return 1
        if ORACLE == "crash" and verdict == "rejected":
            print(f"input rejected: {type(exc).__name__}: {exc}")
            return 0
        raise
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    if ORACLE == "crash":
        print("no crash: call returned", type(result).__name__)
        return 0
    flagged, detail = check_nan(result, list(positional) + list(keyword.values()), DOC, api=API)
    print(("BUG REPRODUCES: " if flagged else "OK: ") + detail)
    return int(flagged)


if __name__ == "__main__":
    sys.exit(main())
"""


def _entries(order, args):
    return [dict(args[name], name=name) for name in order
            if name in args and not args[name].get("omitted")]


def render(finding: Finding):
    sources = [inspect.getsource(module).replace("from __future__ import annotations\n", "")
               for module in (synth, oracles, base)]
    runtime_instance = get_runtime(finding.api)
    runtime = asdict(runtime_instance)
    runtime_class = type(runtime_instance)
    if runtime_class is not base.Runtime:
        sources.append(inspect.getsource(runtime_class))
    constants = {
        "API": finding.api, "ARGS": _entries(finding.order, finding.args),
        "ORACLE": finding.oracle, "KIND": finding.kind, "DOC": finding.doc,
        "CONSTRAINTS_SATISFIED": finding.constraints_satisfied,
        "RECEIVER_PATH": finding.receiver_path,
        "RECEIVER_ARGS": _entries(finding.receiver_order, finding.receiver_args),
    }
    assignments = "\n".join(f"{name} = {value!r}" for name, value in constants.items())
    return ("\n".join(["from __future__ import annotations", "import sys",
                       'nan = float("nan")', 'inf = float("inf")', *sources])
            + "\nRUNTIME = " + runtime_class.__name__ + "(**" + repr(runtime) + ")\n"
            + assignments + "\n" + _RUNTIME)


def write(finding, path):
    with open(path, "w") as handle:
        handle.write(render(finding))


def slug(api, index):
    return f"{api.replace('.', '_')}__{index:03d}.py"


def dump_finding(finding, path):
    with open(path, "w") as handle:
        json.dump(finding.to_json(), handle, indent=2, default=repr)
