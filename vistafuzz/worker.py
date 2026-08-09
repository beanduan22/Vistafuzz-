from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
import traceback
from typing import Any

from . import generation, oracles
from .collector import collect_api, resolve
from .constraints import DATA_LIKE_NAMES, envelope_for, initialize
from .materialize import materialize
from .models import ApiSpec, Value


class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout()


class EventLog:

    def __init__(self, path: str) -> None:
        self.handle = open(path, "a", buffering=1)

    def write(self, event: dict[str, Any]) -> None:
        self.handle.write(json.dumps(event, default=repr) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        try:
            self.handle.close()
        except Exception:
            pass


def _describe(args: dict[str, Value]) -> dict[str, Any]:
    return {name: value.describe() for name, value in args.items()}


def _finding_key(oracle: str, kind: str, args: dict[str, Any]) -> tuple:
    signature = tuple(sorted(
        (name, described.get("recipe", {}).get("pattern", "const"),
         described.get("dtype"), len(described.get("shape") or []))
        for name, described in args.items() if not described.get("omitted")))
    return (oracle, kind, signature)


def _receiver_args(spec: ApiSpec, rng: random.Random,
                   seed: int) -> dict[str, Value]:
    args: dict[str, Value] = {}
    for idx, param in enumerate(spec.receiver_params):
        if param.call_kind in ("var_positional", "var_keyword"):
            continue
        optional = not param.required and param.has_default
        if optional and param.name.lower() not in DATA_LIKE_NAMES and rng.random() > 0.35:
            args[param.name] = Value(kind="default", payload=param.default, omitted=True,
                                     recipe={"form": "const", "value": param.default},
                                     strategy="default")
            continue
        env = envelope_for(param)
        variants = generation.derive_variants(param, args)
        if variants:
            strategy = generation.select_strategy(rng, variants)
            value = generation.generate(param, args, variants, strategy, rng)
        else:
            value = generation.make_value(param, env, args, seed=seed * 131 + idx)
        if value is None:
            value = Value(kind="default", payload=param.default, omitted=True,
                          recipe={"form": "const", "value": param.default})
        args[param.name] = value
    return args


def _receiver_typed(param: Any, class_name: str, signature_text: str) -> bool:
    short = class_name.rsplit(".", 1)[-1].lower()
    if len(short) < 3:
        return False
    if short in (param.doc or "").lower():
        return True
    return f"{param.name}: {short}" in (signature_text or "").lower()


RECEIVER_ATTEMPTS = 6


def _build_receiver(cls: Any, spec: ApiSpec, args: dict[str, Value]) -> Any:
    positional, keyword = materialize(spec.receiver_path, spec.receiver_params, args)
    try:
        return cls(*positional, **keyword)
    except Exception:
        return None


def _make_receiver(cls: Any, spec: ApiSpec, rng: random.Random,
                   seed: int) -> tuple[Any, dict[str, Value]]:
    for attempt in range(RECEIVER_ATTEMPTS):
        args = _receiver_args(spec, rng, seed + attempt * 7919)
        try:
            instance = _build_receiver(cls, spec, args)
        except Exception:
            instance = None
        if instance is not None:
            return instance, args
    try:
        return cls(), {}
    except Exception:
        return None, {}


def _preview(obj: Any, limit: int = 160) -> str:
    try:
        text = repr(obj)
    except Exception:
        text = f"<unrepresentable {type(obj).__name__}>"
    return text[:limit]


def run_job(job: dict[str, Any]) -> int:
    api = job["api"]
    spec = ApiSpec.from_json(job["spec"])
    budget = float(job.get("budget_sec", 60.0))
    case_timeout = int(job.get("case_timeout", 10))
    max_cases = int(job.get("max_cases", 0)) or None
    rng = random.Random(job.get("seed", 0))
    log = EventLog(job["events_path"])

    record = collect_api(api)
    if record is None:
        log.write({"event": "skip", "api": api, "reason": "api not resolvable at runtime"})
        log.close()
        return 0
    target = record.obj
    method_name = api.rsplit(".", 1)[1] if spec.is_method else ""
    receiver_cls = resolve(spec.receiver_path) if spec.is_method else None
    if spec.is_method and receiver_cls is None:
        log.write({"event": "skip", "api": api,
                   "reason": f"receiver class {spec.receiver_path} not resolvable"})
        log.close()
        return 0

    seed_args = initialize(spec.params, seed=job.get("seed", 0))
    if not seed_args and any(p.required for p in spec.params) and not spec.is_method:
        log.write({"event": "skip", "api": api,
                   "reason": "no supported value for a required parameter"})
        log.close()
        return 0

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)

    max_findings = int(job.get("max_findings", 5))
    seen_findings: set[tuple] = set()
    oracle_hits = 0
    deadline = time.time() + budget
    cases = 0
    log.write({"event": "api_start", "api": api, "budget_sec": budget,
               "params": [p.name for p in spec.params]})

    while time.time() < deadline and (max_cases is None or cases < max_cases):
        cases += 1
        new_args = dict(seed_args)
        for spec_param in spec.params:
            if spec_param.call_kind in ("var_positional", "var_keyword"):
                continue
            if not spec_param.required and spec_param.has_default:
                include = 0.5 if spec_param.doc else 0.25
                if rng.random() > include:
                    new_args[spec_param.name] = Value(
                        kind="default", payload=spec_param.default, omitted=True,
                        recipe={"form": "const", "value": spec_param.default},
                        strategy="default")
                    continue
            variants = generation.derive_variants(spec_param, new_args)
            if not variants:
                continue
            strategy = generation.select_strategy(rng, variants)
            value = generation.generate(spec_param, new_args, variants, strategy, rng)
            if value is not None:
                new_args[spec_param.name] = value
        generation.refresh_dependents(spec.params, new_args, rng)

        try:
            positional, keyword = materialize(api, spec.params, new_args)
        except Exception as exc:
            log.write({"event": "result", "case": cases, "status": "unmaterializable",
                       "detail": f"{type(exc).__name__}: {exc}"})
            continue

        receiver_described: dict[str, Any] = {}
        bound = target
        if spec.is_method:
            instance, recv_args = _make_receiver(receiver_cls, spec, rng, cases)
            receiver_described = _describe(recv_args)
            if instance is None:
                log.write({"event": "result", "case": cases, "status": "rejected",
                           "detail": f"could not construct {spec.receiver_path}"})
                continue
            bound = getattr(instance, method_name, None)
            if not callable(bound):
                log.write({"event": "result", "case": cases, "status": "rejected",
                           "detail": f"{method_name} unavailable on the constructed receiver"})
                continue
            for param in spec.params:
                if param.name not in new_args or new_args[param.name].omitted:
                    continue
                typed = _receiver_typed(param, spec.receiver_path, spec.signature_text)
                if not typed and not (param.required and rng.random() < 0.5):
                    continue
                peer, peer_args = _make_receiver(receiver_cls, spec, rng,
                                                 cases + 104729)
                if peer is None:
                    continue
                new_args[param.name] = Value(
                    kind="receiver", payload=peer,
                    recipe={"form": "receiver", "path": spec.receiver_path,
                            "args": {k: v.describe() for k, v in peer_args.items()}},
                    strategy="receiver")
            try:
                positional, keyword = materialize(api, spec.params, new_args)
            except Exception as exc:
                log.write({"event": "result", "case": cases, "status": "unmaterializable",
                           "detail": f"{type(exc).__name__}: {exc}"})
                continue

        described = _describe(new_args)
        log.write({"event": "start", "case": cases, "args": described,
                   "receiver_args": receiver_described,
                   "n_positional": len(positional), "keywords": sorted(keyword)})

        finite_inputs = oracles.inputs_are_finite(list(positional) + list(keyword.values()))
        started = time.time()
        if hasattr(signal, "SIGALRM"):
            signal.alarm(case_timeout)
        try:
            result = bound(*positional, **keyword)
            status, detail = "executed", ""
        except _Timeout:
            oracle_hits += 1
            key = _finding_key("crash", "hang", described)
            report = key not in seen_findings and len(seen_findings) < max_findings
            seen_findings.add(key)
            event = {"event": "result", "case": cases, "status": "hang",
                     "detail": f"exceeded {case_timeout}s"}
            if report:
                event.update({"oracle": "crash", "kind": "hang", "args": described,
                              "receiver_args": receiver_described})
            log.write(event)
            continue
        except BaseException as exc:
            exc_type = type(exc).__name__
            message = str(exc)[:400]
            verdict = oracles.classify_exception(exc_type, message)
            status = "rejected" if verdict == "rejected" else "unexpected_exception"
            detail = f"{exc_type}: {message}"
            if verdict != "rejected":
                detail += "\n" + "".join(traceback.format_exception_only(type(exc), exc))[:200]
            result = None
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        elapsed = round(time.time() - started, 4)

        event: dict[str, Any] = {"event": "result", "case": cases, "status": status,
                                 "elapsed_sec": elapsed}
        if status == "unexpected_exception":
            oracle_hits += 1
            event["detail"] = detail
            key = _finding_key("crash", "exception", described)
            if key not in seen_findings and len(seen_findings) < max_findings:
                event.update({"oracle": "crash", "kind": "exception", "args": described,
                              "receiver_args": receiver_described})
            seen_findings.add(key)
        elif status == "rejected":
            event["detail"] = detail
        elif finite_inputs:
            flagged, why = oracles.check_nan(result)
            if flagged:
                oracle_hits += 1
                event["detail"] = why
                key = _finding_key("nan", "non_finite_output", described)
                if key not in seen_findings and len(seen_findings) < max_findings:
                    event.update({"oracle": "nan", "kind": "non_finite_output",
                                  "args": described, "result_preview": _preview(result)})
                seen_findings.add(key)
        log.write(event)

    log.write({"event": "api_done", "api": api, "cases": cases,
               "oracle_hits": oracle_hits, "unique_findings": len(seen_findings)})
    log.close()
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m vistafuzz.worker <job.json>", file=sys.stderr)
        return 2
    with open(argv[1]) as handle:
        job = json.load(handle)
    return run_job(job)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
