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
from .collector import collect_api
from .constraints import initialize
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

    seed_args = initialize(spec.params, seed=job.get("seed", 0))
    if not seed_args and any(p.required for p in spec.params):
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

        described = _describe(new_args)
        log.write({"event": "start", "case": cases, "args": described,
                   "n_positional": len(positional), "keywords": sorted(keyword)})

        finite_inputs = oracles.inputs_are_finite(list(positional) + list(keyword.values()))
        started = time.time()
        if hasattr(signal, "SIGALRM"):
            signal.alarm(case_timeout)
        try:
            result = target(*positional, **keyword)
            status, detail = "executed", ""
        except _Timeout:
            oracle_hits += 1
            key = _finding_key("crash", "hang", described)
            report = key not in seen_findings and len(seen_findings) < max_findings
            seen_findings.add(key)
            event = {"event": "result", "case": cases, "status": "hang",
                     "detail": f"exceeded {case_timeout}s"}
            if report:
                event.update({"oracle": "crash", "kind": "hang", "args": described})
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
                event.update({"oracle": "crash", "kind": "exception", "args": described})
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
