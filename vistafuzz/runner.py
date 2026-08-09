from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import llm as llm_mod
from . import reproducer
from .collector import ApiRecord, collect, collect_api
from .extraction import extract
from .models import ApiSpec, Finding
from .oracles import crash_verdict, is_resource_exhaustion
from .validation import validate


@dataclass
class RunConfig:
    lib: str
    out_dir: str
    budget_sec: float = 60.0
    case_timeout: int = 10
    max_apis: int | None = None
    max_cases: int | None = None
    max_findings_per_api: int = 5
    extractor: str = "llm"
    seed: int = 0
    max_depth: int = 3
    include: str = ""
    numeric_only: bool = True
    methods: bool = True
    apis: list[str] = field(default_factory=list)
    save_reproducers: bool = True
    llm: llm_mod.LLMConfig | None = None


@dataclass
class ApiOutcome:
    api: str
    status: str
    reason: str = ""
    cases: int = 0
    executed: int = 0
    rejected: int = 0
    oracle_hits: int = 0
    findings: list[Finding] = field(default_factory=list)
    extractor: str = ""

    @property
    def srg(self) -> float:
        total = self.executed + self.rejected
        return (100.0 * self.executed / total) if total else 0.0


def _events(path: str) -> Iterable[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _mark_keyword_flags(args: dict[str, Any], params: list[Any]) -> dict[str, Any]:
    out = {}
    keyword_from_here = False
    ordered = sorted(params, key=lambda p: (p.position if p.position is not None else 1 << 20))
    for param in ordered:
        described = args.get(param.name)
        if described is None or described.get("omitted"):
            if param.call_kind in ("positional", "positional_or_keyword"):
                keyword_from_here = True
            continue
        described = dict(described)
        described["by_keyword"] = bool(param.call_kind == "keyword" or keyword_from_here)
        out[param.name] = described
    return out


def _order(params: list[Any]) -> list[str]:
    return [p.name for p in sorted(params,
                                   key=lambda p: (p.position if p.position is not None else 1 << 20))]


def _finding(api: str, spec: ApiSpec, row: dict[str, Any]) -> Finding:
    return Finding(
        api=api, oracle=row.get("oracle", ""), kind=row.get("kind", ""),
        detail=row.get("detail", ""),
        args=_mark_keyword_flags(row.get("args", {}), spec.params),
        order=_order(spec.params), strategy=row.get("strategy", ""),
        receiver_path=spec.receiver_path,
        receiver_args=_mark_keyword_flags(row.get("receiver_args", {}),
                                          spec.receiver_params),
        receiver_order=_order(spec.receiver_params))


def run_api(record: ApiRecord, config: RunConfig, work_dir: str) -> ApiOutcome:
    spec = extract(record, extractor=config.extractor, config=config.llm)
    result = validate(spec, record)
    if not result.ok or result.spec is None:
        return ApiOutcome(record.api, "excluded", result.reason, extractor=spec.extractor)
    spec = result.spec

    api_slug = record.api.replace(".", "_")
    events_path = os.path.join(work_dir, f"{api_slug}.events.jsonl")
    job_path = os.path.join(work_dir, f"{api_slug}.job.json")
    with open(job_path, "w") as handle:
        json.dump({"api": record.api, "spec": spec.to_json(),
                   "budget_sec": config.budget_sec, "case_timeout": config.case_timeout,
                   "max_cases": config.max_cases or 0, "seed": config.seed,
                   "max_findings": config.max_findings_per_api,
                   "events_path": events_path}, handle, default=repr)

    outcome = ApiOutcome(record.api, "tested", extractor=spec.extractor)
    hard_limit = config.budget_sec + config.case_timeout + 60
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONWARNINGS": "ignore",
           "PYTHONPATH": os.pathsep.join(
               [package_parent] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
    timed_out = False
    returncode = 0
    stderr = ""
    try:
        proc = subprocess.run([sys.executable, "-m", "vistafuzz.worker", job_path],
                              capture_output=True, text=True, env=env,
                              timeout=hard_limit, check=False)
        returncode, stderr = proc.returncode, proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) \
            else (exc.stderr or "")

    rows = list(_events(events_path))
    pending_start: dict[str, Any] | None = None
    for row in rows:
        event = row.get("event")
        if event == "skip":
            outcome.status = "skipped"
            outcome.reason = row.get("reason", "")
        elif event == "start":
            pending_start = row
        elif event == "result":
            pending_start = None
            outcome.cases = max(outcome.cases, int(row.get("case", 0)))
            status = row.get("status")
            if status == "executed":
                outcome.executed += 1
            elif status == "rejected":
                outcome.rejected += 1
            oracle = row.get("oracle")
            if oracle:
                outcome.findings.append(_finding(record.api, spec, row))
        elif event == "api_done":
            outcome.cases = int(row.get("cases", outcome.cases))
            outcome.oracle_hits = int(row.get("oracle_hits", 0))

    verdict = crash_verdict(returncode,
                            -returncode if returncode and returncode < 0 else None,
                            timed_out=timed_out)
    if verdict and pending_start is not None and is_resource_exhaustion(stderr):
        pass
    elif verdict and pending_start is not None:
        kind, detail = verdict
        tail = stderr.strip().splitlines()[-3:]
        row = dict(pending_start)
        row.update({"oracle": "crash", "kind": kind,
                    "detail": detail + (("\n" + "\n".join(tail)) if tail else "")})
        outcome.findings.append(_finding(record.api, spec, row))
    elif verdict and pending_start is None and not rows:
        outcome.status = "error"
        outcome.reason = f"worker failed before any case ({verdict[1]})"
    return outcome


def run(config: RunConfig) -> dict[str, Any]:
    os.makedirs(config.out_dir, exist_ok=True)
    work_dir = os.path.join(config.out_dir, "work")
    repro_dir = os.path.join(config.out_dir, "reproducers")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(repro_dir, exist_ok=True)

    if config.apis:
        records = [r for r in (collect_api(a) for a in config.apis) if r is not None]
    else:
        records = collect(config.lib, max_depth=config.max_depth, limit=config.max_apis,
                          numeric_only=config.numeric_only, include=config.include,
                          methods=config.methods)

    started = time.time()
    outcomes: list[ApiOutcome] = []
    findings_path = os.path.join(config.out_dir, "findings.jsonl")
    n_findings = 0
    with open(findings_path, "w") as findings_file:
        for index, record in enumerate(records, 1):
            print(f"[{index}/{len(records)}] {record.api}", flush=True)
            try:
                outcome = run_api(record, config, work_dir)
            except Exception as exc:
                outcome = ApiOutcome(record.api, "error", f"{type(exc).__name__}: {exc}")
            outcomes.append(outcome)
            for finding in outcome.findings:
                findings_file.write(json.dumps(finding.to_json(), default=repr) + "\n")
                findings_file.flush()
                n_findings += 1
                if config.save_reproducers:
                    path = os.path.join(repro_dir, reproducer.slug(finding.api, n_findings))
                    try:
                        reproducer.write(finding, path)
                    except Exception as exc:
                        print(f"  (reproducer emission failed: {exc})", flush=True)
            if outcome.findings:
                kinds = ", ".join(sorted({f"{f.oracle}/{f.kind}" for f in outcome.findings}))
                print(f"    {len(outcome.findings)} finding(s): {kinds}", flush=True)

    tested = [o for o in outcomes if o.status == "tested"]
    executed = sum(o.executed for o in tested)
    rejected = sum(o.rejected for o in tested)
    summary = {
        "library": config.lib,
        "extractor": config.extractor,
        "budget_sec_per_api": config.budget_sec,
        "apis_discovered": len(records),
        "apis_method_based": len([r for r in records if r.is_method]),
        "apis_tested": len(tested),
        "apis_covered": len([o for o in tested if o.executed > 0]),
        "apis_excluded": len([o for o in outcomes if o.status == "excluded"]),
        "apis_skipped": len([o for o in outcomes if o.status == "skipped"]),
        "apis_error": len([o for o in outcomes if o.status == "error"]),
        "cases_generated": sum(o.cases for o in outcomes),
        "cases_executed": executed,
        "cases_rejected": rejected,
        "valid_generation_rate_pct": round(100.0 * executed / (executed + rejected), 2)
        if (executed + rejected) else 0.0,
        "findings": n_findings,
        "oracle_hits_total": sum(o.oracle_hits for o in outcomes),
        "findings_by_oracle": {
            "crash": sum(1 for o in outcomes for f in o.findings if f.oracle == "crash"),
            "nan": sum(1 for o in outcomes for f in o.findings if f.oracle == "nan"),
        },
        "elapsed_sec": round(time.time() - started, 1),
    }
    with open(os.path.join(config.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(config.out_dir, "apis.jsonl"), "w") as handle:
        for outcome in outcomes:
            handle.write(json.dumps({
                "api": outcome.api, "status": outcome.status, "reason": outcome.reason,
                "cases": outcome.cases, "executed": outcome.executed,
                "rejected": outcome.rejected, "srg_pct": round(outcome.srg, 2),
                "oracle_hits": outcome.oracle_hits,
                "findings": len(outcome.findings), "extractor": outcome.extractor,
            }) + "\n")
    return summary
