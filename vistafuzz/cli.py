from __future__ import annotations

import argparse
import json
import os
import sys

from . import llm as llm_mod
from .collector import collect, collect_api
from .extraction import extract
from .runner import RunConfig, run
from .validation import validate


def _llm_config(args: argparse.Namespace) -> llm_mod.LLMConfig:
    return llm_mod.LLMConfig.from_env(
        backend=getattr(args, "llm_backend", "") or None,
        model=getattr(args, "llm_model", "") or None,
        host=getattr(args, "llm_host", "") or None,
        base_url=getattr(args, "llm_base_url", "") or None,
    )


def _add_llm_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--extractor", choices=["llm", "signature"], default="llm",
                        help="llm: prompt the model (default); "
                             "signature: offline fallback, no LLM required")
    parser.add_argument("--llm-backend", default="", choices=["", "ollama", "openai"])
    parser.add_argument("--llm-model", default="", help="e.g. qwen2.5-coder:32b")
    parser.add_argument("--llm-host", default="", help="ollama host URL")
    parser.add_argument("--llm-base-url", default="", help="OpenAI-compatible base URL")


def cmd_apis(args: argparse.Namespace) -> int:
    records = collect(args.lib, max_depth=args.max_depth, limit=args.limit,
                      numeric_only=not args.all, include=args.include)
    for record in records:
        print(f"{record.api}{record.signature_text}")
    print(f"\n{len(records)} API(s) discovered in {args.lib}", file=sys.stderr)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    record = collect_api(args.api)
    if record is None:
        print(f"cannot resolve {args.api}", file=sys.stderr)
        return 2
    spec = extract(record, extractor=args.extractor, config=_llm_config(args))
    result = validate(spec, record)
    payload = {
        "validated": result.ok,
        "reason": result.reason,
        "dropped_parameters": result.dropped,
        "signature_corrections": result.corrected,
        "spec": (result.spec or spec).to_json(),
    }
    text = json.dumps(payload, indent=2, default=repr)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0 if result.ok else 1


def cmd_fuzz(args: argparse.Namespace) -> int:
    config = RunConfig(
        lib=args.lib or (args.api[0].split(".")[0] if args.api else ""),
        out_dir=args.out,
        budget_sec=args.budget,
        case_timeout=args.case_timeout,
        max_apis=args.max_apis,
        max_cases=args.max_cases,
        max_findings_per_api=args.max_findings_per_api,
        extractor=args.extractor,
        seed=args.seed,
        max_depth=args.max_depth,
        include=args.include,
        numeric_only=not args.all,
        apis=list(args.api or []),
        save_reproducers=not args.no_reproducers,
        llm=_llm_config(args),
    )
    if not config.lib:
        print("either --lib or --api is required", file=sys.stderr)
        return 2
    summary = run(config)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = os.path.join(args.out, "summary.json")
    if not os.path.exists(path):
        print(f"no summary.json under {args.out}", file=sys.stderr)
        return 2
    with open(path) as handle:
        print(json.dumps(json.load(handle), indent=2))
    findings = os.path.join(args.out, "findings.jsonl")
    if os.path.exists(findings):
        with open(findings) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for row in rows[:args.limit or 20]:
            print(f"\n{row['oracle']}/{row['kind']}  {row['api']}\n  {row['detail'][:300]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vistafuzz",
                                     description="LLM-guided document-based fuzzing "
                                                 "for arbitrary Python libraries")
    sub = parser.add_subparsers(dest="command", required=True)

    apis = sub.add_parser("apis", help="discover documented numerical APIs")
    apis.add_argument("--lib", required=True)
    apis.add_argument("--max-depth", type=int, default=3)
    apis.add_argument("--limit", type=int, default=None)
    apis.add_argument("--include", default="", help="regex filter on the API path")
    apis.add_argument("--all", action="store_true", help="skip the numeric-API filter")
    apis.set_defaults(func=cmd_apis)

    ext = sub.add_parser("extract", help="extract and validate one API's ParamSpecs")
    ext.add_argument("--api", required=True)
    ext.add_argument("--out", default="")
    _add_llm_flags(ext)
    ext.set_defaults(func=cmd_extract)

    fuzz = sub.add_parser("fuzz", help="run the full workflow over a library")
    fuzz.add_argument("--lib", default="")
    fuzz.add_argument("--api", action="append", help="fuzz only this API (repeatable)")
    fuzz.add_argument("--out", required=True)
    fuzz.add_argument("--budget", type=float, default=60.0,
                      help="per-API testing budget in seconds (default 60)")
    fuzz.add_argument("--case-timeout", type=int, default=10)
    fuzz.add_argument("--max-apis", type=int, default=None)
    fuzz.add_argument("--max-cases", type=int, default=None,
                      help="cap the cases per API (default: bounded only by --budget)")
    fuzz.add_argument("--max-findings-per-api", type=int, default=5,
                      help="unique findings kept per API after deduplication")
    fuzz.add_argument("--seed", type=int, default=0)
    fuzz.add_argument("--max-depth", type=int, default=3)
    fuzz.add_argument("--include", default="")
    fuzz.add_argument("--all", action="store_true")
    fuzz.add_argument("--no-reproducers", action="store_true")
    _add_llm_flags(fuzz)
    fuzz.set_defaults(func=cmd_fuzz)

    report = sub.add_parser("report", help="print a finished run")
    report.add_argument("--out", required=True)
    report.add_argument("--limit", type=int, default=20)
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
