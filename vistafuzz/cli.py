from __future__ import annotations

import argparse
import json
import os
import sys

from . import llm as llm_mod
from .collector import collect, collect_api, has_runtime_signature
from .extraction import extract
from .generation import VARIANTS
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
                             "signature: explicit offline diagnostic, no LLM required")
    parser.add_argument("--llm-backend", default="", choices=["", "ollama", "openai"])
    parser.add_argument("--llm-model", default="", help="e.g. qwen2.5-coder:32b")
    parser.add_argument("--llm-host", default="", help="ollama host URL")
    parser.add_argument("--llm-base-url", default="", help="OpenAI-compatible base URL")
    parser.add_argument("--spec-dir", default="", help="persistent extraction cache")
    parser.add_argument("--prompt-input", choices=["doc+sig", "doc-only", "sig-only"], default="doc+sig")
    parser.add_argument("--signature-policy", choices=["runtime", "documented"], default="runtime",
                        help="runtime: require independently introspectable calling information; "
                             "documented: exploratory use of binding docstrings")


def cmd_apis(args: argparse.Namespace) -> int:
    records = collect(args.lib, max_depth=args.max_depth, limit=args.limit,
                      numeric_only=not args.all, include=args.include,
                      methods=args.methods and not args.no_methods)
    for record in records:
        marker = "  [method]" if record.is_method else ""
        print(f"{record.api}{record.signature_text}{marker}")
    print(f"\n{len(records)} API(s) discovered in {args.lib}", file=sys.stderr)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    record = collect_api(args.api)
    if record is None:
        print(f"cannot resolve {args.api}", file=sys.stderr)
        return 2
    if args.signature_policy == "runtime" and not has_runtime_signature(record):
        print(f"excluded {args.api}: no usable runtime signature", file=sys.stderr)
        return 1
    spec = extract(record, extractor=args.extractor, config=_llm_config(args),
                   cache_dir=args.spec_dir, prompt_input=args.prompt_input)
    result = validate(spec, record, signature_policy=args.signature_policy)
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
    apis = list(args.api or [])
    if args.api_list:
        with open(args.api_list) as handle:
            payload = handle.read()
        try:
            values = json.loads(payload)
            values = values.get("apis", []) if isinstance(values, dict) else values
        except json.JSONDecodeError:
            values = [line.strip() for line in payload.splitlines() if line.strip()]
        apis.extend(row["api"] if isinstance(row, dict) else row for row in values)
    config = RunConfig(
        lib=args.lib or (apis[0].split(".")[0] if apis else ""),
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
        methods=args.methods and not args.no_methods,
        apis=list(dict.fromkeys(apis)),
        save_reproducers=not args.no_reproducers,
        llm=_llm_config(args),
        spec_dir=args.spec_dir,
        variant=args.variant,
        prompt_input=args.prompt_input,
        python_coverage=args.coverage,
        native_coverage=args.native_coverage,
        native_build_dir=args.native_build_dir,
        generator=args.generator,
        signature_policy=args.signature_policy,
    )
    if not config.lib:
        print("either --lib or --api is required", file=sys.stderr)
        return 2
    if args.benchmark:
        from .benchmark import run_benchmark

        summary = run_benchmark(config, args.benchmark, args.repeat)
    elif args.repeat > 1 or args.variants or args.budgets:
        from .experiments import run_repeated

        summary = run_repeated(config, args.repeat,
                               args.variants.split(",") if args.variants else None,
                               [float(x) for x in args.budgets.split(",")] if args.budgets else None)
    else:
        summary = run(config)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_common(args):
    from .experiments import common_apis

    values = common_apis(args.inputs)
    with open(args.out, "w") as handle:
        json.dump(values, handle, indent=2)
    print(f"wrote {len(values)} common APIs to {args.out}")
    return 0


def cmd_bugs(args):
    from .issue_stats import markdown, summarize

    print(markdown(summarize(args.file)))
    return 0


def cmd_triage(args):
    from .triage import review_findings

    summary = review_findings(args.findings, args.decisions, args.out)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_adaptation(args):
    from .runtime_stats import measure

    print(json.dumps(measure(), indent=2))
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
    apis.add_argument("--no-methods", action="store_true",
                      help="only module-level functions; skip class methods")
    apis.add_argument("--methods", action="store_true", help="include receiver construction outside the paper protocol")
    apis.set_defaults(func=cmd_apis)

    ext = sub.add_parser("extract", help="extract and validate one API's ParamSpecs")
    ext.add_argument("--api", required=True)
    ext.add_argument("--out", default="")
    _add_llm_flags(ext)
    ext.set_defaults(func=cmd_extract)

    fuzz = sub.add_parser("fuzz", help="run the full workflow over a library")
    fuzz.add_argument("--lib", default="")
    fuzz.add_argument("--api", action="append", help="fuzz only this API (repeatable)")
    fuzz.add_argument("--api-list", default="", help="JSON or text manifest of target APIs")
    fuzz.add_argument("--benchmark", default="", help="known-bug manifest with per-bug validators")
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
    fuzz.add_argument("--no-methods", action="store_true",
                      help="only module-level functions; skip class methods")
    fuzz.add_argument("--no-reproducers", action="store_true")
    fuzz.add_argument("--methods", action="store_true", help="include receiver construction outside the paper protocol")
    fuzz.add_argument("--variant", choices=list(VARIANTS), default="full")
    fuzz.add_argument("--variants", default="", help="comma-separated ablation variants")
    fuzz.add_argument("--budgets", default="", help="comma-separated budgets in seconds")
    fuzz.add_argument("--repeat", type=int, default=1)
    fuzz.add_argument("--coverage", action="store_true", help="measure Python coverage in API workers")
    fuzz.add_argument("--native-coverage", default="", help="LCOV trace from an instrumented native build")
    fuzz.add_argument("--native-build-dir", default="", help="reset and collect GCOV counters through LCOV")
    fuzz.add_argument("--generator", choices=["paramspec", "llm-direct"], default="paramspec")
    _add_llm_flags(fuzz)
    fuzz.set_defaults(func=cmd_fuzz)

    report = sub.add_parser("report", help="print a finished run")
    report.add_argument("--out", required=True)
    report.add_argument("--limit", type=int, default=20)
    report.set_defaults(func=cmd_report)
    common = sub.add_parser("common", help="write the intersection of supported APIs")
    common.add_argument("inputs", nargs="+", help="per-tool apis.jsonl files")
    common.add_argument("--out", required=True)
    common.set_defaults(func=cmd_common)
    bugs = sub.add_parser("bugs", help="summarize the issue ledger using the paper's columns")
    bugs.add_argument("--file", default="bug_review.csv")
    bugs.set_defaults(func=cmd_bugs)
    triage = sub.add_parser("triage", help="record documentation-checked, minimized findings for reporting")
    triage.add_argument("--findings", required=True, help="campaign findings.jsonl")
    triage.add_argument("--decisions", required=True, help="manual review decisions in JSON")
    triage.add_argument("--out", required=True)
    triage.set_defaults(func=cmd_triage)
    adaptation = sub.add_parser("adaptation", help="measure dedicated runtime source lines")
    adaptation.set_defaults(func=cmd_adaptation)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
