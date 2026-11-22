from __future__ import annotations

import json
import os
import statistics
from dataclasses import replace

from .runner import run


def _stats(values):
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values), "runs": values}


def common_apis(paths):
    cohorts = []
    for path in paths:
        with open(path) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        cohorts.append({row["api"] for row in rows if row.get("executed", 0) > 0})
    return sorted(set.intersection(*cohorts)) if cohorts else []


def run_repeated(config, repeats=5, variants=None, budgets=None):
    if repeats < 1:
        raise ValueError("repeat count must be positive")
    variants = variants or [config.variant]
    budgets = budgets or [config.budget_sec]
    os.makedirs(config.out_dir, exist_ok=True)
    cache_dir = config.spec_dir or os.path.join(config.out_dir, "specs")
    results = []
    for variant in variants:
        for budget in budgets:
            summaries = []
            signatures = set()
            for repeat in range(repeats):
                directory = os.path.join(config.out_dir, variant, f"{budget:g}s", str(repeat + 1))
                if os.path.exists(os.path.join(directory, "summary.json")):
                    raise FileExistsError(f"experiment output already exists: {directory}")
                trial = replace(config, variant=variant, budget_sec=budget, out_dir=directory,
                                seed=config.seed + repeat, spec_dir=cache_dir,
                                native_coverage=config.native_coverage.format(
                                    variant=variant, budget=budget, repeat=repeat + 1))
                summary = run(trial)
                summaries.append(summary)
                with open(os.path.join(directory, "findings.jsonl")) as handle:
                    for line in handle:
                        finding = json.loads(line)
                        signature = (finding["api"], finding["oracle"], finding["kind"])
                        signatures.add(signature)
            result = {"variant": variant, "budget_sec": budget, "repeats": repeats,
                      "valid_generation_rate_pct": _stats([s["valid_generation_rate_pct"] for s in summaries]),
                      "apis_covered": _stats([s["apis_covered"] for s in summaries]),
                      "candidate_findings": _stats([s["findings"] for s in summaries]),
                      "candidate_api_failure_groups_union": len(signatures),
                      "preprocessing_sec": sum(s["preprocessing_sec"] for s in summaries)}
            for cohort in ("R+", "R-"):
                result["vgr_" + cohort] = _stats(
                    [s["vgr_by_relationship"][cohort]["vgr_pct"] for s in summaries])
            if all("coverage" in s for s in summaries):
                result["coverage"] = _stats([s["coverage"]["total_lines"] for s in summaries])
            results.append(result)
    with open(os.path.join(config.out_dir, "experiments.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    return results
