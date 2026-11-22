from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import replace

from .runner import run


def _validator(path, base):
    file_name, separator, function = path.rpartition(":")
    if not separator:
        file_name, function = path, "matches"
    file_name = os.path.join(base, file_name) if not os.path.isabs(file_name) else file_name
    spec = importlib.util.spec_from_file_location("vistafuzz_bug_validator", file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, function)
    if not callable(fn):
        raise ValueError(f"invalid benchmark validator: {path}")
    return fn


def run_benchmark(config, manifest_path, repeats=5):
    with open(manifest_path) as handle:
        entries = json.load(handle)
    entries = entries["bugs"] if isinstance(entries, dict) else entries
    identifiers = [entry["id"] for entry in entries]
    if len(set(identifiers)) != len(entries):
        raise ValueError("benchmark bug identifiers must be unique")
    validators = {}
    for entry in entries:
        if entry["oracle"] not in ("crash", "nan"):
            raise ValueError(f"unsupported oracle for {entry['id']}")
        validators[entry["id"]] = _validator(entry["validator"], os.path.dirname(manifest_path))
    results = {entry["id"]: {"api": entry["api"], "oracle": entry["oracle"],
                            "supported_runs": [], "detected_runs": []} for entry in entries}
    cache = config.spec_dir or os.path.join(config.out_dir, "specs")
    for index in range(repeats):
        out_dir = os.path.join(config.out_dir, str(index + 1))
        if os.path.exists(os.path.join(out_dir, "summary.json")):
            raise FileExistsError(out_dir)
        trial = replace(config, out_dir=out_dir, seed=config.seed + index,
                        apis=list(dict.fromkeys(entry["api"] for entry in entries)), spec_dir=cache)
        run(trial)
        with open(os.path.join(out_dir, "apis.jsonl")) as handle:
            supported = {row["api"] for line in handle if (row := json.loads(line)).get("executed", 0)}
        with open(os.path.join(out_dir, "findings.jsonl")) as handle:
            findings = [json.loads(line) for line in handle if line.strip()]
        for entry in entries:
            result = results[entry["id"]]
            if entry["api"] in supported:
                result["supported_runs"].append(index + 1)
            candidates = [f for f in findings
                          if f["api"] == entry["api"] and f["oracle"] == entry["oracle"]]
            if any(validators[entry["id"]](finding) for finding in candidates):
                result["detected_runs"].append(index + 1)
    summary = {"bugs": results, "total": len(entries), "repeats": repeats,
               "detected": sum(bool(result["detected_runs"]) for result in results.values())}
    summary["detected_by_oracle"] = {
        oracle: sum(bool(result["detected_runs"]) and result["oracle"] == oracle
                    for result in results.values()) for oracle in ("crash", "nan")}
    with open(os.path.join(config.out_dir, "benchmark.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary
