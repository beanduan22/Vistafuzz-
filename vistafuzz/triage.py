from __future__ import annotations

import json
from pathlib import Path


def review_findings(findings_path, decisions_path, out_dir):
    with open(findings_path) as handle:
        findings = [json.loads(line) for line in handle if line.strip()]
    decisions_path = Path(decisions_path)
    with decisions_path.open() as handle:
        payload = json.load(handle)
    decisions = payload.get("reviews", []) if isinstance(payload, dict) else payload
    if not isinstance(decisions, list):
        raise ValueError("triage decisions must be a list")
    reviewed = {}
    for decision in decisions:
        index = decision.get("finding_index")
        if type(index) is not int or not 1 <= index <= len(findings):
            raise ValueError(f"invalid finding_index: {index!r}")
        if index in reviewed:
            raise ValueError(f"duplicate review of finding {index}")
        finding = findings[index - 1]
        if decision.get("api") != finding["api"]:
            raise ValueError(f"API mismatch for finding {index}")
        status = decision.get("decision")
        if status not in ("reported", "excluded", "pending"):
            raise ValueError(f"invalid decision for finding {index}: {status!r}")
        reason = decision.get("reason")
        if status in ("reported", "excluded") and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError(f"review reason required for finding {index}")
        if status == "reported":
            for field in ("documented_input_valid", "behavior_violates_documentation",
                          "minimized", "reproduced"):
                if decision.get(field) is not True:
                    raise ValueError(f"finding {index} requires {field}=true before reporting")
            evidence = decision.get("documentation_evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(f"documentation evidence required for finding {index}")
            path = decision.get("minimized_reproducer")
            if not isinstance(path, str) or not path:
                raise ValueError(f"minimized reproducer required for finding {index}")
            path = Path(path)
            path = path if path.is_absolute() else decisions_path.parent / path
            if not path.is_file():
                raise ValueError(f"minimized reproducer does not exist for finding {index}: {path}")
            decision = {**decision, "minimized_reproducer": str(path.resolve())}
        reviewed[index] = decision

    rows = []
    for index, finding in enumerate(findings, 1):
        review = reviewed.get(index, {"finding_index": index, "decision": "pending"})
        rows.append({**finding, "finding_index": index, "review": review})
    summary = {"flagged": len(rows)}
    for status in ("reported", "excluded", "pending"):
        summary[status] = sum(row["review"]["decision"] == status for row in rows)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in (("reviewed_findings.jsonl", rows),
                          ("reports.jsonl", [r for r in rows if r["review"]["decision"] == "reported"])):
        with (destination / name).open("w") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
    with (destination / "triage.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary
