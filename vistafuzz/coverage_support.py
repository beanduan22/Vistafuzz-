from __future__ import annotations

import os
import subprocess
import sys


def worker_command(job_path, library, coverage_dir=""):
    if not coverage_dir:
        return [sys.executable, "-m", "vistafuzz.worker", job_path]
    import coverage

    os.makedirs(coverage_dir, exist_ok=True)
    return [sys.executable, "-m", "coverage", "run", "--parallel-mode",
            "--data-file=" + os.path.join(coverage_dir, ".coverage"),
            "--source=" + library, "-m", "vistafuzz.worker", job_path]


def native_lines(path):
    covered = set()
    source = ""
    with open(path) as handle:
        for line in handle:
            if line.startswith("SF:"):
                source = line[3:].strip()
            elif line.startswith("DA:"):
                number, count, *_ = line[3:].strip().split(",")
                if int(count) > 0:
                    covered.add((source, int(number)))
    return len(covered)


def reset_native(build_dir):
    subprocess.run(["lcov", "--zerocounters", "--directory", build_dir], check=True,
                   capture_output=True, text=True)


def capture_native(build_dir, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    subprocess.run(["lcov", "--capture", "--directory", build_dir,
                    "--output-file", output], check=True, capture_output=True, text=True)
    return output


def summarize(coverage_dir="", native_file=""):
    result = {"python_measured": bool(coverage_dir), "native_measured": bool(native_file),
              "python_lines": 0, "native_lines": 0}
    if coverage_dir:
        import coverage

        cov = coverage.Coverage(data_file=os.path.join(coverage_dir, ".coverage"))
        cov.combine(keep=True)
        cov.save()
        data = cov.get_data()
        result["python_lines"] = sum(len(data.lines(path) or []) for path in data.measured_files())
        if data.measured_files():
            cov.json_report(outfile=os.path.join(coverage_dir, "python.json"))
    if native_file:
        result["native_lines"] = native_lines(native_file)
    result["total_lines"] = result["python_lines"] + result["native_lines"]
    return result
