#!/usr/bin/env bash
set -euo pipefail

LIBRARY="${1:-numpy}"
BUDGET="${2:-60}"
REPEATS="${3:-5}"
OUT_ROOT="${OUT_ROOT:-runs}"
PY="${PY:-python}"

"$PY" -m vistafuzz.cli fuzz --lib "$LIBRARY" --out "$OUT_ROOT/$LIBRARY" \
    --spec-dir "$OUT_ROOT/specs" --budget "$BUDGET" --repeat "$REPEATS" --coverage
