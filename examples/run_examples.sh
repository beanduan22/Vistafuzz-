#!/usr/bin/env bash
set -euo pipefail

BUDGET="${1:-60}"
OUT_ROOT="${OUT_ROOT:-runs}"
PY="${PY:-python}"

run_lib () {
  local lib="$1"; shift
  echo "=== $lib (budget ${BUDGET}s per API) ==="
  "$PY" -m vistafuzz.cli fuzz --lib "$lib" --out "$OUT_ROOT/$lib" \
        --budget "$BUDGET" "$@" || echo "  ($lib skipped: not importable here)"
}

"$PY" -m vistafuzz.cli apis --lib numpy --limit 10

"$PY" -m vistafuzz.cli extract --api numpy.clip --out "$OUT_ROOT/specs/numpy.clip.json"
"$PY" -m vistafuzz.cli fuzz --api numpy.clip --out "$OUT_ROOT/numpy-clip" --budget "$BUDGET"

run_lib torch       --max-apis 200
run_lib tensorflow  --max-apis 200
run_lib jax         --max-apis 200
run_lib keras       --max-apis 200
run_lib paddle      --max-apis 200
run_lib oneflow     --max-apis 200
run_lib mindspore   --max-apis 200
run_lib chainer     --max-apis 200
run_lib numpy       --max-apis 200
run_lib scipy       --max-apis 200
run_lib sklearn     --max-apis 200
run_lib cv2         --max-apis 200

run_lib numpy --max-apis 20 --extractor signature

"$PY" -m vistafuzz.cli report --out "$OUT_ROOT/numpy"
