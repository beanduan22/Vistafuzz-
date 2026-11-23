set -euo pipefail

task_python="${1:-python}"
task_output="${2:-runs/oneflow-smoke.json}"
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_native_libs="$("$task_python" - <<'PY'
from pathlib import Path
import os
import sysconfig

site = Path(sysconfig.get_path('purelib'))
print(os.pathsep.join(str(path) for path in (
    site / 'nvidia' / 'cudnn' / 'lib', site / 'nvidia' / 'cublas' / 'lib'
) if path.is_dir()))
PY
)"

if [[ -n "$task_native_libs" ]]; then
    export LD_LIBRARY_PATH="${task_native_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${VISTAFUZZ_ONEFLOW_MASTER_PORT:-29643}"
export WORLD_SIZE=1
export RANK=-1
export LOCAL_RANK=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec "$task_python" "$task_root/scripts/check_runtime.py" --lib oneflow --out "$task_output"
