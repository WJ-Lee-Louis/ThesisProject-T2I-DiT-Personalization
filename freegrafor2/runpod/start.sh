#!/usr/bin/env bash
set -euo pipefail
cd /workspace/freegraftor
source env.sh
mkdir -p results/seed42_512/logs
/usr/bin/python3.11 - <<'PY'
from pathlib import Path
import subprocess
root=Path('/workspace/freegraftor')
with (root/'results/seed42_512/logs/controller.log').open('a') as log:
    guarded=(root/'recovery/completion-guard.json').exists()
    script='completion_guard.py' if guarded else 'benchmark.py'
    python='/opt/freegraftor-runtime/generation-local/bin/python' if guarded else '/usr/bin/python3.11'
    args=[python,'-u',str(root/'scripts'/script)]+([] if guarded else ['run'])
    p=subprocess.Popen(args,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
(root/'results/seed42_512/controller.pid').write_text(str(p.pid)+'\n')
print(f'Controller started: {p.pid} (generation, subject-mask recovery, then metrics; two A100s).')
PY
