# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stop only the recorded sessions for one release run on the current host."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
run = sys.argv[1]
assert run.replace("-", "").replace("_", "").isalnum()
host = socket.gethostname().split(".")[0]
assert host in ("xb01-gpu-200b-0030", "xb01-gpu-200b-0024")
assert os.getuid() == PACKAGE.stat().st_uid == 1000
node = PACKAGE / "state/runs" / run / host


def members(rec):
    leader = Path("/proc") / str(rec["pid"])
    if leader.exists():
        assert leader.stat().st_uid == os.getuid()
        ticks = (leader / "stat").read_text().rsplit(")", 1)[1].split()[19]
        assert ticks == str(rec["start_ticks"]), "PID reused; refusing to signal"
        assert os.getpgid(rec["pid"]) == rec["pgid"]
    rows = subprocess.check_output(["ps", "-eo", "pid=,sid=,pgid=,uid="], text=True)
    result = []
    for row in rows.splitlines():
        pid, sid, pgid, uid = map(int, row.split())
        if sid == rec["pid"] and pgid == rec["pgid"]:
            assert uid == os.getuid()
            result.append(pid)
    return result


records = [
    json.loads((node / name).read_text())
    for name in ("process.json", "master-process.json")
    if (node / name).exists()
]
assert records, "No owned process records"
for rec in records:
    if members(rec):
        os.killpg(rec["pgid"], signal.SIGTERM)
deadline = time.monotonic() + 90
while time.monotonic() < deadline and any(members(r) for r in records):
    time.sleep(2)
for rec in records:
    if members(rec):
        os.killpg(rec["pgid"], signal.SIGKILL)
time.sleep(2)
left = [pid for rec in records for pid in members(rec)]
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
).strip()
result = {
    "time": time.time(),
    "host": host,
    "run": run,
    "owned_remaining": left,
    "gpu_compute_pids": apps,
}
(node / "stopped.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
assert not left, "Owned session cleanup incomplete"
