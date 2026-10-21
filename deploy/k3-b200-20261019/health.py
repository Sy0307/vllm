# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wait for the rank0 API; fail if its recorded server exits or is replaced."""

import json
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
host = socket.gethostname().split(".")[0]
assert host in ("xb01-gpu-200b-0030", "xb01-gpu-200b-0009")
run = sys.argv[1]
assert run.replace("-", "").replace("_", "").isalnum()
record = json.loads((PACKAGE / "state/runs" / run / host / "process.json").read_text())
deadline = time.monotonic() + 1800
while time.monotonic() < deadline:
    process = Path("/proc") / str(record["pid"])
    assert process.exists() and process.stat().st_uid == os.getuid()
    fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
    assert fields[19] == str(record["start_ticks"]) and fields[0] != "Z"
    try:
        with urllib.request.urlopen("http://127.0.0.1:18995/health", timeout=3) as r:
            if r.status == 200:
                print("HEALTHY")
                break
    except OSError:
        pass
    time.sleep(3)
else:
    raise TimeoutError("API readiness timed out")
