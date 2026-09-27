# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, E702, E741, SIM115

"""Ops-release variant (no champion probe RPCs). Native InferenceX 900s scored run (no profiler) for paired spec / no-spec arms.

Timeline (per-batch worker/runner/PP spans, TP0 of both stages) from formal
t=240s to t=480s; one 8 s Kineto capture at formal t=300s. No score is claimed.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

R = Path("/gpfs/mszn/data/k3-benchmarks/k3-pd-four-node-20260923")
O = Path(sys.argv[1])  # release run dir, e.g. <release>/run/<deployment>
SUBDIR = "client-1h"
H = Path("/gpfs/mszn/data/k3-benchmarks/ref440542")
OUT = O / SUBDIR
OUT.mkdir(exist_ok=False)
URL = "http://10.18.1.25:18984"


def api(path, body=None, timeout=60):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode()


def rpc(name, *args):
    return api(
        "/collective_rpc",
        {"method": name, "args": list(args), "timeout": 240},
        timeout=250,
    )


def metrics():
    raw = api("/metrics", timeout=5)
    j = {}
    for line in raw.splitlines():
        if line.startswith(
            (
                "vllm:num_requests_running{",
                "vllm:num_requests_waiting{",
                "vllm:num_preemptions_total{",
                "vllm:generation_tokens_total{",
                "vllm:prompt_tokens_by_source_total{",
                "vllm:spec_decode",
                "vllm:kv_cache_usage_perc",
                "vllm:prefix_cache",
            )
        ):
            k, v = line.rsplit(" ", 1)
            j[k] = float(v)
    return j


def status(**kw):
    (OUT / "status.json").write_text(json.dumps(dict(time=time.time(), **kw)))


t0 = time.time()
while True:
    try:
        api("/health", timeout=3)
        info = {"probe": None, "note": "ops release: no worker extension"}
        break
    except Exception as ex:  # startup
        status(step="waiting_ready", err=repr(ex)[:200], elapsed=time.time() - t0)
        if time.time() - t0 > 1800:
            raise
        time.sleep(10)
(OUT / "ready.json").write_text(json.dumps(info, indent=1))
cmd = json.loads(
    (R / "unified-next-optimizations-20260924/baseline/score/manifest.json").read_text()
)["command"]
cmd = list(cmd)
cmd[cmd.index("--benchmark-duration") + 1] = (
    "3600"  # full-hour, same params as champion
)
cmd[cmd.index("--output-artifact-dir") + 1] = str(OUT / "aiperf_artifacts")
env = dict(os.environ)
env.update(
    PYTHONPATH=str(H / "tokenizer-path-compat"),
    HF_HOME=str(H / "hf-cache"),
    HF_HUB_OFFLINE="1",
    HF_DATASETS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
    AIPERF_DATASET_MMAP_CACHE_DIR=str(H / "mmap-cache"),
    AIPERF_DATASET_CONFIGURATION_TIMEOUT="180",
    AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="180",
    TERM="xterm-256color",
)
(OUT / "manifest.json").write_text(
    json.dumps({"command": cmd, "scope": __doc__}, indent=1)
)
start = time.monotonic()
formal = None
tl = False
cap = None
tl_done = False
reason = None
with (OUT / "client.log").open("w") as log, (OUT / "metrics.jsonl").open("w") as mon:
    p = subprocess.Popen(
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (OUT / "harness-process.json").write_text(json.dumps({"pid": p.pid}))
    try:
        while p.poll() is None and time.monotonic() - start < 9000:
            text = (OUT / "client.log").read_text(errors="replace")
            if formal is None and "Phase profiling (profiling) started" in text:
                formal = time.monotonic()
            m = metrics()
            mon.write(json.dumps({"time": time.time(), "metrics": m}) + "\n")
            mon.flush()
            f = time.monotonic() - formal if formal else None
            if False:
                (OUT / "timeline-begin.json").write_text(
                    json.dumps(
                        {
                            "time": time.time(),
                            "r": rpc("timeline_begin", "score1h", 1, 9000),
                        }
                    )
                )
                tl = True
            if False:
                (OUT / "timeline-end.json").write_text(
                    json.dumps({"time": time.time(), "r": rpc("timeline_end")})
                )
                tl_done = True
            status(
                step="running", formal_elapsed=f, timeline=tl, capture=cap, metrics=m
            )
            time.sleep(2)
        reason = "harness_exit" if p.poll() is not None else "deadline"
        (OUT / "server-metrics-final.txt").write_text(api("/metrics", timeout=10))
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        if tl and not tl_done:
            (OUT / "timeline-end.json").write_text(
                json.dumps({"time": time.time(), "r": rpc("timeline_end")})
            )
        status(step="finished", reason=reason, rc=p.returncode)
