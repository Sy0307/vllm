# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native InferenceX AgentX replay (same aiperf command as the 09-24..09-27 runs).

usage: run_agentx.py RUN TAG [--duration 900] [--concurrency 96] [--warmup 10]
                     [--profile-at S --profile-for S] [--url URL]
Runs on prod30. Writes runs/RUN/agentx-TAG/.
"""

import argparse
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "state"
H = Path("/gpfs/mszn/data/k3-benchmarks/ref440542")
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("tag")
ap.add_argument("--duration", type=int, default=900)
ap.add_argument("--concurrency", type=int, default=96)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--grace", type=int, default=1800)
ap.add_argument("--profile-at", type=float, default=0)
ap.add_argument("--profile-for", type=float, default=0)
ap.add_argument("--url", default="http://127.0.0.1:18995")
ap.add_argument("--no-reset", action="store_true")
ap.add_argument("--cprof-at", type=float, default=0)
ap.add_argument(
    "--warmup-base",
    action="store_true",
    help="time --*-at offsets from warmup start instead of profiling start",
)
ap.add_argument("--cprof-for", type=float, default=0)
ap.add_argument("--unsafe", action="store_true")
ap.add_argument("--flag-file", default="")
ap.add_argument("--flag-on-at", type=float, default=0)
ap.add_argument("--timeline-at", type=float, default=0)
ap.add_argument("--timeline-for", type=float, default=0)
ap.add_argument("--syncdbg-at", type=float, default=0)
ap.add_argument("--syncdbg-for", type=float, default=0)
a = ap.parse_args()
assert socket.gethostname().split(".")[0] == "xb01-gpu-200b-0030"
URL = a.url
out = ROOT / "runs" / a.run / f"agentx-{a.tag}"
out.mkdir(parents=True, exist_ok=False)


def api(path, body=None, timeout=30, method=None):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw.decode()


def metrics():
    raw = api("/metrics", timeout=5)
    parsed = {}
    for line in raw.splitlines():
        if not line.startswith("vllm:"):
            continue
        key, val = line.rsplit(" ", 1)
        name = key.split("{", 1)[0]
        if "_bucket" in name:
            continue
        if any(
            s in name
            for s in (
                "tokens",
                "requests",
                "preempt",
                "cache_usage",
                "prefix_cache",
                "spec_decode",
                "time_to_first",
                "time_per_output",
                "inter_token",
                "queue_time",
                "kv_",
            )
        ):
            with contextlib.suppress(ValueError):
                parsed[key] = float(val)
    return raw, parsed


def idle(timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, m = metrics()
        vals = [
            v
            for k, v in m.items()
            if k.startswith(
                ("vllm:num_requests_running{", "vllm:num_requests_waiting{")
            )
        ]
        if len(vals) >= 2 and not any(vals):
            return
        time.sleep(1)
    raise TimeoutError("engine did not drain")


api("/health", timeout=5)
idle()
if not a.no_reset:
    (out / "cache-reset.json").write_text(
        json.dumps({"result": api("/reset_prefix_cache", {}, timeout=60)})
    )
(out / "server-before.prom").write_text(api("/metrics"))
cmd = [
    str(H / "harness-venv/bin/aiperf"),
    "profile",
    "--scenario",
    "inferencex-agentx-mvp",
    "--url",
    URL,
    "--endpoint",
    "/v1/chat/completions",
    "--endpoint-type",
    "chat",
    "--streaming",
    "--model",
    "kimi-k3",
    "--tokenizer",
    "/gpfs/mszn/models/moonshotai/Kimi-K3",
    "--concurrency",
    str(a.concurrency),
    "--benchmark-duration",
    str(a.duration),
    "--stats-interval",
    "30",
    "--random-seed",
    "42",
    "--failed-request-threshold",
    "0.25",
    "--trajectory-start-min-ratio",
    "0.25",
    "--trajectory-start-max-ratio",
    "0.75",
    "--warmup-requests-per-lane",
    str(a.warmup),
    "--trace-idle-gap-cap-seconds",
    "300",
    "--warmup-grace-period",
    str(a.grace),
    "--use-server-token-count",
    "--no-gpu-telemetry",
    "--tokenizer-trust-remote-code",
    "--num-dataset-entries",
    "393",
    "--slice-duration",
    "1.0",
    "--server-metrics",
    URL + "/metrics",
    "--output-artifact-dir",
    str(out / "aiperf_artifacts"),
    "--public-dataset",
    "semianalysis_cc_traces_weka_062126",
]
if a.unsafe:
    cmd += ["--unsafe-override"]
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
for k in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
):
    env.pop(k, None)
(out / "manifest.json").write_text(
    json.dumps({"command": cmd, "args": vars(a), "started_at": time.time()}, indent=1)
)
for h in ("xb01-gpu-200b-0030", "xb01-gpu-200b-0024"):
    p = ROOT / "runs" / a.run / h / "config.json"
    if p.exists():
        (out / f"config-{h}.json").write_text(p.read_text())

stop = threading.Event()


def sampler():
    with (out / "metrics.jsonl").open("w") as f:
        while not stop.is_set():
            row = {"time": time.time()}
            try:
                row["metrics"] = metrics()[1]
            except Exception as exc:
                row["error"] = repr(exc)
            f.write(json.dumps(row) + "\n")
            f.flush()
            stop.wait(1)


threading.Thread(target=sampler, daemon=True).start()
reason = None
cp_started = cp_stopped = False
warmup_start = profile_start = None
prof_started = prof_stopped = False
tl_started = tl_stopped = sd_started = sd_stopped = False
flag_on = False
if a.flag_file:
    Path(a.flag_file).write_text("0")


def rpc(method, *args):
    return api(
        "/collective_rpc",
        {"method": method, "args": [str(x) for x in args], "timeout": 300},
        timeout=320,
    )


offset = 0
t0 = time.monotonic()
deadline = t0 + 10800  # outer supervision only; native flags unchanged
with (out / "client.log").open("w") as log:
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (out / "harness-process.json").write_text(json.dumps({"pid": proc.pid}))
    try:
        while proc.poll() is None:
            with (out / "client.log").open(errors="replace") as m:
                m.seek(offset)
                new = m.read()
                offset = m.tell()
            if "Phase warmup (warmup) started" in new and warmup_start is None:
                warmup_start = time.time()
                if a.warmup_base:
                    profile_start = warmup_start
            if (
                "Phase profiling (profiling) started" in new
                and profile_start is None
                and not a.warmup_base
            ):
                profile_start = time.time()
            if (
                a.profile_at
                and profile_start
                and not prof_started
                and time.time() - profile_start > a.profile_at
            ):
                (out / "profile-start.json").write_text(
                    json.dumps(
                        {
                            "t": time.time(),
                            "r": str(api("/start_profile", {}, timeout=120)),
                        }
                    )
                )
                prof_started = True
            if (
                prof_started
                and not prof_stopped
                and time.time() - profile_start > a.profile_at + a.profile_for
            ):
                ts = time.time()
                r = api("/stop_profile", {}, timeout=900)
                (out / "profile-stop.json").write_text(
                    json.dumps({"t": ts, "done": time.time(), "r": str(r)})
                )
                prof_stopped = True
            el = time.time() - profile_start if profile_start else -1
            if (
                a.timeline_at
                and profile_start
                and not tl_started
                and el > a.timeline_at
            ):
                (out / "timeline-begin.json").write_text(
                    json.dumps(rpc("k3_timeline_begin", "agentx", str(out)))
                )
                tl_started = True
            if tl_started and not tl_stopped and el > a.timeline_at + a.timeline_for:
                (out / "timeline-end.json").write_text(
                    json.dumps(rpc("k3_timeline_end"))
                )
                tl_stopped = True
            if a.flag_file and profile_start and not flag_on and el > a.flag_on_at:
                Path(a.flag_file).write_text("1")
                flag_on = True
                (out / "flag-on.json").write_text(
                    json.dumps({"t": time.time(), "el": el})
                )
            if a.cprof_at and profile_start and not cp_started and el > a.cprof_at:
                (out / "cprof-begin.json").write_text(
                    json.dumps(rpc("k3_cprofile", "1"))
                )
                cp_started = True
            if cp_started and not cp_stopped and el > a.cprof_at + a.cprof_for:
                (out / "cprof-end.json").write_text(
                    json.dumps(rpc("k3_cprofile", "0", str(out.resolve() / "cprof")))
                )
                cp_stopped = True
            if a.syncdbg_at and profile_start and not sd_started and el > a.syncdbg_at:
                (out / "syncdbg-begin.json").write_text(
                    json.dumps(rpc("k3_sync_debug", "1"))
                )
                sd_started = True
            if sd_started and not sd_stopped and el > a.syncdbg_at + a.syncdbg_for:
                (out / "syncdbg-end.json").write_text(
                    json.dumps(rpc("k3_sync_debug", "0", str(out / "syncdbg")))
                )
                sd_stopped = True
            (out / "status.json").write_text(
                json.dumps(
                    {
                        "elapsed": time.monotonic() - t0,
                        "warmup_start": warmup_start,
                        "profile_start": profile_start,
                        "rc": proc.poll(),
                    }
                )
            )
            if time.monotonic() > deadline:
                reason = "deadline"
                break
            time.sleep(2)
    except BaseException as exc:
        reason = "exception:" + repr(exc)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        rc = proc.returncode
        stop.set()
        try:
            if tl_started and not tl_stopped:
                (out / "timeline-end.json").write_text(
                    json.dumps(rpc("k3_timeline_end"))
                )
            if sd_started and not sd_stopped:
                (out / "syncdbg-end.json").write_text(
                    json.dumps(rpc("k3_sync_debug", "0", str(out / "syncdbg")))
                )
        except Exception as exc:
            (out / "rpc-cleanup-error.txt").write_text(repr(exc))
        try:
            idle(120)
            (out / "server-after.prom").write_text(api("/metrics"))
        except Exception as exc:
            (out / "cleanup-error.txt").write_text(repr(exc))

summary = {"rc": rc, "reason": reason}
p = out / "aiperf_artifacts" / "profile_export_aiperf.json"
if p.exists():
    d = json.loads(p.read_text())
    summary["aiperf_keys"] = list(d.keys())[:40]
    for k in (
        "output_token_throughput",
        "request_throughput",
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
        "output_sequence_length",
        "input_sequence_length",
        "request_count",
        "error_request_count",
    ):
        if k in d:
            v = d[k]
            summary[k] = (
                {x: v.get(x) for x in ("avg", "p50", "p90", "p99", "unit")}
                if isinstance(v, dict)
                else v
            )
logtext = (out / "client.log").read_text(errors="replace")
m = re.search(
    r"Phase profiling \(profiling\) sending complete \| sent=([\d,]+)", logtext
)
summary["profiling_sent"] = m and m[1]
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print("AGENTX_DONE", json.dumps(summary)[:3000], flush=True)
