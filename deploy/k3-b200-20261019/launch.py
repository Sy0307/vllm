# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Owned two-node K3 serving session (prod30/prod24 or prod09/prod25).

usage: launch.py RUN VARIANT_JSON
Loads the locked B200 runtime, applies the serving or benchmark variant, and
starts the server in its own session. Records identity and effective inputs
under state/runs/RUN/HOST.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from host_memory import check_mooncake_memory

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE / "state"
assert os.getuid() == PACKAGE.stat().st_uid == 1000
ROOT.mkdir(exist_ok=True)
host = socket.gethostname().split(".")[0]
API_PORT, MASTER_PORT = 18995, 29795
MODEL = "/gpfs/mszn/models/moonshotai/Kimi-K3"
DRAFT = "/gpfs/mszn/models/Inferact/Kimi-K3-DSpark"
DEFAULT_OVERLAY = str(PACKAGE / "overlay")

run, variant_path = sys.argv[1], sys.argv[2]
assert run.replace("-", "").replace("_", "").isalnum(), run
variant = json.loads(Path(variant_path).read_text())
nodes = variant.get("nodes", ["xb01-gpu-200b-0030", "xb01-gpu-200b-0024"])
assert nodes in (
    ["xb01-gpu-200b-0030", "xb01-gpu-200b-0024"],
    ["xb01-gpu-200b-0009", "xb01-gpu-200b-0025"],
), nodes
RANK = nodes.index(host)
SELF_IP = f"10.18.1.{int(host.rsplit('-', 1)[1])}"
RANK0_IP = f"10.18.1.{int(nodes[0].rsplit('-', 1)[1])}"
memory_preflight = None
if variant.get("mooncake", False):
    memory_preflight = check_mooncake_memory(variant.get("mc_segment", "160GB"))
    print("HOST_MEMORY", json.dumps(memory_preflight), flush=True)

assert not subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
).strip(), "GPU occupied"
for port in (API_PORT, MASTER_PORT):
    assert not subprocess.check_output(
        ["ss", "-H", "-ltn", f"sport = :{port}"], text=True
    ).strip(), f"port {port} busy"

spec = json.loads((Path(__file__).resolve().parent / "prod_runtime.json").read_text())
overlay = variant.get("overlay", DEFAULT_OVERLAY)
if not Path(overlay).is_absolute():
    overlay = str((PACKAGE / overlay).resolve())
subprocess.run(
    [sys.executable, str(PACKAGE / "verify_release.py"), overlay], check=True
)
assert Path(overlay, "vllm", "__init__.py").exists(), overlay
state = ROOT / "state"
subst = {
    "{RANK0_IP}": RANK0_IP,
    "{SELF_IP}": SELF_IP,
    "{OVERLAY}": overlay,
    "{STATE_DIR}": str(state),
    "{HOST}": host,
    "{NODE_DIR}": "",
}


def sub(s):
    for k, v in subst.items():
        s = s.replace(k, v)
    return s


MODEL = variant.get("model_path", MODEL)  # e.g. node-local NVMe copy
DRAFT = variant.get("draft_path", DRAFT)
args = [spec["python"], "-m", "vllm.entrypoints.cli.main", "serve", MODEL]
args += [sub(a) for a in spec["serve_args"]]
args += [sub(a) for a in spec["rank_args"][str(RANK)]]


def set_arg(flag, value):
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args.extend([flag, value])


def remove_arg(flag, has_value=True):
    if flag in args:
        i = args.index(flag)
        del args[i : i + (2 if has_value else 1)]


set_arg("--port", str(API_PORT))
set_arg("--master-port", str(MASTER_PORT))
sc = json.loads(args[args.index("--speculative-config") + 1])
sc["model"] = DRAFT
sc.update(variant.get("spec_update", {}))
for k in variant.get("spec_remove", []):
    sc.pop(k, None)
set_arg("--speculative-config", json.dumps(sc))
# Mooncake is off unless the variant explicitly keeps it.
MC_RPC, MC_METRICS, MC_HTTP = 18796, 19796, 18797
if not variant.get("mooncake", False):
    remove_arg("--kv-transfer-config")
else:
    kv = json.loads(args[args.index("--kv-transfer-config") + 1])
    kv["kv_connector_extra_config"]["cache_prefix"] = f"k3-release-20261019-{run}"
    set_arg("--kv-transfer-config", json.dumps(kv))
for flag, value in variant.get("args_set", {}).items():
    if isinstance(value, (dict, list)):
        value = json.dumps(value)
    set_arg(flag, str(value))
for flag, value in variant.get("rank_args_set", {}).get(str(RANK), {}).items():
    set_arg(flag, str(value))
for flag in variant.get("args_remove", []):
    remove_arg(
        flag,
        has_value=not flag.startswith("--no-")
        and flag
        not in (
            "--enable-expert-parallel",
            "--trust-remote-code",
            "--enable-prefix-caching",
            "--enable-prompt-tokens-details",
            "--enable-auto-tool-choice",
            "--use-replayssm",
            "--headless",
            "--language-model-only",
            "--enforce-eager",
        ),
    )
args += variant.get("args_add", [])

env = {
    k: v
    for k, v in os.environ.items()
    if k in ("HOME", "USER", "LOGNAME", "SHELL", "LANG")
}
env["TERM"] = "dumb"
env.update({k: sub(v) for k, v in spec["env"].items()})
for k in list(env):
    if (
        k.startswith(("MC_", "MOONCAKE_")) and not variant.get("mooncake", False)
    ) or k in ("VLLM_BUILD_COMMIT", "VLLM_IMAGE_TAG"):
        env.pop(k)
env["HF_HUB_CACHE"] = env["HUGGINGFACE_HUB_CACHE"] = str(ROOT / "cache" / "hf")
env["VLLM_MM_INPUT_SHARED_STORAGE_PATH"] = f"/dev/shm/k3-release-20261019-{run}"
for k, v in variant.get("env", {}).items():
    if v is None:
        env.pop(k, None)
    else:
        env[k] = sub(str(v))
if variant.get("extra_pythonpath"):
    env["PYTHONPATH"] = (
        env["PYTHONPATH"]
        + ":"
        + ":".join(
            str(PACKAGE) if p == "{PACKAGE}" else p for p in variant["extra_pythonpath"]
        )
    )

subprocess.run([args[0], str(PACKAGE / "check_environment.py")], env=env, check=True)

node = ROOT / "runs" / run / host
node.mkdir(parents=True, exist_ok=False)
if variant.get("mooncake", False):
    mc = {
        "metadata_server": "P2PHANDSHAKE",
        "global_segment_size": variant.get("mc_segment", "160GB"),
        "local_buffer_size": "4GB",
        "protocol": "rdma",
        "device_name": "ib7s400p0,ib7s400p1,ib7s400p2,ib7s400p3",
        "mode": "embedded",
        "enable_offload": False,
        "master_server_address": f"{RANK0_IP}:{MC_RPC}",
    }
    (node / "mooncake.json").write_text(json.dumps(mc, indent=1))
    env["MOONCAKE_CONFIG_PATH"] = str(node / "mooncake.json")
    env["MOONCAKE_REQUESTER_LOCAL_HOSTNAME"] = SELF_IP
    if RANK == 0:
        for port in (MC_RPC, MC_METRICS, MC_HTTP):
            assert not subprocess.check_output(
                ["ss", "-H", "-ltn", f"sport = :{port}"], text=True
            ).strip(), port
        mcmd = [
            "/gpfs/mszn/data/k3-benchmarks/nightly-9521-env/bin/mooncake_master",
            f"--rpc_port={MC_RPC}",
            f"--metrics_port={MC_METRICS}",
            "--enable_http_metadata_server=true",
            f"--http_metadata_server_port={MC_HTTP}",
            "--default_kv_lease_ttl=10000",
            "--default_kv_soft_pin_ttl=1800000",
            "--allow_evict_soft_pinned_objects=true",
            "--eviction_ratio=0.05",
            "--eviction_high_watermark_ratio=0.9",
            "--enable_offload=false",
        ]
        with (node / "master.log").open("w") as mlog:
            mp = subprocess.Popen(
                mcmd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=mlog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(node),
            )
        (node / "master-process.json").write_text(
            json.dumps(
                {
                    "pid": mp.pid,
                    "pgid": os.getpgid(mp.pid),
                    "start_ticks": Path(f"/proc/{mp.pid}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[19],
                    "cmd": mcmd,
                }
            )
        )
        time.sleep(3)
        assert mp.poll() is None, (node / "master.log").read_text()[-2000:]
Path(env["VLLM_MM_INPUT_SHARED_STORAGE_PATH"]).mkdir(
    mode=0o700, parents=True, exist_ok=True
)
for d in ("vllm", "triton", "inductor"):
    (state / "cache" / host / d).mkdir(parents=True, exist_ok=True)
(ROOT / "cache" / "hf").mkdir(parents=True, exist_ok=True)

prefix = variant.get("cmd_prefix", [])
full = [sub(x).replace("{NODE}", str(node)) for x in prefix + args]
(node / "config.json").write_text(
    json.dumps(
        {
            "host": host,
            "rank": RANK,
            "run": run,
            "variant": variant,
            "args": full,
            "env": env,
            "physical_gpus": list(range(8)),
            "memory_preflight": memory_preflight,
            "time": time.time(),
        },
        indent=1,
    )
)
with (node / "server.log").open("w") as log:
    proc = subprocess.Popen(
        full,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(node),
    )
rec = {
    "pid": proc.pid,
    "pgid": os.getpgid(proc.pid),
    "start_ticks": Path(f"/proc/{proc.pid}/stat")
    .read_text()
    .rsplit(")", 1)[1]
    .split()[19],
    "host": host,
    "run": run,
    "time": time.time(),
}
(node / "process.json").write_text(json.dumps(rec))
# isolated campaign does not change shared current pointer
print("STARTED", json.dumps(rec), flush=True)
