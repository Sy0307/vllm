#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Post-start acceptance for the K3 Unified spec55 release. Run on BOTH nodes.

Checks the running server process against the plan the launcher wrote
(exact command line and environment from /proc), key serving parameters,
startup logs and the frozen overlay. On rank0 it also sends one 16-token
greedy request and checks that speculative decoding counters advance.
/server_info is not used: its system-env collection fails in this runtime.

  python3 verify.py --site site.json          # quick
  python3 verify.py --site site.json --full   # also re-hash every overlay file
"""

import argparse
import hashlib
import json
import socket
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def drafts(base):
    for line in get(base + "/metrics").splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total{"):
            return float(line.rsplit(" ", 1)[1])
    return None


def arg(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(HERE / "site.json"))
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true")
    opts = ap.parse_args()
    rt = json.loads((HERE / "runtime.json").read_text())
    site = json.loads(Path(opts.site).read_text())
    host = socket.gethostname().split(".")[0]
    names = [n["hostname"] for n in site["nodes"]]
    rank = names.index(host)
    node_dir = Path(site["release_dir"]) / "run" / site["deployment"] / host
    plan = json.loads((node_dir / "launch.json").read_text())
    pid = json.loads((node_dir / "server-process.json").read_text())["pid"]
    ok = True

    def expect(cond, what):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + what)
        ok &= bool(cond)

    proc = Path(f"/proc/{pid}")
    expect(proc.exists(), f"server process {pid} alive on {host}")
    cmd = proc.joinpath("cmdline").read_bytes().decode().split("\0")[:-1]
    expect(cmd == plan["args"], "process command line == launch.json")
    env = dict(
        x.split("=", 1)
        for x in proc.joinpath("environ").read_bytes().decode().split("\0")
        if "=" in x
    )
    bad = sorted(k for k, v in plan["environment"].items() if env.get(k) != v)
    expect(not bad, f"process environment == launch.json ({bad[:5]})")
    for k, v in rt["env"].items():
        if k.startswith("VLLM_K3_"):
            expect(env.get(k) == v, f"{k}={v}")

    expect(
        (
            arg(cmd, "--tensor-parallel-size"),
            arg(cmd, "--pipeline-parallel-size"),
            arg(cmd, "--decode-context-parallel-size"),
            arg(cmd, "--dcp-comm-backend"),
        )
        == ("8", "2", "8", "a2a"),
        "TP8 x PP2 x DCP8 a2a",
    )
    expect(
        "--enable-expert-parallel" in cmd
        and arg(cmd, "--all2all-backend") == "deepep_v2",
        "EP + deepep_v2",
    )
    kernel = json.loads(arg(cmd, "--kernel-config"))
    expect(kernel.get("moe_backend") == "deep_gemm_mega_moe", "MegaMoE")
    spec = json.loads(arg(cmd, "--speculative-config"))
    expect(
        spec["method"] == "dspark" and spec["num_speculative_tokens"] == 4,
        "DSpark K4",
    )
    expect(
        spec["rejection_sample_method"] == "block"
        or (opts.allow_synthetic and spec["rejection_sample_method"] == "synthetic"),
        f"rejection_sample_method={spec['rejection_sample_method']} "
        "(synthetic is benchmark-only)",
    )
    kv = json.loads(arg(cmd, "--kv-transfer-config"))
    expect(
        kv["kv_connector"] == "MooncakeStoreConnector"
        and kv["kv_connector_extra_config"].get("experimental_recoverssm_store")
        is True,
        "MooncakeStore with same-engine RecoverSSM store",
    )
    expect(
        (
            arg(cmd, "--max-num-seqs"),
            arg(cmd, "--max-num-batched-tokens"),
            arg(cmd, "--gpu-memory-utilization"),
            arg(cmd, "--kv-cache-dtype"),
        )
        == ("192", "8768", "0.88", "fp8"),
        "max_num_seqs 192 / batched 8768 / mem 0.88 / FP8 KV",
    )
    expect("--use-replayssm" in cmd, "RecoverSSM (--use-replayssm)")

    log = (node_dir / "server.log").read_text(errors="replace")
    startup = log.split("Application startup complete.")[0]
    expect("Traceback" not in startup, "no Traceback during startup")
    expect("EngineDeadError" not in log, "no EngineDeadError")
    if rank == 0:
        expect(
            "Kimi K3 model-level sequence parallelism is enabled." in log,
            "model-level SP enabled",
        )

    manifest = Path(site["release_dir"]) / "overlay-SHA256SUMS"
    expect(sha256(manifest) == site["overlay_manifest_sha256"], "overlay manifest")
    if opts.full:
        overlay = Path(site["release_dir"]) / "overlay"
        mism = [
            rel
            for h, rel in (
                line.split("  ", 1)
                for line in manifest.read_text().splitlines()
                if line
            )
            if sha256(overlay / rel) != h
        ]
        expect(not mism, f"full overlay re-hash ({len(mism)} mismatches)")

    if rank == 0:
        base = f"http://{site['nodes'][0]['ip']}:{rt['ports']['api']}"
        get(base + "/health")
        before = drafts(base)
        body = json.dumps(
            {
                "model": "kimi-k3",
                "prompt": "Hello",
                "max_tokens": 16,
                "temperature": 0,
                "ignore_eos": True,
            }
        ).encode()
        req = urllib.request.Request(
            base + "/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
        expect(out["usage"]["completion_tokens"] == 16, "basic 16-token request")
        time.sleep(2)
        after = drafts(base)
        expect(
            before is not None and after is not None and after > before,
            f"spec decode drafts advanced ({before} -> {after})",
        )
    print("VERIFY", host, "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
