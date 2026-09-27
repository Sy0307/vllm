#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Post-start acceptance for the K3 Unified spec55 release. Read-only except one
16-token greedy request. Run on the rank0 node after /health succeeds.

  python3 verify.py --site site.json            # config, logs, env, overlay, request
  python3 verify.py --site site.json --full     # also re-hash every overlay file
"""

import argparse
import hashlib
import json
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
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(HERE / "site.json"))
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    rt = json.loads((HERE / "runtime.json").read_text())
    site = json.loads(Path(args.site).read_text())
    rank0 = site["nodes"][0]
    base = f"http://{rank0['ip']}:{rt['ports']['api']}"
    state = Path(site["release_dir"]) / "run" / site["deployment"]
    ok = True

    def expect(cond, what):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + what)
        ok &= bool(cond)

    get(base + "/health")
    print("PASS /health")
    c = json.loads(get(base + "/server_info?config_format=json"))["vllm_config"]
    c = json.loads(c) if isinstance(c, str) else c
    p = c["parallel_config"]
    expect(
        (
            p["tensor_parallel_size"],
            p["pipeline_parallel_size"],
            p["decode_context_parallel_size"],
        )
        == (8, 2, 8),
        "TP8 x PP2 x DCP8",
    )
    expect(p["dcp_comm_backend"] == "a2a", "DCP a2a")
    expect(
        p["enable_expert_parallel"] and p["all2all_backend"] == "deepep_v2",
        "EP + deepep_v2",
    )
    expect(c["kernel_config"]["moe_backend"] == "deep_gemm_mega_moe", "MegaMoE")
    s = c["speculative_config"]
    expect(
        s and s["method"] == "dspark" and s["num_speculative_tokens"] == 4, "DSpark K4"
    )
    expect(
        s and s["rejection_sample_method"] == "block",
        "rejection_sample_method=block (synthetic is benchmark-only)",
    )
    sc = c["scheduler_config"]
    expect(
        (sc["max_num_seqs"], sc["max_num_batched_tokens"]) == (192, 8768),
        "max_num_seqs 192 / batched 8768",
    )
    expect(c["cache_config"]["cache_dtype"] == "fp8", "FP8 KV cache")
    kv = c["kv_transfer_config"]
    expect(
        kv["kv_connector"] == "MooncakeStoreConnector"
        and kv["kv_connector_extra_config"].get("experimental_recoverssm_store")
        is True,
        "MooncakeStore with same-engine RecoverSSM store",
    )

    for n in site["nodes"]:
        log = (state / n["hostname"] / "server.log").read_text(errors="replace")
        if n is rank0:
            expect(
                "Kimi K3 model-level sequence parallelism is enabled." in log,
                "model-level SP log",
            )
        expect("Traceback" not in log, f"no Traceback in {n['hostname']} server.log")

    # Environment actually seen by the rank0 server process.
    pid = json.loads((state / rank0["hostname"] / "server-process.json").read_text())[
        "pid"
    ]
    env = dict(
        x.split("=", 1)
        for x in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0")
        if "=" in x
    )
    for k, v in rt["env"].items():
        if k.startswith("VLLM_K3_"):
            expect(env.get(k) == v, f"{k}={v}")

    manifest = Path(site["release_dir"]) / "overlay-SHA256SUMS"
    expect(
        sha256(manifest) == site["overlay_manifest_sha256"], "overlay manifest checksum"
    )
    if args.full:
        bad = [
            rel
            for h, rel in (
                line.split("  ", 1)
                for line in manifest.read_text().splitlines()
                if line
            )
            if sha256(Path(site["release_dir"]) / "overlay" / rel) != h
        ]
        expect(not bad, f"full overlay re-hash ({len(bad)} mismatches)")

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
    out = json.loads(urllib.request.urlopen(req, timeout=300).read())
    expect(out["usage"]["completion_tokens"] == 16, "basic 16-token request")
    print("VERIFY", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
