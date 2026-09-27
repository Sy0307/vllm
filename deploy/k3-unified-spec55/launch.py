#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kimi K3 Unified (DSpark K4 spec, TP8 x PP2 x DCP8 + EP + SP) node launcher.

Run once on each node of the pair; rank is taken from the hostname in site.json.
The launcher only checks and starts processes. It never kills processes, never
installs packages and never changes drivers.

  python3 launch.py --site site.json            # checks only, prints the plan
  python3 launch.py --site site.json --launch   # checks, then starts

--synthetic-acceptance R1,R2,R3,R4 replaces real rejection sampling with the
synthetic sampler at the given per-position acceptance. Benchmark use only:
outputs are not the model's real outputs. Never use it for serving traffic.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def subst(value, table):
    if isinstance(value, str):
        for k, v in table.items():
            value = value.replace("{" + k + "}", v)
        return value
    if isinstance(value, list):
        return [subst(x, table) for x in value]
    if isinstance(value, dict):
        return {k: subst(v, table) for k, v in value.items()}
    return value


def check(rt, site, rank, host):
    errors = []
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if apps:
        errors.append(f"GPU occupied by pids: {apps}")
    gpus = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        )
        .strip()
        .splitlines()
    )
    if len(gpus) != 8 or not all("B200" in g for g in gpus):
        errors.append(f"expected 8x B200, got {gpus}")
    ecc = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).split()
    if any(x not in ("0", "[N/A]") for x in ecc):
        errors.append(f"uncorrected volatile ECC errors: {ecc}")
    ports = [rt["ports"]["api"], rt["ports"]["torch_master"]]
    if rank == 0:
        ports += [
            rt["ports"]["mooncake_rpc"],
            rt["ports"]["mooncake_metrics"],
            rt["ports"]["mooncake_http_metadata"],
        ]
    busy = subprocess.check_output(
        ["ss", "-H", "-ltn", " or ".join(f"sport = :{p}" for p in ports)], text=True
    ).strip()
    if busy:
        errors.append(f"ports in use:\n{busy}")
    mem_kib = int(
        next(
            line.split()[1]
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
    )
    if mem_kib < rt["host_mem_min_gib"] * 1024**2:
        errors.append(
            f"MemAvailable {mem_kib / 1024**2:.0f} GiB < {rt['host_mem_min_gib']} GiB "
            "(MooncakeStore registers 200 GB per worker)"
        )
    for p in [rt["python"], rt["mooncake_master"]["binary"], *rt["models"].values()]:
        if not Path(p).exists():
            errors.append(f"missing: {p}")
    overlay = Path(site["release_dir"]) / "overlay"
    manifest = Path(site["release_dir"]) / "overlay-SHA256SUMS"
    if not manifest.exists():
        errors.append(f"missing overlay manifest {manifest}")
    elif sha256(manifest) != site["overlay_manifest_sha256"]:
        errors.append("overlay manifest checksum differs from site.json")
    else:
        # Spot-check the files this release changed; verify.py re-hashes all.
        want = dict(
            line.split("  ", 1)[::-1]
            for line in manifest.read_text().splitlines()
            if line
        )
        for rel in rt["changed_files"]:
            key = "vllm/" + rel
            if key not in want or sha256(overlay / key) != want[key]:
                errors.append(f"overlay file differs from release: {key}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(HERE / "site.json"))
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--synthetic-acceptance", default="")
    args = ap.parse_args()
    rt = json.loads((HERE / "runtime.json").read_text())
    site = json.loads(Path(args.site).read_text())
    host = socket.gethostname().split(".")[0]
    names = [n["hostname"] for n in site["nodes"]]
    assert host in names, f"{host} is not in site.json nodes {names}"
    rank = names.index(host)
    state = Path(site["release_dir"]) / "run" / site["deployment"]
    node_dir = state / host
    table = {
        "RANK0_IP": site["nodes"][0]["ip"],
        "SELF_IP": site["nodes"][rank]["ip"],
        "OVERLAY": str(Path(site["release_dir"]) / "overlay"),
        "STATE_DIR": str(state),
        "NODE_DIR": str(node_dir),
        "HOST": host,
    }

    errors = check(rt, site, rank, host)
    serve = [
        rt["python"],
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        rt["models"]["target"],
    ]
    serve += subst(rt["serve_args"] + rt["rank_args"][str(rank)], table)
    if args.synthetic_acceptance:
        rates = [float(x) for x in args.synthetic_acceptance.split(",")]
        i = serve.index("--speculative-config")
        spec = json.loads(serve[i + 1])
        assert len(rates) == spec["num_speculative_tokens"]
        spec["rejection_sample_method"] = "synthetic"
        spec["synthetic_acceptance_rates"] = rates
        serve[i + 1] = json.dumps(spec)
    env = {
        k: v
        for k, v in os.environ.items()
        if k in ("HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG")
    }
    env.update(subst(rt["env"], table))
    mooncake = subst(rt["mooncake"], table)
    plan = {
        "host": host,
        "rank": rank,
        "release": rt["release"],
        "args": serve,
        "environment": env,
        "mooncake": mooncake,
        "synthetic_acceptance": args.synthetic_acceptance or None,
    }
    if errors:
        print("PRECHECK FAILED")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    print("PRECHECK PASSED", host, "rank", rank)
    if not args.launch:
        print(json.dumps(plan, indent=1)[:4000])
        return
    node_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("vllm", "triton", "inductor"):
        (state / "cache" / host / sub).mkdir(parents=True, exist_ok=True)
    (node_dir / "launch.json").write_text(json.dumps(plan, indent=1))
    (node_dir / "mooncake.json").write_text(json.dumps(mooncake, indent=2))
    if rank == 0:
        cmd = [rt["mooncake_master"]["binary"], *rt["mooncake_master"]["args"]]
        with (node_dir / "mooncake-master.log").open("w") as f:
            m = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        time.sleep(2)
        assert m.poll() is None, (node_dir / "mooncake-master.log").read_text()[-3000:]
        (node_dir / "mooncake-master.json").write_text(
            json.dumps({"pid": m.pid, "command": cmd})
        )
    with (node_dir / "server.log").open("w") as f:
        p = subprocess.Popen(
            serve,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (node_dir / "server-process.json").write_text(
        json.dumps({"pid": p.pid, "started_at": time.time()})
    )
    print("STARTED", host, "rank", rank, "pid", p.pid, "log", node_dir / "server.log")


if __name__ == "__main__":
    main()
