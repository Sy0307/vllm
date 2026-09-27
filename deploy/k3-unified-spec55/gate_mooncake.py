# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, E702, E741, SIM115

"""Mooncake/RecoverSSM correctness gate (run on a fresh deployment before traffic;
it resets the GPU prefix cache and requests top-5 logprobs).

usage: gate_mooncake.py OUT.json [seed] [base_url]

Relative correctness gate for Mooncake-loaded KDA RecoverSSM prefixes.

Engine is not batch/shape invariant, so token equality across different cached
lengths is not a valid criterion. Per prompt:
  run0 cold, run1 local hit, run1b local hit (determinism at identical shape),
  reset GPU cache, run2 Mooncake hit (different cached length).
Metrics on greedy continuations with top-5 logprobs:
  agree(a,b): common prefix length; lpdiff(a,b): mean |logprob| delta of the
  chosen token over the common prefix; conf(x): mean chosen-token logprob.
Pass: run1 == run1b (determinism), external>0 for run2, and store deviation
not worse than the cold-vs-local deviation (d12 <= 1.5*d01 + 0.02) and store
confidence within 0.1 nats of local.
"""

import json
import random
import sys
import time
import urllib.request

URL = sys.argv[3] if len(sys.argv) > 3 else "http://10.18.1.25:18984"


def post(path, body, timeout=900):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode()


def metrics():
    with urllib.request.urlopen(URL + "/metrics", timeout=10) as r:
        raw = r.read().decode()
    out = {}
    for l in raw.splitlines():
        if l.startswith("vllm:prompt_tokens_by_source_total{"):
            k, v = l.rsplit(" ", 1)
            out[k.split('source="')[1].split('"')[0]] = float(v)
    return out


def gen(tokens, n=64):
    r = post(
        "/v1/completions",
        {
            "model": "kimi-k3",
            "prompt": tokens,
            "max_tokens": n,
            "temperature": 0,
            "ignore_eos": True,
            "logprobs": 5,
        },
    )
    c = r["choices"][0]["logprobs"]
    return {
        "tok": c["tokens"],
        "lp": c["token_logprobs"],
        "top": c["top_logprobs"],
        "cached": r["usage"]["prompt_tokens_details"]["cached_tokens"],
    }


def agree(a, b):
    n = 0
    for x, y in zip(a["tok"], b["tok"]):
        if x != y:
            break
        n += 1
    return n


def lpdiff(a, b):
    n = max(1, agree(a, b))
    return sum(abs(a["lp"][i] - b["lp"][i]) for i in range(n)) / n


def conf(a):
    return sum(a["lp"]) / len(a["lp"])


results = []
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 11)
# natural-ish text: repeat a vocabulary-limited random walk so the model has signal
for L in (30000, 90000, 150000):
    vocab = [rng.randrange(1000, 60000) for _ in range(2000)]
    prompt = [vocab[(i * 7 + rng.randrange(3)) % 2000] for i in range(L)]
    r0 = gen(prompt)
    r1 = gen(prompt)
    r1b = gen(prompt)
    time.sleep(8)
    post("/reset_prefix_cache", {})
    time.sleep(2)
    mid = metrics()
    r2 = gen(prompt)
    after = metrics()
    ext = after.get("external_kv_transfer", 0) - mid.get("external_kv_transfer", 0)
    row = {
        "len": L,
        "cached": [r0["cached"], r1["cached"], r1b["cached"], r2["cached"]],
        "external_run2": ext,
        "determinism_local": r1["tok"] == r1b["tok"],
        "agree01": agree(r0, r1),
        "agree12": agree(r1, r2),
        "agree02": agree(r0, r2),
        "d01": lpdiff(r0, r1),
        "d12": lpdiff(r1, r2),
        "d02": lpdiff(r0, r2),
        "conf": [conf(r0), conf(r1), conf(r2)],
        "tok0": r0["tok"],
        "lp0": r0["lp"],
        "tok1": r1["tok"],
        "lp1": r1["lp"],
    }
    row["pass"] = bool(
        row["determinism_local"]
        and ext > 0
        and row["d12"] <= 1.5 * row["d01"] + 0.02
        and abs(row["conf"][2] - row["conf"][1]) < 0.1
    )
    results.append(row)
    print("ROW", json.dumps(row), flush=True)
print("GATE", "PASS" if all(r["pass"] for r in results) else "FAIL", flush=True)
json.dump(results, open(sys.argv[1], "w"), indent=1)
