# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded real-sampling Chat Completions acceptance probes on an isolated group."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:18995")
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
assert not args.output.exists(), "Evidence file already exists"
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
results = []


def request(name, body, expected):
    payload = dict(
        model="kimi-k3",
        temperature=0,
        max_tokens=512,
        messages=[{"role": "user", "content": "What is 1 + 1?"}],
    )
    payload.update(body)
    started = time.time()
    req = urllib.request.Request(
        args.url + "/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            status, raw = response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read().decode()
    row = {
        "name": name,
        "request": payload,
        "http_status": status,
        "elapsed_s": time.time() - started,
        "raw_response": raw,
    }
    results.append(row)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    assert status == expected, (name, status, raw[:300])
    if status != 200:
        row["error"] = json.loads(raw)["error"]
        return row
    if not payload.get("stream"):
        data = json.loads(raw)
        choice = data["choices"][0]
        row.update(
            message=choice["message"],
            finish_reason=choice["finish_reason"],
            usage=data.get("usage"),
        )
        return row
    message = {"content": "", "reasoning": "", "tool_calls": []}
    calls = {}
    finish = None
    saw_done = False
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            saw_done = True
            continue
        data = json.loads(data)
        assert "error" not in data, data
        if not data.get("choices"):
            continue
        choice = data["choices"][0]
        finish = choice.get("finish_reason") or finish
        delta = choice["delta"]
        message["content"] += delta.get("content") or ""
        message["reasoning"] += (
            delta.get("reasoning") or delta.get("reasoning_content") or ""
        )
        for call in delta.get("tool_calls") or []:
            target = calls.setdefault(
                call["index"], {"id": "", "function": {"name": "", "arguments": ""}}
            )
            target["id"] = call.get("id") or target["id"]
            function = call.get("function") or {}
            target["function"]["name"] += function.get("name") or ""
            target["function"]["arguments"] += function.get("arguments") or ""
    assert saw_done, name
    message["tool_calls"] = [calls[k] for k in sorted(calls)]
    row.update(message=message, finish_reason=finish)
    return row


for stream in (False, True):
    row = request(
        f"named_function_stream_{stream}",
        {
            "stream": stream,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        },
        400,
    )
    assert "specified' is incompatible with thinking enabled" in row["error"]["message"]
    for where, messages in (
        ("user", [{"role": "user", "content": "hi", "partial": True}]),
        (
            "nonfinal",
            [
                {"role": "assistant", "content": "prefix", "partial": True},
                {"role": "user", "content": "hi"},
            ],
        ),
    ):
        row = request(
            f"partial_{where}_stream_{stream}",
            {"stream": stream, "messages": messages},
            400,
        )
        assert "final assistant" in row["error"]["message"]
    row = request(
        f"partial_valid_stream_{stream}",
        {
            "stream": stream,
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "1 + 1 等于几？只说答案。"},
                {
                    "role": "assistant",
                    "content": "答案是",
                    "partial": True,
                    "reasoning_content": "1 加 1 等于 2。",
                },
            ],
        },
        200,
    )
    message = row["message"]
    assert row["finish_reason"] == "stop", row
    assert "2" in (message.get("content") or ""), row
    assert not (message.get("content") or "").startswith("答案是"), row
    assert not (message.get("reasoning") or message.get("reasoning_content")), row
    for choice in ("auto", "required"):
        row = request(
            f"tool_{choice}_stream_{stream}",
            {
                "stream": stream,
                "tools": tools,
                "tool_choice": choice,
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "user",
                        "content": "请调用 get_weather 查询 Beijing 的天气，"
                        "等工具结果再回答。",
                    }
                ],
            },
            200,
        )
        assert row["finish_reason"] == "tool_calls", row
        calls = row["message"].get("tool_calls") or []
        assert calls and calls[0]["function"]["name"] == "get_weather", row
        json.loads(calls[0]["function"]["arguments"])
args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"passed": len(results), "output": str(args.output)}))
