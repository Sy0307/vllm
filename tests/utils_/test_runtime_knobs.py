# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os

import vllm.utils.runtime_knobs as rk


def _write(path, values, mtime_ns):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(values, f)
    os.replace(tmp, path)
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_poll_reads_changes_and_keeps_values_on_bad_json(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "_POLL_INTERVAL_S", 0.0)
    path = tmp_path / "knobs.json"
    knobs = rk.RuntimeKnobs(str(path))
    assert not knobs.poll()  # missing file
    assert knobs.get("a", 7) == 7

    _write(path, {"a": 1}, 1_000_000_000)
    assert knobs.poll()
    assert knobs.get("a", 7) == 1
    assert not knobs.poll()  # unchanged mtime

    path.write_text("{not json")
    os.utime(path, ns=(2_000_000_000, 2_000_000_000))
    assert not knobs.poll()
    assert knobs.get("a", 7) == 1  # previous values kept

    _write(path, {"a": 0, "b": 2}, 3_000_000_000)
    assert knobs.poll()
    assert knobs.get("a", 7) == 0 and knobs.get("b", 0) == 2


def test_poll_interval_limits_stat_calls(tmp_path, monkeypatch):
    path = tmp_path / "knobs.json"
    _write(path, {"a": 1}, 1_000_000_000)
    knobs = rk.RuntimeKnobs(str(path))
    clock = [100.0]
    monkeypatch.setattr(rk.time, "monotonic", lambda: clock[0])
    assert knobs.poll()
    _write(path, {"a": 2}, 2_000_000_000)
    clock[0] += rk._POLL_INTERVAL_S / 2
    assert not knobs.poll() and knobs.get("a", 0) == 1
    clock[0] += rk._POLL_INTERVAL_S
    assert knobs.poll() and knobs.get("a", 0) == 2


def test_unset_env_disables(monkeypatch):
    monkeypatch.delenv("VLLM_RUNTIME_KNOBS", raising=False)
    monkeypatch.setattr(rk, "_KNOBS", None)
    monkeypatch.setattr(rk, "_KNOBS_RESOLVED", False)
    assert rk.get_runtime_knobs() is None
