# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime cap on in-flight batches (VLLM_RUNTIME_KNOBS pp_inflight_batches)."""

import json
from collections import deque
from concurrent.futures import Future
from unittest.mock import MagicMock

import vllm.utils.runtime_knobs as rk
from vllm.v1.engine.core import EngineCore


def _core(size: int, knobs):
    core = MagicMock()
    core.batch_queue = deque()
    core.batch_queue_size = size
    core._runtime_knobs = knobs
    core.is_mm_encoder_only = False
    core.is_pooling_model = False
    core.scheduler.has_requests.return_value = True
    core.scheduler.get_grammar_bitmask.return_value = None

    def schedule(*args, **kwargs):
        out = MagicMock()
        out.total_num_scheduled_tokens = 8
        out.pending_structured_output_tokens = False
        return out

    core.scheduler.schedule.side_effect = schedule

    def done():
        f: Future = Future()
        f.set_result(MagicMock())
        return f

    core.model_executor.execute_model.side_effect = lambda *a, **k: done()
    core.model_executor.sample_tokens.side_effect = lambda *a, **k: done()
    return core


def _knobs(tmp_path, monkeypatch, value):
    monkeypatch.setattr(rk, "_POLL_INTERVAL_S", 0.0)
    path = tmp_path / "knobs.json"
    path.write_text(json.dumps({"pp_inflight_batches": value}))
    return rk.RuntimeKnobs(str(path)), path


def test_queue_fills_to_limit_then_blocks(tmp_path, monkeypatch):
    knobs, _ = _knobs(tmp_path, monkeypatch, 2)
    core = _core(4, knobs)
    assert EngineCore.step_with_batch_queue(core)[0] is None
    assert len(core.batch_queue) == 1
    # Second step reaches the cap of 2: it schedules, then drains the oldest.
    EngineCore.step_with_batch_queue(core)
    assert core.scheduler.schedule.call_count == 2
    assert len(core.batch_queue) == 1
    assert core.batch_queue_limit == 2


def test_lowered_limit_only_drains(tmp_path, monkeypatch):
    knobs, path = _knobs(tmp_path, monkeypatch, 4)
    core = _core(4, knobs)
    for _ in range(3):
        EngineCore.step_with_batch_queue(core)
    assert len(core.batch_queue) == 3 and core.scheduler.schedule.call_count == 3
    path.write_text(json.dumps({"pp_inflight_batches": 2}))
    import os

    os.utime(path, ns=(10**18, 10**18))
    # At or over the new cap: no scheduling, the oldest batch drains each step,
    # down to cap - 1 (the steady state at step entry), order kept.
    oldest, second = core.batch_queue[-1], core.batch_queue[-2]
    EngineCore.step_with_batch_queue(core)
    assert core.scheduler.schedule.call_count == 3
    assert len(core.batch_queue) == 2 and oldest not in core.batch_queue
    EngineCore.step_with_batch_queue(core)
    assert core.scheduler.schedule.call_count == 3
    assert len(core.batch_queue) == 1 and second not in core.batch_queue
    # Below the cap again: schedules one, fills to the cap, drains the oldest.
    EngineCore.step_with_batch_queue(core)
    assert core.scheduler.schedule.call_count == 4 and len(core.batch_queue) == 1


def test_no_knobs_keeps_queue_size():
    core = _core(2, None)
    EngineCore.step_with_batch_queue(core)
    assert core.batch_queue_limit == 2 and len(core.batch_queue) == 1
