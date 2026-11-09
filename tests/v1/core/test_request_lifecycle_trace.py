# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_REQ_TRACE: one row per request with the stages from arrival to its
first output token."""

import csv

import pytest

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _output(scheduler):
    """One sampled token per request whose prompt is done, none mid-prefill
    (what the model runner returns for a partial prefill)."""
    running = list(scheduler.running)
    return ModelRunnerOutput(
        req_ids=[r.request_id for r in running],
        req_id_to_index={r.request_id: i for i, r in enumerate(running)},
        sampled_token_ids=[
            [1000] if r.num_computed_tokens >= r.num_prompt_tokens else []
            for r in running
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _rows(path):
    with open(path) as f:
        return {row["req"]: row for row in csv.DictReader(f)}


def test_trace_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_REQ_TRACE", raising=False)
    scheduler = create_scheduler()
    assert scheduler._req_trace is None


def test_trace_records_chunked_prefill_to_first_token(monkeypatch, tmp_path):
    path = tmp_path / "req.csv"
    monkeypatch.setenv("VLLM_REQ_TRACE", str(path))
    # 64-token budget: a 100-token prompt needs two chunks, a 20-token prompt
    # waits for the budget the first one leaves.
    scheduler = create_scheduler(max_num_batched_tokens=64)
    (long_req,) = create_requests(num_requests=1, num_tokens=100, req_ids=["long"])
    (short_req,) = create_requests(num_requests=1, num_tokens=20, req_ids=["short"])
    scheduler.add_request(long_req)
    scheduler.add_request(short_req)

    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {long_req.request_id: 64}
    scheduler.update_from_output(out, _output(scheduler))  # no token: mid-prefill

    out = scheduler.schedule()
    assert out.num_scheduled_tokens[long_req.request_id] == 36
    assert out.num_scheduled_tokens["short"] == 20
    scheduler.update_from_output(out, _output(scheduler))

    scheduler._req_trace._file.flush()
    rows = _rows(path)
    assert set(rows) == {long_req.request_id, "short"}
    long_row, short_row = rows[long_req.request_id], rows["short"]
    assert long_row["prompt"] == "100" and long_row["chunks"] == "2"
    assert short_row["prompt"] == "20" and short_row["chunks"] == "1"
    for row in (long_row, short_row):
        assert row["local"] == "0" and row["ext"] == "0" and row["async_load"] == "0"
        stages = ("add", "admit", "first_sched", "last_sched", "first_tok")
        t = [float(row[k]) for k in stages]
        assert t == sorted(t)
    # The short request was admitted only in the second step.
    assert int(short_row["first_step"]) > int(long_row["first_step"])
    assert scheduler._req_trace._recs == {}


def test_trace_writes_row_for_request_aborted_before_first_token(monkeypatch, tmp_path):
    path = tmp_path / "req.csv"
    monkeypatch.setenv("VLLM_REQ_TRACE", str(path))
    scheduler = create_scheduler(max_num_batched_tokens=64)
    (req,) = create_requests(num_requests=1, num_tokens=100)
    scheduler.add_request(req)
    out = scheduler.schedule()
    scheduler.update_from_output(out, _output(scheduler))
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler._req_trace._file.flush()
    row = _rows(path)[req.request_id]
    assert row["first_tok"] == "" and row["chunks"] == "1"
    assert scheduler._req_trace._recs == {}
