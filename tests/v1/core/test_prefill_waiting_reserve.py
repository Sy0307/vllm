# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_PREFILL_WAITING_RESERVE: a running prefill chunk leaves part of the
step budget to queued requests instead of taking all of it."""

import pytest

from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _output(scheduler):
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


@pytest.mark.parametrize(
    "reserve, expected", [(0, {"long": 64}), (32, {"long": 32, "short": 20})]
)
def test_running_prefill_leaves_budget_for_waiting(monkeypatch, reserve, expected):
    monkeypatch.setenv("VLLM_PREFILL_WAITING_RESERVE", str(reserve))
    scheduler = create_scheduler(max_num_batched_tokens=64)
    (long_req,) = create_requests(num_requests=1, num_tokens=200, req_ids=["long"])
    scheduler.add_request(long_req)
    out = scheduler.schedule()
    # Nobody waits yet: the long prompt takes the whole budget.
    assert out.num_scheduled_tokens == {"long": 64}
    scheduler.update_from_output(out, _output(scheduler))

    (short_req,) = create_requests(num_requests=1, num_tokens=20, req_ids=["short"])
    scheduler.add_request(short_req)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == expected


def test_reserve_only_applies_while_requests_wait(monkeypatch):
    monkeypatch.setenv("VLLM_PREFILL_WAITING_RESERVE", "32")
    scheduler = create_scheduler(max_num_batched_tokens=64)
    (long_req,) = create_requests(num_requests=1, num_tokens=200, req_ids=["long"])
    scheduler.add_request(long_req)
    for _ in range(2):
        out = scheduler.schedule()
        assert out.num_scheduled_tokens == {"long": 64}
        scheduler.update_from_output(out, _output(scheduler))
