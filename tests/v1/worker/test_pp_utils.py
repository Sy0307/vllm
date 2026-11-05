# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from collections import deque
from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import torch

from vllm.v1.worker.gpu import pp_utils


def _batch(
    num_computed,
    prefill_len,
    num_scheduled,
    is_prefilling=None,
    max_seq_len=None,
):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
        is_prefilling_np=(
            np.array(is_prefilling, dtype=np.bool_)
            if is_prefilling is not None
            else None
        ),
        max_seq_len_np=(
            np.array(max_seq_len, dtype=np.int32) if max_seq_len is not None else None
        ),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


def test_excludes_final_prefill_chunk_reaching_its_length_cap():
    """A max_tokens=1 final chunk hands its token off and leaves the engine.

    Disaggregated prefill submits max_tokens=1, so the request finishes with
    the final chunk's single sample and no follow-up step on this engine
    reads the broadcast payload.
    """
    batch = _batch(
        num_computed=[1000],
        prefill_len=[1004],
        num_scheduled=[4],
        is_prefilling=[True],
        max_seq_len=[1005],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_final_prefill_chunk_with_generation_budget_left():
    """The same final chunk, but the request keeps decoding here."""
    batch = _batch(
        num_computed=[1000],
        prefill_len=[1004],
        num_scheduled=[4],
        is_prefilling=[True],
        max_seq_len=[1004 + 256],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_length_cap_overrun_decode_row_is_kept():
    """Decoding rows are never excluded, even past their length cap.

    Speculative decoding can transiently overrun prompt_len + max_tokens
    while the scheduler still runs the request; dropping the row would
    desynchronize the PP stages.
    """
    batch = _batch(
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
        is_prefilling=[False],
        max_seq_len=[12176],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_finishing_prefill_row_does_not_drop_a_sibling_row():
    """Only finishing rows are excluded; a decoding sibling keeps its row."""
    batch = _batch(
        num_computed=[1000, 10],
        prefill_len=[1004, 8],
        num_scheduled=[4, 1],
        is_prefilling=[True, False],
        max_seq_len=[1005, 1024],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def _deferred_handler(monkeypatch, log, delay):
    """A non-last-rank PPHandler on CPU whose collectives append to `log`."""
    handler = pp_utils.PPHandler.__new__(pp_utils.PPHandler)
    handler.is_last_rank = False
    handler.last_rank = 1
    handler.num_speculative_steps = 2
    handler.max_sample_len = 3
    handler.device = torch.device("cpu")
    handler.main_stream = Mock()
    handler.broadcast_stream = Mock()
    handler.broadcast_group = None
    handler.queue = deque([None, None])
    handler.req_idx_gen_np = np.zeros(4, dtype=np.int32)
    handler.deferred_recv_launch_delay = delay
    handler.recv_launch_delay = delay
    handler.pending_post_model_receive = None
    monkeypatch.setattr(pp_utils.torch.cuda, "stream", lambda s: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None)
    monkeypatch.setattr(
        pp_utils.torch.distributed,
        "broadcast",
        lambda tensor, src, group: log.append(("recv", tensor)),
    )
    return handler


def _decode_batch():
    batch = _batch(num_computed=[10, 20], prefill_len=[5, 5], num_scheduled=[3, 3])
    batch.idx_mapping_np = np.array([0, 1], dtype=np.int32)
    batch.idx_mapping = torch.tensor([0, 1])
    batch.num_draft_tokens = 4
    return batch


def _run_steps(handler, log, num_steps, first_step=0):
    """Drive the non-last-rank call order of GPUModelRunner for each step."""
    tensors = {}
    for step in range(first_step, first_step + num_steps):
        out = handler.get_prev_sampled_outputs()
        log.append(("consume", step, None if out is None else out["num_sampled"]))
        log.append(("model", step))
        handler.launch_post_model_receive()
        handler.receive(
            _decode_batch(),
            recv_drafts=True,
            restore_model_state=lambda step=step: log.append(("restore", step)),
        )
        slot = handler.queue[-1]
        for t in (slot.sampled_tokens, slot.combined, slot.draft_tokens):
            tensors[id(t)] = step
    return tensors


def _posted_steps(log, tensors):
    return [tensors[id(entry[1])] for entry in log if entry[0] == "recv"]


def test_deferred_receive_posts_after_next_model_in_step_order(monkeypatch):
    """The receive for step T waits until step T+1's model kernels are queued,
    so its NCCL kernel never sits on SMs that step T+1's forward needs, and
    receives are still posted once each, in the last rank's broadcast order."""
    log: list = []
    handler = _deferred_handler(monkeypatch, log, delay=1)
    tensors = _run_steps(handler, log, num_steps=5)

    # Three broadcasts (tokens, counts, drafts) per step, in step order; the
    # last step's receive waits for a model that has not run yet.
    assert _posted_steps(log, tensors) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    for step in range(3):
        first_recv = next(
            i for i, e in enumerate(log) if e[0] == "recv" and tensors[id(e[1])] == step
        )
        assert log.index(("model", step + 1)) < first_recv
        # Posted before the step that consumes it (step + 2) starts.
        consume = next(i for i, e in enumerate(log) if e[:2] == ("consume", step + 2))
        assert first_recv < consume
        assert log[consume][2] is not None

    handler.flush_pending_collectives()
    assert _posted_steps(log, tensors)[12:] == [4, 4, 4]


def test_immediate_receive_before_deferral_is_enabled(monkeypatch):
    """Warmup keeps upstream timing: each receive is posted inside receive()."""
    log: list = []
    handler = _deferred_handler(monkeypatch, log, delay=0)
    tensors = _run_steps(handler, log, num_steps=2)
    first_recv = next(i for i, e in enumerate(log) if e[0] == "recv")
    assert log.index(("model", 0)) < first_recv < log.index(("consume", 1, None))
    assert _posted_steps(log, tensors) == [0, 0, 0, 1, 1, 1]


def test_no_forward_step_posts_pending_receive(monkeypatch):
    """An idle step has no model kernels to wait behind, so the pending receive
    is posted before the next consume, keeping FIFO order with later steps."""
    log: list = []
    handler = _deferred_handler(monkeypatch, log, delay=1)
    tensors = _run_steps(handler, log, num_steps=2)
    # Step 2 is idle: consume, then the no-forward branch posts step 1.
    handler.get_prev_sampled_outputs()
    handler.launch_post_model_receive()
    assert _posted_steps(log, tensors) == [0, 0, 0, 1, 1, 1]


def test_postprocess_state_is_restored_once_right_before_its_step_is_consumed(
    monkeypatch,
):
    """Step T's model state (e.g. the RecoverSSM commit inputs) is restored when
    step T is consumed, after its receive was posted and before the outputs go
    to its postprocess, and never for a step none of whose requests survive."""
    log: list = []
    handler = _deferred_handler(monkeypatch, log, delay=1)
    _run_steps(handler, log, num_steps=4)

    restores = [e[1] for e in log if e[0] == "restore"]
    assert restores == [0, 1]  # steps 2 and 3 are not consumed yet
    for step in restores:
        restore = log.index(("restore", step))
        assert log.index(("model", step + 1)) < restore
        consume = next(i for i, e in enumerate(log) if e[:2] == ("consume", step + 2))
        assert restore < consume

    # Both requests of step 2 are freed before step 4 consumes it.
    handler.on_req_idx_freed(0)
    handler.on_req_idx_freed(1)
    _run_steps(handler, log, num_steps=1, first_step=4)
    assert ("restore", 2) not in log
    assert ("consume", 4, None) in log
