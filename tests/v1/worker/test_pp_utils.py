# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from collections import deque
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu import pp_utils


@pytest.mark.parametrize(
    ("need_sampled", "freed", "active_slots"),
    [
        ([True, True, True], [], [4, 1, 6]),
        ([True, False, True], [], [4, 6]),
        ([True, True, True], [6], [4, 1]),
        ([True, False, True], [6], [4]),
        ([False, False, False], [], []),
        ([True, True, True], [4, 1, 6], []),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not current_platform.is_cuda(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_received_drafts_preserve_prefill_and_reused_slots(
    need_sampled, freed, active_slots, device
):
    """Only live sampling rows replace drafts after a delayed PP receive."""
    mapping = np.array([4, 1, 6], dtype=np.int32)
    original = torch.arange(7 * 4, device=device).reshape(7, 4)
    target = original.clone()
    received = torch.arange(12, device=device).reshape(3, 4) + 1000
    handler = object.__new__(pp_utils.PPHandler)
    handler.device = torch.device(device)
    handler.main_stream = (
        torch.accelerator.current_stream() if device == "cuda" else Mock()
    )
    event = handler.main_stream.record_event()
    handler.req_idx_gen_np = np.zeros(7, dtype=np.int32)
    handler.queue = deque(
        [
            pp_utils.PendingRecv(
                event=event,
                sampled_tokens=torch.zeros(3, 1, dtype=torch.int64, device=device),
                num_sampled=torch.ones(3, dtype=torch.int32, device=device),
                num_rejected=torch.zeros(3, dtype=torch.int32, device=device),
                idx_mapping=torch.as_tensor(mapping, device=device),
                idx_mapping_np=mapping,
                need_sampled_mask=np.array(need_sampled, dtype=bool),
                gen_at_receive_np=np.zeros(3, dtype=np.int32),
                draft_tokens=received,
            )
        ]
    )
    for request_slot in freed:
        handler.on_req_idx_freed(request_slot)
    output = handler.get_prev_sampled_outputs(target)
    for row, request_slot in enumerate(mapping):
        expected = (
            received[row] if request_slot in active_slots else original[request_slot]
        )
        torch.testing.assert_close(target[request_slot], expected)
    for request_slot in [0, 2, 3, 5]:
        torch.testing.assert_close(target[request_slot], original[request_slot])
    if active_slots:
        assert output is not None
        assert output["idx_mapping"].tolist() == [
            int(slot) if slot in active_slots else -1 for slot in mapping
        ]
    else:
        assert output is None


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
