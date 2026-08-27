# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import torch

from vllm.v1.worker.gpu import pp_utils


def test_broadcast_drafts_uses_compact_request_order(monkeypatch):
    handler = pp_utils.PPHandler.__new__(pp_utils.PPHandler)
    handler.is_last_rank = True
    handler.last_rank = 1
    handler.main_stream = object()
    handler.broadcast_stream = Mock()
    handler.broadcast_group = object()

    monkeypatch.setattr(
        pp_utils,
        "compute_need_sampled_mask",
        lambda _batch: np.array([True, True]),
    )
    monkeypatch.setattr(pp_utils.torch.cuda, "stream", lambda _stream: nullcontext())

    sent = []
    monkeypatch.setattr(
        pp_utils.torch.distributed,
        "broadcast",
        lambda tensor, **_kwargs: sent.append(tensor.clone()),
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda _self, _stream: None)

    draft_tokens = torch.tensor([[20, 21], [10, 11]])
    input_batch = Mock(spec=pp_utils.InputBatch)
    input_batch.num_reqs = 2

    handler.broadcast_drafts(draft_tokens, input_batch)

    assert len(sent) == 1
    assert torch.equal(sent[0], draft_tokens)
