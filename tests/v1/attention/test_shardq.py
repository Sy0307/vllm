# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ token split and exchanges (vllm/v1/attention/ops/shardq.py)."""

import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.ops.shardq import (
    fold_split,
    indexer_rows,
    make_shardq_split,
    shard_q,
    unshard_out,
)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("lens", [[4096], [65, 300, 7000], [1000, 64, 64, 129]])
def test_fold_split_partitions_tokens(world, lens):
    qsl = np.r_[0, np.cumsum(lens)]
    per_rank = fold_split(qsl, world)
    merged = np.sort(np.concatenate(per_rank))
    np.testing.assert_array_equal(merged, np.arange(qsl[-1]))
    for idx in per_rank:
        assert np.all(np.diff(idx) > 0)
    # Each rank takes one early and one late piece of every request, so its
    # summed causal positions stay within a couple of tokens per request of
    # the mean.
    positions = [
        sum(int((idx[(idx >= a) & (idx < b)] - a).sum()) for a, b in zip(qsl, qsl[1:]))
        for idx in per_rank
    ]
    assert max(positions) - min(positions) <= sum(4 * n for n in lens)


@pytest.mark.parametrize("num_decode_tokens", [0, 1, 37])
def test_indexer_rows_cover_decodes_then_local_prefill(num_decode_tokens):
    """Mixed steps: every rank scores the decode rows in place, then its own
    prefill tokens compactly; across ranks each prefill row is scored once."""
    world, lens = 4, [700, 64, 3001]
    qsl = np.r_[0, np.cumsum(lens)]
    num_prefill = int(qsl[-1])
    prefill_rows = []
    for rank in range(world):
        split = make_shardq_split(qsl, world, rank, torch.device("cpu"))
        host, rows = indexer_rows(split, num_decode_tokens)
        np.testing.assert_array_equal(host, rows.numpy())
        assert np.all(np.diff(host) > 0)
        np.testing.assert_array_equal(
            host[:num_decode_tokens], np.arange(num_decode_tokens)
        )
        # Row p of the buffer holds the p-th scored row: this rank's prefill
        # tokens follow the decode rows in the order the attention reads them.
        np.testing.assert_array_equal(
            host[num_decode_tokens:] - num_decode_tokens, split.local_idx_cpu
        )
        prefill_rows.append(host[num_decode_tokens:])
    np.testing.assert_array_equal(
        np.sort(np.concatenate(prefill_rows)),
        np.arange(num_decode_tokens, num_decode_tokens + num_prefill),
    )


def _round_trip(rank, world, port, use_sm, q_dtype, num_leading_rows, pad):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.manual_seed(0)
    lens = [700, 64, 3001]
    heads, dim = 4, 576
    full = torch.randn(sum(lens), world * heads, dim, device="cuda").to(q_dtype)
    split = make_shardq_split(
        np.r_[0, np.cumsum(lens)], world, rank, torch.device("cuda")
    )
    local_heads = full[:, rank * heads : (rank + 1) * heads].contiguous()
    group = dist.group.WORLD
    for _ in range(2):  # The second pass reuses the symmetric buffers.
        sharded = shard_q(
            local_heads,
            split,
            group,
            use_sm=use_sm,
            padded_heads=world * heads + pad if pad else None,
        )
        assert sharded.shape[1] == world * heads + pad
        # Bitwise, as the exchanges only move bytes (and fp8 lacks torch.equal).
        expected = full[split.local_idx]
        received = sharded.view(torch.uint8)
        assert torch.equal(received[:, : world * heads], expected.view(torch.uint8))
        assert not received[:, world * heads :].any()
        sharded = sharded[:, : world * heads]
        # Attention outputs are a head-padded slice; unshard reads its strides.
        padded = torch.zeros(sharded.shape[0], world * heads + 8, 512, device="cuda")
        padded[:, : world * heads] = sharded[..., :512].float()
        out = padded.bfloat16()[:, : world * heads]
        restored = unshard_out(
            out, split, group, use_sm=use_sm, num_leading_rows=num_leading_rows
        )
        assert restored.shape[0] == num_leading_rows + sum(lens)
        assert torch.equal(
            restored[num_leading_rows:], local_heads[..., :512].bfloat16()
        )
    dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_sm", [False, True])
@pytest.mark.parametrize("num_leading_rows", [0, 33])
@pytest.mark.parametrize("pad", [0, 8])
def test_shard_q_round_trip(use_sm, q_dtype, num_leading_rows, pad):
    world = 2
    mp.spawn(
        _round_trip,
        args=(world, get_open_port(), use_sm, q_dtype, num_leading_rows, pad),
        nprocs=world,
    )


def _scored_rows(chunks, query_lens, world, rank):
    """Rows each planned chunk scores on ``rank``: its fold rows in the chunk."""
    starts = np.concatenate(([0], np.cumsum(query_lens)))
    own = fold_split(starts, world)[rank]
    scored = []
    for req_slice, query_slice, _ in chunks:
        lo = starts[req_slice.start] + query_slice.start
        hi = starts[req_slice.start] + query_slice.stop
        scored.append(own[(own >= lo) & (own < hi)])
    return own, scored


@pytest.mark.parametrize("rank", [0, 3, 7])
@pytest.mark.parametrize(
    "query_lens,seq_lens",
    [
        ([8192], [700_000]),  # one long-context prefill chunk
        ([5, 3000, 17, 9792], [40, 90_000, 200_000, 350_000]),
        ([9, 1, 2], [9, 300_001, 2]),  # fewer rows than pieces
    ],
)
def test_local_indexer_chunks_score_each_own_row_once(rank, query_lens, seq_lens):
    """A DCP ShardQ step's indexer chunks are planned on the rank's own fold
    rows: together they score each of them exactly once, a chunk's rows times
    its keys stay within the logits budget, and only a request's later
    sub-chunks reuse its gathered keys."""
    from vllm.v1.attention.ops.dcp_shardq.context import plan_local_indexer_chunks

    world, budget = 8, 512 << 20
    query_lens = np.array(query_lens, dtype=np.int64)
    seq_lens = np.array(seq_lens, dtype=np.int64)
    for elem_bytes in (4, 2):
        chunks = plan_local_indexer_chunks(
            query_lens, seq_lens, world, rank, 1 << 22, budget, elem_bytes
        )
        own, scored = _scored_rows(chunks, query_lens, world, rank)
        all_scored = np.concatenate(scored) if scored else np.empty(0, np.int64)
        assert np.array_equal(np.sort(all_scored), own)
        for (req_slice, _, skip), rows in zip(chunks, scored):
            keys = int(seq_lens[req_slice].sum())
            assert len(rows) * keys * elem_bytes <= budget or len(rows) == 1
            prev = chunks[chunks.index((req_slice, _, skip)) - 1]
            assert skip == (
                req_slice.stop - req_slice.start == 1
                and prev[0] == req_slice
                and chunks.index((req_slice, _, skip)) > 0
            )


def test_local_indexer_chunks_fill_the_budget_for_a_long_prefill():
    """The 8192-row chunk of a 700K-token request: planning on all ranks' rows
    leaves each rank about a world-th of the budget's rows per call, cut at
    the fold pieces into short tails (some ranks take 8 calls, down to 8 rows);
    planning on its own rows (BF16) scores every rank's 1024 rows in 3 calls."""
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.attention.ops.dcp_shardq.context import plan_local_indexer_chunks

    world, budget = 8, 512 << 20
    query_lens = np.array([8192], dtype=np.int64)
    seq_lens = np.array([700_000], dtype=np.int64)
    global_plan = [
        (r, q, q.start > 0)
        for r, q in DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
            torch.from_numpy(seq_lens), torch.from_numpy(query_lens), 1 << 22, budget
        )
    ]
    old_calls, new_calls = [], []
    for rank in range(world):
        _, rows = _scored_rows(global_plan, query_lens, world, rank)
        old_calls.append([len(r) for r in rows if len(r)])
        chunks = plan_local_indexer_chunks(
            query_lens, seq_lens, world, rank, 1 << 22, budget, 2
        )
        _, rows = _scored_rows(chunks, query_lens, world, rank)
        new_calls.append([len(r) for r in rows])
    assert all(sorted(calls) == [258, 383, 383] for calls in new_calls)
    assert max(map(len, old_calls)) == 8
    assert min(min(calls) for calls in old_calls) < 16
