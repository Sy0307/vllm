# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

requires_flashinfer_mla = pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="FlashInfer MLA requires compute capability 10 or above.",
)


def test_mla_dcp_gathered_query_reserves_backend_head_storage():
    from vllm.v1.attention.ops.dcp import reserve_query_head_storage

    query = torch.randn(3, 24, 576, dtype=torch.bfloat16)

    padded = reserve_query_head_storage(query, 128)

    assert padded.shape == query.shape
    assert padded.stride() == query.stride()
    assert padded.untyped_storage().nbytes() >= 3 * 128 * 576 * 2
    assert torch.equal(padded, query)


@requires_flashinfer_mla
def test_flashinfer_mla_selects_backend_for_gathered_heads():
    from vllm.v1.attention.backends.mla.flashinfer_mla import (
        _select_mla_decode_backend,
    )

    assert _select_mla_decode_backend(6) is None
    assert _select_mla_decode_backend(24) == "cute-dsl"


@requires_flashinfer_mla
def test_flashinfer_mla_forward_uses_gathered_head_count(monkeypatch):
    import vllm.v1.attention.backends.mla.flashinfer_mla as flashinfer_mla

    impl = MagicMock()
    impl.bmm1_scale = 1.0
    impl.bmm2_scale = 1.0
    impl.need_to_return_lse_for_decode = True
    impl.dcp_world_size = 1
    impl.dcp_rank = 0
    impl.cp_kv_cache_interleave_size = 1
    impl.num_heads = 6
    impl.qk_nope_head_dim = 128
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.kv_cache_dtype = "auto"

    attn_metadata = MagicMock()
    attn_metadata.causal = True
    attn_metadata.num_decode_tokens = 2
    attn_metadata.num_decodes = 2
    attn_metadata.max_seq_len = 1
    attn_metadata.decode.block_table = torch.zeros(2, 1, dtype=torch.int32)
    attn_metadata.decode.seq_lens = torch.ones(2, dtype=torch.int32)
    attn_metadata.decode.max_query_len = 1

    query = torch.ones(2, 24, 576, dtype=torch.bfloat16)
    kv_cache = torch.ones(1, 128, 576, dtype=torch.bfloat16)
    kernel = MagicMock(
        return_value=(
            torch.ones(2, 1, 24, 512, dtype=torch.bfloat16),
            torch.ones(2, 24, dtype=torch.float32),
        )
    )
    monkeypatch.setattr(flashinfer_mla, "_get_workspace_buffer", MagicMock())
    monkeypatch.setattr(flashinfer_mla, "trtllm_batch_decode_with_kv_cache_mla", kernel)
    layer = MagicMock()

    output, lse = flashinfer_mla.FlashInferMLAImpl.forward_mqa(
        impl, query, kv_cache, attn_metadata, layer
    )

    assert kernel.call_args.kwargs["backend"] == "cute-dsl"
    assert output.shape == (2, 24, 512)
    assert lse is not None and lse.shape == (2, 24)


@requires_flashinfer_mla
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize("query_lens", [(3, 3), (3, 1)], ids=["uniform", "ragged"])
def test_flashinfer_mla_forward_uses_native_dcp_api(monkeypatch, causal, query_lens):
    import vllm.v1.attention.backends.mla.flashinfer_mla as flashinfer_mla

    impl = MagicMock()
    impl.bmm1_scale = 1.0
    impl.bmm2_scale = 1.0
    impl.need_to_return_lse_for_decode = True
    impl.num_heads = 6
    impl.qk_nope_head_dim = 128
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.kv_cache_dtype = "auto"
    impl.dcp_world_size = 8
    impl.dcp_rank = 3
    impl.cp_kv_cache_interleave_size = 1
    impl._flattened_decode_metadata = MethodType(
        flashinfer_mla.FlashInferMLAImpl._flattened_decode_metadata, impl
    )

    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    query_len = max(query_lens)
    split_rows = not causal or len(set(query_lens)) > 1
    lens = torch.tensor(query_lens)
    block_table = torch.tensor([[10], [20]], dtype=torch.int32)
    seq_lens = torch.tensor([4, 5], dtype=torch.int32)
    global_causal_seq_lens = torch.tensor([31, 35], dtype=torch.int32)
    attn_metadata = MagicMock()
    attn_metadata.causal = causal
    attn_metadata.num_decode_tokens = num_tokens
    attn_metadata.num_decodes = num_reqs
    attn_metadata.max_seq_len = 9
    attn_metadata.decode.block_table = block_table
    attn_metadata.decode.seq_lens = seq_lens
    attn_metadata.decode.dcp_tot_seq_lens = global_causal_seq_lens
    attn_metadata.decode.max_query_len = query_len
    attn_metadata.decode.query_start_loc = torch.tensor(
        [0, *lens.cumsum(0).tolist()], dtype=torch.int32
    )
    attn_metadata.decode.query_len = 0

    query = torch.ones(num_tokens, 24, 576, dtype=torch.bfloat16)
    kv_cache = torch.ones(2, 128, 576, dtype=torch.bfloat16)
    kernel_batch = num_tokens if split_rows else num_reqs
    kernel_query_len = 1 if split_rows else query_len
    kernel = MagicMock(
        return_value=(
            torch.ones(
                kernel_batch,
                kernel_query_len,
                24,
                512,
                dtype=torch.bfloat16,
            ),
            torch.ones(
                kernel_batch,
                kernel_query_len,
                24,
                dtype=torch.float32,
            ),
        )
    )
    monkeypatch.setattr(flashinfer_mla, "_get_workspace_buffer", MagicMock())
    monkeypatch.setattr(flashinfer_mla, "trtllm_batch_decode_with_kv_cache_mla", kernel)

    output, lse = flashinfer_mla.FlashInferMLAImpl.forward_mqa(
        impl, query, kv_cache, attn_metadata, MagicMock()
    )

    call = kernel.call_args.kwargs
    assert call["query"].shape == (kernel_batch, kernel_query_len, 24, 576)
    expected_block_table = block_table
    expected_seq_lens = seq_lens
    expected_global_seq_lens = global_causal_seq_lens
    if split_rows:
        expected_block_table = block_table.repeat_interleave(lens, dim=0)
        expected_seq_lens = seq_lens.repeat_interleave(lens)
        expected_global_seq_lens = global_causal_seq_lens.repeat_interleave(lens)
    if causal and split_rows:
        # Each split row sees KV only up to its own position in its request.
        rows_after = torch.cat([torch.arange(n - 1, -1, -1) for n in query_lens])
        expected_global_seq_lens = expected_global_seq_lens - rows_after.int()
    torch.testing.assert_close(call["block_tables"], expected_block_table)
    torch.testing.assert_close(call["seq_lens"], expected_seq_lens)
    assert call["backend"] == "cute-dsl"
    assert call["enable_dcp"] is True
    assert call["cp_world"] == 8
    assert call["cp_rank"] == 3
    torch.testing.assert_close(
        call["causal_seqlens_kv_global"], expected_global_seq_lens
    )
    assert "multi_ctas_kv_counter_buffer" not in call
    assert flashinfer_mla.FlashInferMLAImpl.lse_base_on_e
    assert output.shape == (num_tokens, 24, 512)
    assert lse is not None and lse.shape == (num_tokens, 24)


# DSpark drafts: a uniform non-causal block of 1 + k query rows per request.
_DRAFT_QUERY = 5


def _draft_builder(monkeypatch, dcp_world, use_pcp=False, device="cpu"):
    """A FlashInfer MLA builder whose MLA-common base is stubbed to the state
    the decode path of build() reads; FlashInfer MLA's own __init__ runs."""
    from vllm.model_executor.layers.attention import mla_attention
    from vllm.v1.attention.backends.mla.flashinfer_mla import (
        FlashInferMLAMetadataBuilder,
    )

    def common_init(
        self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, **_
    ):
        self.metadata_cls = metadata_cls
        self.model_config = SimpleNamespace(get_head_size=lambda: 576)
        self.device = torch.device(device)
        self.use_pcp = use_pcp
        self.dcp_world_size = dcp_world
        self.cp_kv_cache_interleave_size = 1
        self.reorder_batch_threshold = _DRAFT_QUERY
        self.non_causal_multi_token_decode = True
        # A prefill-only buffer: the draft's decode build must leave it alone.
        self.chunked_prefill_workspace = torch.zeros(16, 576, device=device)

    monkeypatch.setattr(mla_attention.MLACommonMetadataBuilder, "__init__", common_init)
    parallel = SimpleNamespace(
        decode_context_parallel_size=dcp_world, cp_kv_cache_interleave_size=1
    )
    return FlashInferMLAMetadataBuilder(
        None, [], SimpleNamespace(parallel_config=parallel), device
    )


def _draft_batch(seq_lens, dcp_world, dcp_rank=3, device="cpu"):
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

    batch = BatchSpec(seq_lens=seq_lens, query_lens=[_DRAFT_QUERY] * len(seq_lens))
    common = create_common_attn_metadata(batch, 64, device, max_block_idx=4096)
    # The runner's block table has a fixed width, whichever batch it holds.
    common.block_table_tensor = torch.randint(
        0, 4096, (len(seq_lens), 128), dtype=torch.int32, device=device
    )
    common.causal = False
    # The speculator passes the draft's configured bound, not the batch's.
    common.max_seq_len = 8192
    if dcp_world > 1:
        common.dcp_local_seq_lens = get_dcp_local_seq_lens(
            common.seq_lens, dcp_world, dcp_rank, 1
        )
    return common


@pytest.mark.parametrize("use_pcp", [False, True])
@pytest.mark.parametrize("dcp_world", [1, 8])
def test_draft_decode_update_flag_matches_build_writes(monkeypatch, dcp_world, use_pcp):
    """The in-place update is a no-op, so the builder may claim support only
    when its decode build writes no builder-owned tensor a captured graph
    reads; PCP stays on the eager rebuild."""
    pytest.importorskip("flashinfer")
    builder = _draft_builder(monkeypatch, dcp_world, use_pcp)
    common = _draft_batch([300, 1025, 77], dcp_world)

    def owned_bytes():
        return {
            name: value.view(torch.uint8).clone()
            for name, value in vars(builder).items()
            if isinstance(value, torch.Tensor)
        }

    before = owned_bytes()
    metadata = builder.build(0, common)
    after = owned_bytes()
    assert before and all(torch.equal(before[k], after[k]) for k in before)
    assert metadata.num_decodes == 3 and metadata.num_prefills == 0
    assert builder.supports_draft_decode_metadata_update is (not use_pcp)


@pytest.mark.parametrize("dcp_world", [1, 8])
def test_draft_decode_update_matches_fresh_build(monkeypatch, dcp_world):
    """A replayed draft graph reads the metadata built at capture. After the
    next batch lands in the same persistent buffers and the in-place update
    runs, every tensor the decode reads matches a fresh build of that batch."""
    pytest.importorskip("flashinfer")
    torch.manual_seed(0)
    builder = _draft_builder(monkeypatch, dcp_world)
    captured = _draft_batch([300, 1025, 77], dcp_world)
    metadata = builder.build(0, captured)

    replayed = _draft_batch([4096, 9, 513], dcp_world)
    # The speculator stages the next batch into the captured buffers (the
    # DCP-local lengths via prepare_dcp_local_seq_lens before the update).
    captured.seq_lens.copy_(replayed.seq_lens)
    captured.query_start_loc.copy_(replayed.query_start_loc)
    captured.block_table_tensor.copy_(replayed.block_table_tensor)
    if dcp_world > 1:
        captured.dcp_local_seq_lens.copy_(replayed.dcp_local_seq_lens)
    builder.update_draft_decode_metadata(metadata)

    expected = _draft_builder(monkeypatch, dcp_world).build(0, replayed)
    decode, fresh = metadata.decode, expected.decode
    assert torch.equal(decode.seq_lens, fresh.seq_lens)
    assert torch.equal(decode.query_start_loc, fresh.query_start_loc)
    assert torch.equal(decode.block_table, fresh.block_table)
    if dcp_world > 1:
        assert torch.equal(decode.dcp_tot_seq_lens, fresh.dcp_tot_seq_lens)
        # Local lengths, not global: the native DCP kernel's per-rank KV.
        assert not torch.equal(decode.seq_lens, decode.dcp_tot_seq_lens)
    else:
        assert decode.dcp_tot_seq_lens is None
    # Host-side decode shape and bound are fixed by the captured batch.
    assert (metadata.num_decodes, metadata.num_decode_tokens) == (
        expected.num_decodes,
        expected.num_decode_tokens,
    )
    assert metadata.max_seq_len == expected.max_seq_len
    assert decode.max_query_len == fresh.max_query_len


@requires_flashinfer_mla
@pytest.mark.parametrize("dcp_world", [1, 8])
def test_draft_graph_replay_after_update_matches_fresh_build(monkeypatch, dcp_world):
    """End to end on the real decode kernel: a draft attention graph captured
    with one batch's metadata, replayed after the next batch is staged into the
    same buffers and the in-place update runs, matches an eager decode of that
    batch with freshly built metadata."""
    import vllm.v1.attention.backends.mla.flashinfer_mla as flashinfer_mla

    torch.manual_seed(0)
    device = "cuda"
    num_heads = 16
    impl = MagicMock()
    impl.bmm1_scale = 576**-0.5
    impl.bmm2_scale = 1.0
    impl.need_to_return_lse_for_decode = dcp_world > 1
    impl.qk_nope_head_dim = 128
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.kv_cache_dtype = "auto"
    impl.dcp_world_size = dcp_world
    impl.dcp_rank = 3 if dcp_world > 1 else 0
    impl._mla_counter_bytes = None
    impl._mla_counter_max_batch = 1024
    impl._flattened_decode_metadata = MethodType(
        flashinfer_mla.FlashInferMLAImpl._flattened_decode_metadata, impl
    )

    def decode(metadata):
        return flashinfer_mla.FlashInferMLAImpl.forward_mqa(
            impl, query, kv_cache, metadata, MagicMock()
        )

    builder = _draft_builder(monkeypatch, dcp_world, device=device)
    captured = _draft_batch([300, 1025, 77], dcp_world, device=device)
    num_tokens = 3 * _DRAFT_QUERY
    query = torch.randn(num_tokens, num_heads, 576, dtype=torch.bfloat16, device=device)
    kv_cache = torch.randn(4096, 64, 576, dtype=torch.bfloat16, device=device)

    # Warm up (JIT, workspaces) with throwaway metadata, then capture with a
    # fresh build, as the draft graph manager does.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode(builder.build(0, captured))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    metadata = builder.build(0, captured)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = decode(metadata)

    replayed = _draft_batch([4096, 9, 513], dcp_world, device=device)
    captured.seq_lens.copy_(replayed.seq_lens)
    captured.query_start_loc.copy_(replayed.query_start_loc)
    captured.block_table_tensor.copy_(replayed.block_table_tensor)
    if dcp_world > 1:
        captured.dcp_local_seq_lens.copy_(replayed.dcp_local_seq_lens)
    builder.update_draft_decode_metadata(metadata)
    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_lse = decode(builder.build(0, replayed))
    assert torch.equal(out, ref_out)
    if dcp_world > 1:
        assert torch.equal(lse, ref_lse)
