# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import (
    RecoverSSMState,
    _postprocess_recoverssm_align_kernel,
)
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    preprocess_mamba_align_fused_kernel,
)


def test_prepare_attn_forwards_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=0)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state.recoverssm = None

    positions = torch.tensor([1536], dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        num_reqs_after_padding=1,
        num_tokens_after_padding=1,
        query_start_loc_np=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        num_scheduled_tokens=torch.tensor([1], dtype=torch.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=torch.tensor([1537], dtype=torch.int32),
        seq_lens=torch.tensor([1537], dtype=torch.int32),
        is_prefilling_np=torch.tensor([False]).numpy(),
        dcp_local_seq_lens=None,
        positions=positions,
        prompt_lens=torch.tensor([1024], dtype=torch.int32),
    )
    expected_metadata = {"layer": object()}
    build_attn_metadata = Mock(return_value=expected_metadata)
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)

    metadata = state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )

    assert metadata is expected_metadata
    assert build_attn_metadata.call_args.kwargs["positions"] is positions


@pytest.mark.parametrize("warmup_first", [True, False])
def test_aligned_metadata_and_state_copies_keep_separate_table_bindings(
    monkeypatch: pytest.MonkeyPatch, warmup_first: bool
) -> None:
    """Batch-order metadata must not rebind delayed request-slot state copies."""
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(
        num_speculative_tokens=0,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    state.model_config = SimpleNamespace(max_model_len=8192)
    state.max_num_reqs = 4
    state.device = torch.device("cpu")
    state._align_mode = True
    state.recoverssm = None
    state._mamba_ctx = state._mamba_metadata_ctx = None
    state._mamba_state_copy_funcs = Mock()
    state.num_accepted_tokens_gpu = torch.ones(4, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.zeros(4, dtype=torch.int32)
    state._mamba_src_col_gpu = torch.zeros(4, dtype=torch.int32)
    state._mamba_src_off_gpu = torch.zeros(4, dtype=torch.int32)
    state._get_mamba_group_info = Mock(
        return_value=([0], SimpleNamespace(block_size=8))
    )
    monkeypatch.setattr(
        mamba_hybrid, "preprocess_mamba_align_fused_kernel", MagicMock()
    )
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", Mock(return_value={}))

    class Context:
        is_initialized = False
        run_fused_precopy = Mock()

        def initialize_from_forward_context(self, config, forward, funcs, tables):
            assert not self.is_initialized
            self.tables = tables
            self.is_initialized = True

        def compute_aligned_state_indices(self, seq_lens, num_reqs):
            return torch.stack([table[:num_reqs, :1] for table in self.tables])

    monkeypatch.setattr(
        mamba_hybrid.MambaSpecDecodeGPUContext,
        "create",
        Mock(side_effect=lambda **kwargs: Context()),
    )
    source = torch.full((4, 2), 17, dtype=torch.int32)
    gathered = torch.full((1, 2), 29, dtype=torch.int32)
    builder = SimpleNamespace(mamba_aligned_state_indices=None)
    group = SimpleNamespace(get_metadata_builder=lambda _: builder)
    batch = SimpleNamespace(
        num_reqs=1, num_tokens=1, num_reqs_after_padding=1,
        num_tokens_after_padding=1, max_query_len=1,
        query_start_loc_np=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([1537], dtype=torch.int32),
        seq_lens=torch.tensor([1537], dtype=torch.int32),
        is_prefilling_np=torch.tensor([False]).numpy(), dcp_local_seq_lens=None,
        positions=torch.tensor([1536]), prompt_lens=torch.tensor([1024]),
        idx_mapping=torch.tensor([2], dtype=torch.int32),
    )
    config = Mock()

    def metadata():
        state.prepare_attn(
            batch, CUDAGraphMode.NONE, (gathered,), torch.empty(0),
            [[group]], config, for_capture=True,
        )

    def copies():
        state.preprocess_state(batch, (source,), config, torch.zeros(4))

    first, second = (metadata, copies) if warmup_first else (copies, metadata)
    for value in (29, 41):
        gathered.fill_(value)
        first()
        second()
        assert state._mamba_ctx.tables[0] is source
        torch.testing.assert_close(builder.mamba_aligned_state_indices, gathered[:, :1])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("sampled", "column"), [(0, -1), (1, 0), (3, 1)])
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias(
    sampled: int, column: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([6, 7], dtype=torch.int32, device="cuda"),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, sampled], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    group = SimpleNamespace(layer_names=["layer"])

    state.recoverssm.record_step({"layer": metadata}, [[group]], for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, column, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("accepted_len", [1, 5])
def test_recoverssm_boundary_state_is_copied_into_next_allocated_block(
    accepted_len: int,
) -> None:
    def tensor(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    idx = tensor([0])
    accepted = tensor([accepted_len])
    tracked = tensor([0])
    neutral = accepted.clone()
    _postprocess_recoverssm_align_kernel[(1,)](
        idx,
        accepted,
        None,
        tensor([1664 - accepted_len]),
        tracked,
        neutral,
        MAMBA_BLOCK_SIZE=1664,
        BLOCK_TABLE_WIDTH=3,
    )
    source = tensor([-1])
    bias = tensor([-1])
    preprocess_mamba_align_fused_kernel[(1,)](
        idx,
        tracked,
        tensor([1664]),
        tensor([0, 5]),
        neutral,
        source,
        bias,
        1,
        BLOCK_SIZE=256,
        MAMBA_BLOCK_SIZE=1664,
    )
    assert source.tolist() == [0]
    assert tracked.tolist() == [1]
    assert bias.tolist() == [0]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("null_value", [31.0, float("nan")])
def test_recoverssm_deferred_postprocess_preserves_intervening_request_state(
    null_value: float,
) -> None:
    """A deferred commit must not copy through another batch's live block table."""

    def tensor(values, dtype=torch.int32):
        return torch.tensor(values, device="cuda", dtype=dtype)

    cache = torch.ones((3, 128), device="cuda")
    cache[0].fill_(null_value)
    cache[1].fill_(2)
    before = cache.clone()
    # Request A's saved metadata names block 1. The same live batch row has
    # since been gathered for request B, whose state is in block 2.
    live_table = tensor([[2, 0]])
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.snapshot_for_deferred_commit.return_value = metadata
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=None,
        block_table=tensor([[1, 0]]),
        num_computed_tokens=tensor([7]),
        block_size=8,
    )
    ctx = MambaSpecDecodeGPUContext(
        state_base_addrs=tensor([cache.data_ptr()], torch.int64),
        state_block_strides=tensor([cache.stride(0) * 4], torch.int64),
        state_elem_sizes=tensor([4]),
        state_inner_sizes=tensor([128], torch.int64),
        state_conv_widths=tensor([0]),
        state_group_indices=tensor([0]),
        state_dim_row_count=tensor([0]),
        state_dim_row_stride=tensor([0], torch.int64),
        block_size=8,
        num_states=1,
        mamba_group_ids=[0],
        num_groups=1,
        num_accepted_tokens_out=tensor([1, 1]),
        block_table_ptrs=tensor([live_table.data_ptr()], torch.int64),
        block_table_stride_req=2,
        is_initialized=True,
    )
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = ctx
    state._mamba_state_idx_gpu = tensor([0, 0])
    state.num_accepted_tokens_gpu = tensor([1, 1])
    state.recoverssm = RecoverSSMState()
    state.recoverssm.record_step(
        {"layer": metadata},
        [[SimpleNamespace(layer_names=["layer"])]],
        for_capture=False,
    )
    restore = state.recoverssm.defer_step()
    restore()

    state.postprocess_state(tensor([0]), tensor([1]), tensor([8, 5]))

    # The mocked accepted-state commit writes no cache data. In particular,
    # request B must never receive the null block through legacy postprocess.
    assert torch.equal(cache.view(torch.uint8), before.view(torch.uint8))
    assert state._mamba_state_idx_gpu.tolist() == [0, 0]
