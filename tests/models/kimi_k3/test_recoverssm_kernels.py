# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CuTe KDA RecoverSSM verify and state commit against the Triton reference
(``causal_conv1d_update`` + ``kda_recoverssm_verify``, and the Triton commit), on
Mamba page views as vLLM binds them (one padded page per slot holding the conv
and SSM state) and per-layer verify records indexed by batch row."""

from math import prod
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip("The CuTe RecoverSSM kernels need SM10x", allow_module_level=True)
pytest.importorskip("cutlass")

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (  # noqa: E402
    KDARecoverSSMCommitContext,
    KDARecoverSSMRecords,
    kda_recoverssm_verify,
)
from vllm.models.kimi_k3.nvidia.recoverssm.kernels import (  # noqa: E402
    CuteRecoverSSMVerify,
    compile_state_commit,
)

D = 128
LOWER_BOUND = -5.0
DEVICE = torch.device("cuda")
# Extra bytes per page: vLLM pads Mamba pages to the attention page size.
PAGE_PADDING = 4096


def _pages(
    slots: int,
    heads: int,
    window: int,
    dim_first: bool,
    state_dtype: torch.dtype = torch.bfloat16,
    conv_dtype: torch.dtype = torch.bfloat16,
):
    """Page views of (conv state, SSM state) with a padded slot stride, filled
    with random values."""
    conv_dim, conv_len = 3 * heads * D, window + 2
    conv_shape = (
        (slots, conv_dim, conv_len) if dim_first else (slots, conv_len, conv_dim)
    )
    specs = [
        (conv_dtype, conv_shape),
        (state_dtype, (slots, heads, D, D)),
    ]
    sizes = [dtype.itemsize * prod(shape[1:]) for dtype, shape in specs]
    page = sum(sizes) + PAGE_PADDING
    raw = torch.randn(slots, page // 4, device=DEVICE).view(torch.uint8)
    views = []
    for i, (dtype, shape) in enumerate(specs):
        inner = torch.empty(shape[1:], device="meta").stride()
        views.append(
            raw.view(dtype)
            .view(-1)
            .as_strided(
                shape,
                (page // dtype.itemsize, *inner),
                sum(sizes[:i]) // dtype.itemsize,
            )
        )
    conv, state = views
    # Bounded values: random bits are not finite floats.
    conv.copy_(torch.randn_like(conv))
    state.copy_(torch.randn_like(state) * 0.1)
    return (raw, *views)


def _records(rows: int, heads: int, window: int) -> KDARecoverSSMRecords:
    records = KDARecoverSSMRecords.allocate(
        rows, heads, window, D, torch.bfloat16, DEVICE
    )
    records.correction.normal_()
    records.key_gate.normal_()
    return records


def _clone_records(records: KDARecoverSSMRecords) -> KDARecoverSSMRecords:
    return KDARecoverSSMRecords(records.correction.clone(), records.key_gate.clone())


def _clone_pages(pages):
    raw = pages[0].clone()
    base = pages[0].data_ptr()
    return (
        raw,
        *(
            raw.view(t.dtype)
            .view(-1)
            .as_strided(t.shape, t.stride(), (t.data_ptr() - base) // t.element_size())
            for t in pages[1:]
        ),
    )


def _inputs(
    batch: int, heads: int, window: int, varlen: bool, row_pad: int, row_offset: int
):
    """Raw projections laid out as the KDA in-projection output: q/k/v, then the
    output gate, f_a, beta and the alignment padding in one row; the spec-decode
    tokens start ``row_offset`` rows in (after a batch's non-spec tokens)."""
    lens = torch.full((batch,), window, dtype=torch.int32)
    if varlen and batch > 2:
        lens[0] = window - 2
        lens[1] = 0
    query_start = torch.zeros(batch + 1, dtype=torch.int32)
    query_start[1:] = torch.cumsum(lens, 0)
    tokens = int(query_start[-1])
    width = 4 * heads * D + D + heads + row_pad
    projected = torch.randn(
        row_offset + tokens, width, device=DEVICE, dtype=torch.bfloat16
    )[row_offset:]
    mixed_qkv = projected[:, : 3 * heads * D]
    raw_beta = projected[:, 4 * heads * D + D : 4 * heads * D + D + heads]
    raw_gate = torch.randn(tokens, heads * D, device=DEVICE, dtype=torch.bfloat16)
    return lens, query_start.to(DEVICE), mixed_qkv, raw_gate, raw_beta


def _compiled_verify(heads, window, dim_first, conv_weight_dtype):
    verify = CuteRecoverSSMVerify(
        num_heads=heads,
        window=window,
        conv_dim_first=dim_first,
        conv_weight_dtype=conv_weight_dtype,
        device=DEVICE,
    )
    verify.compile()
    return verify


def _check_verify(
    heads=96,
    dim_first=True,
    conv_weight_dtype=torch.float32,
    batch=8,
    varlen=False,
    row_pad=0,
    recoverssm_accepted=True,
    window=5,
    row_offset=0,
    dead_slot=0,
):
    """Run the CuTe verify and the Triton reference on the same pages; returns
    whether the CuTe verify launched (it declines unaligned inputs)."""
    torch.manual_seed(batch * 7 + heads + row_pad + window)
    slots = 3 * batch + 1
    lens, query_start, mixed_qkv, raw_gate, raw_beta = _inputs(
        batch, heads, window, varlen, row_pad, row_offset
    )
    tokens = mixed_qkv.shape[0]
    A_log = torch.randn(heads, device=DEVICE) * 0.5
    dt_bias = torch.randn(heads * D, device=DEVICE) * 0.5
    conv_weight = (torch.randn(3 * heads * D, 4, device=DEVICE) * 0.5).to(
        conv_weight_dtype
    )
    accepted = (
        torch.ones(batch, dtype=torch.int32, device=DEVICE)
        if recoverssm_accepted
        else torch.randint(1, window + 1, (batch,), dtype=torch.int32, device=DEVICE)
    )
    state_indices = torch.randperm(slots - 1, device=DEVICE)[:batch].to(torch.int32) + 1
    if varlen and batch > 4:
        # Padded requests: CUDA graph batches pad with the null block.
        state_indices[3:5] = dead_slot

    pages = _pages(slots, heads, window, dim_first)
    ref = _clone_pages(pages)
    initial = _clone_pages(pages)
    _, conv, state = pages
    _, ref_conv, ref_state = ref
    # More rows than requests: rows past the batch are left alone.
    records = _records(batch + 2, heads, window)
    ref_records = _clone_records(records)
    initial_records = _clone_records(records)

    # Reference: the stock convolution, then the Triton verify. (The stock
    # convolution writes out of bounds for slot -1, so it gets the null block.)
    ref_indices = state_indices.clamp(min=0)
    ref_out = torch.full(
        (1, tokens, heads, D), 7.0, device=DEVICE, dtype=torch.bfloat16
    )
    conv_out = torch.empty_like(mixed_qkv)
    causal_conv1d_update(
        mixed_qkv,
        ref_conv if dim_first else ref_conv.transpose(-1, -2),
        conv_weight,
        None,
        activation="silu",
        conv_state_indices=ref_indices,
        num_accepted_tokens=accepted,
        query_start_loc=query_start,
        max_query_len=window,
        validate_data=False,
        out=conv_out,
    )
    q, k, v = (
        x.unflatten(-1, (heads, D)).unsqueeze(0) for x in conv_out.split(heads * D, -1)
    )
    kda_recoverssm_verify(
        q=q,
        k=k,
        v=v,
        raw_g=raw_gate.view(1, tokens, heads, D),
        raw_beta=raw_beta.unsqueeze(0),
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        checkpoint_state=ref_state,
        records=ref_records,
        query_start_loc=query_start,
        state_indices=ref_indices,
        spec_query_len=window,
        out=ref_out,
    )

    verify = _compiled_verify(heads, window, dim_first, conv_weight_dtype)
    out = torch.full((tokens, heads * D), 7.0, device=DEVICE, dtype=torch.bfloat16)
    launched = verify(
        mixed_qkv=mixed_qkv,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        state=state,
        records=records,
        query_start_loc=query_start,
        state_indices=state_indices,
        num_accepted_tokens=accepted,
        conv_state=conv,
        conv_weight=conv_weight,
        out=out,
    )
    torch.accelerator.synchronize()
    if not launched:
        assert (out == 7.0).all()
        for page, before in zip(pages[1:], initial[1:]):
            assert torch.equal(page, before)
        return False
    assert int(verify.counter) == 0  # left zeroed for the next launch

    live = [b for b in range(batch) if int(state_indices[b]) > 0 and int(lens[b]) > 0]
    bounds = query_start.tolist()
    live_tokens = [t for b in live for t in range(bounds[b], bounds[b + 1])]
    dead_tokens = [
        t
        for b in range(batch)
        if int(state_indices[b]) <= 0
        for t in range(bounds[b], bounds[b + 1])
    ]
    ref_out = ref_out.view(tokens, heads * D)
    torch.testing.assert_close(
        out[live_tokens].float(), ref_out[live_tokens].float(), atol=3e-2, rtol=3e-2
    )
    if dead_tokens:
        assert (out[dead_tokens] == 0).all()
    for b in live:
        slot, n = int(state_indices[b]), int(lens[b])
        torch.testing.assert_close(
            records.correction[b, :, :n],
            ref_records.correction[b, :, :n],
            atol=3e-2,
            rtol=3e-2,
        )
        torch.testing.assert_close(
            records.key_gate[b, :, :n], ref_records.key_gate[b, :, :n]
        )
        # The rolled conv state: two history rows, then the raw tokens.
        if dim_first:
            torch.testing.assert_close(
                conv[slot, :, : n + 2], ref_conv[slot, :, : n + 2]
            )
        else:
            torch.testing.assert_close(conv[slot, : n + 2], ref_conv[slot, : n + 2])
    # The checkpoint is never written; other slots' pages and other rows'
    # records are untouched.
    torch.testing.assert_close(state, ref_state, atol=0, rtol=0)
    untouched = sorted(set(range(slots)) - {int(state_indices[b]) for b in live})
    for page, before in zip(pages[1:], initial[1:]):
        assert torch.equal(page[untouched], before[untouched])
    for record, before in zip(
        (records.correction, records.key_gate),
        (initial_records.correction, initial_records.key_gate),
    ):
        assert torch.equal(record[batch:], before[batch:])
    return True


@pytest.mark.parametrize("heads", [96, 12])
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("conv_weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch,varlen", [(1, False), (8, False), (13, True)])
@pytest.mark.parametrize("row_pad", [0, 3])
@pytest.mark.parametrize("recoverssm_accepted", [True, False])
@torch.inference_mode()
def test_cute_verify_matches_triton(
    heads, dim_first, conv_weight_dtype, batch, varlen, row_pad, recoverssm_accepted
):
    """``row_pad`` 3 misaligns the projection rows, so the dim-first layout falls
    back from the warp-specialized verify to the plain tcgen05 one.
    ``recoverssm_accepted`` feeds RecoverSSM's accepted counts (always 1: the
    commit compacts the conv state); otherwise the stock spec-decode counts."""
    assert _check_verify(
        heads, dim_first, conv_weight_dtype, batch, varlen, row_pad, recoverssm_accepted
    )


@pytest.mark.parametrize("window", [2, 8])
@pytest.mark.parametrize("dim_first", [True, False])
@torch.inference_mode()
def test_cute_verify_window_sizes(window, dim_first):
    """1 and 7 speculative tokens (the 1-8 token windows the kernels support)."""
    assert _check_verify(window=window, dim_first=dim_first, batch=13, varlen=True)


@pytest.mark.parametrize("row_pad,launches", [(0, True), (3, False)])
@torch.inference_mode()
def test_cute_verify_token_offset(row_pad, launches):
    """Spec-decode tokens after a mixed batch's other tokens start one row into
    the projection: with 16-byte aligned rows the CuTe verify runs, otherwise it
    declines (writing nothing) and the caller runs the Triton verify."""
    assert _check_verify(row_pad=row_pad, row_offset=1) == launches


@torch.inference_mode()
def test_cute_verify_pad_slot_id():
    """Requests padded with PAD_SLOT_ID (-1) are dead like null-block ones."""
    assert _check_verify(batch=13, varlen=True, dead_slot=-1)


@torch.inference_mode()
def test_cute_verify_rejects_unsupported_state_pages():
    """The CuTe verify rejects state pages the TMA cannot stream: padded value
    rows, or a page off a 16-byte boundary."""
    heads, window, batch = 16, 5, 2
    _, query_start, mixed_qkv, raw_gate, raw_beta = _inputs(
        batch, heads, window, False, 0, 0
    )
    tokens = mixed_qkv.shape[0]
    _, conv, state = _pages(4, heads, window, True)
    verify = _compiled_verify(heads, window, True, torch.float32)
    args = dict(
        mixed_qkv=mixed_qkv,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        A_log=torch.zeros(heads, device=DEVICE),
        dt_bias=torch.zeros(heads * D, device=DEVICE),
        lower_bound=LOWER_BOUND,
        state=state,
        records=_records(batch, heads, window),
        query_start_loc=query_start,
        state_indices=torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE),
        num_accepted_tokens=torch.ones(batch, dtype=torch.int32, device=DEVICE),
        conv_state=conv,
        conv_weight=torch.zeros(3 * heads * D, 4, device=DEVICE),
        out=torch.empty(tokens, heads * D, device=DEVICE, dtype=torch.bfloat16),
    )
    # Padded value rows: the state TMA reads K-element rows back to back.
    padded = torch.zeros(*state.shape[:-1], D + 8, device=DEVICE, dtype=torch.bfloat16)[
        ..., :D
    ]
    with pytest.raises(ValueError, match="state strides"):
        verify(**{**args, "state": padded})
    # A state 8 bytes off a 16-byte boundary.
    shifted = torch.zeros(state.numel() + 8, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="16-byte boundaries"):
        verify(**{**args, "state": shifted[4 : 4 + state.numel()].view(state.shape)})


def _layer(pages, records: KDARecoverSSMRecords, heads: int) -> SimpleNamespace:
    _, conv, state = pages
    return SimpleNamespace(
        pages=pages,
        kv_cache=(conv, state),
        recoverssm_records=records,
        A_log=torch.randn(heads, device=DEVICE) * 0.5,
        dt_bias=torch.randn(heads * D, device=DEVICE) * 0.5,
        local_num_heads=heads,
        head_dim=D,
        gate_lower_bound=LOWER_BOUND,
    )


@pytest.mark.parametrize("heads", [96, 12, 3])
@pytest.mark.parametrize("batch", [1, 8, 13])
@pytest.mark.parametrize("window", [2, 5, 8])
@torch.inference_mode()
def test_cute_state_commit_matches_triton(heads, batch, window, monkeypatch):
    """The CuTe fold of every layer's accepted tokens equals the Triton fold,
    with varlen windows, rejected-everything (0 accepted) and dead requests."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
        lambda: "DS",
    )
    torch.manual_seed(heads + batch + window)
    num_layers = 3
    slots = 2 * batch + 1
    layers = []
    for _ in range(num_layers):
        layers.append(
            _layer(
                _pages(slots, heads, window, dim_first=True),
                _records(batch, heads, window),
                heads,
            )
        )
    lens = torch.full((batch,), window, dtype=torch.int32)
    if batch > 2:
        lens[0] = min(2, window)
    query_start = torch.zeros(batch + 1, dtype=torch.int32)
    query_start[1:] = torch.cumsum(lens, 0)
    query_start = query_start.to(DEVICE)
    accepted = torch.randint(0, window + 1, (batch,), dtype=torch.int32, device=DEVICE)
    state_indices = torch.randperm(slots - 1, device=DEVICE)[:batch].to(torch.int32) + 1
    if batch > 3:
        state_indices[3] = 0

    compile_state_commit(heads, window, DEVICE, align=False)
    initial = [_clone_pages(layer.pages) for layer in layers]
    results = {}
    for cute in (False, True):
        for layer, before in zip(layers, initial):
            layer.pages[0].copy_(before[0])
        context = KDARecoverSSMCommitContext.create(
            layers,
            spec_query_len=window,
            max_num_reqs=batch,
            backend="cute" if cute else "triton",
        )
        context.commit(accepted, state_indices, query_start)
        torch.accelerator.synchronize()
        results[cute] = [layer.kv_cache[1].clone() for layer in layers]
    states_before = [before[2] for before in initial]

    for triton_state, cute_state, before in zip(
        results[False], results[True], states_before
    ):
        torch.testing.assert_close(
            cute_state.float(), triton_state.float(), atol=2e-2, rtol=2e-2
        )
        # Requests with no accepted tokens keep their state, as do unused slots.
        for b in range(batch):
            slot = int(state_indices[b])
            if slot > 0 and int(accepted[b]) == 0:
                assert torch.equal(cute_state[slot], before[slot])
        unused = sorted(set(range(slots)) - set(state_indices.tolist()))
        assert torch.equal(cute_state[unused], before[unused])


@pytest.mark.parametrize("heads", [96, 16])
@torch.inference_mode()
def test_recoverssm_steps_match_stock_spec_decode(heads, monkeypatch):
    """Several verify -> sample -> commit steps of RecoverSSM (CuTe verify and
    commit, one state slot per request) produce the outputs of stock
    speculative decoding (``causal_conv1d_update`` + ``fused_recurrent_kda``
    over 1 + k state slots per request)."""
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_recurrent_kda

    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
        lambda: "DS",
    )
    torch.manual_seed(heads)
    window, batch, steps = 5, 6, 4
    tokens = batch * window
    query_start = torch.arange(0, tokens + 1, window, dtype=torch.int32, device=DEVICE)
    conv_weight = torch.randn(3 * heads * D, 4, device=DEVICE) * 0.5

    # RecoverSSM: one slot per request.
    layer = _layer(
        _pages(batch + 1, heads, window, dim_first=True),
        _records(batch, heads, window),
        heads,
    )
    rss_slots = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    # Stock: 1 + k slots per request; the request's state starts in column 0.
    _, ref_conv, ref_state = _pages(batch * window + 1, heads, window, True)
    ref_slots = torch.arange(
        1, batch * window + 1, dtype=torch.int32, device=DEVICE
    ).view(batch, window)
    ref_conv[ref_slots[:, 0]] = layer.kv_cache[0][rss_slots]
    ref_state[ref_slots[:, 0]] = layer.kv_cache[1][rss_slots]

    verify = _compiled_verify(heads, window, True, torch.float32)
    compile_state_commit(heads, window, DEVICE, align=False)
    context = KDARecoverSSMCommitContext.create(
        [layer], spec_query_len=window, max_num_reqs=batch, backend="cute"
    )
    ones = torch.ones(batch, dtype=torch.int32, device=DEVICE)
    accepted = ones.clone()  # the stock path's counts from the previous step
    for _ in range(steps):
        _, _, mixed_qkv, raw_gate, raw_beta = _inputs(batch, heads, window, False, 0, 0)
        out = torch.empty(tokens, heads * D, device=DEVICE, dtype=torch.bfloat16)
        assert verify(
            mixed_qkv=mixed_qkv,
            raw_gate=raw_gate,
            raw_beta=raw_beta,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=LOWER_BOUND,
            state=layer.kv_cache[1],
            records=layer.recoverssm_records,
            query_start_loc=query_start,
            state_indices=rss_slots,
            num_accepted_tokens=ones,
            conv_state=layer.kv_cache[0],
            conv_weight=conv_weight,
            out=out,
        )
        conv_out = torch.empty_like(mixed_qkv)
        causal_conv1d_update(
            mixed_qkv,
            ref_conv,
            conv_weight,
            None,
            activation="silu",
            conv_state_indices=ref_slots[:, 0],
            num_accepted_tokens=accepted,
            query_start_loc=query_start,
            max_query_len=window,
            validate_data=False,
            out=conv_out,
        )
        q, k, v = (
            x.unflatten(-1, (heads, D)).unsqueeze(0)
            for x in conv_out.split(heads * D, -1)
        )
        ref_out, _ = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            raw_g=raw_gate.view(1, tokens, heads, D),
            raw_beta=raw_beta.unsqueeze(0),
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=LOWER_BOUND,
            initial_state=ref_state,
            cu_seqlens=query_start,
            ssm_state_indices=ref_slots,
            num_accepted_tokens=accepted,
        )
        torch.testing.assert_close(
            out.float(), ref_out.view(tokens, -1).float(), atol=3e-2, rtol=3e-2
        )
        # Sample: accept 1..k+1 tokens per request, then commit them.
        accepted = torch.randint(
            1, window + 1, (batch,), dtype=torch.int32, device=DEVICE
        )
        context.commit(accepted, rss_slots, query_start)


@pytest.mark.parametrize("window", [2, 8])
@torch.inference_mode()
def test_cute_verify_cuda_graph_replay(window):
    """A captured verify replays with new inputs (every launch, including the
    captured one, leaves the work counter zeroed)."""
    heads, batch = 96, 8
    torch.manual_seed(window)
    _, query_start, mixed_qkv, raw_gate, raw_beta = _inputs(
        batch, heads, window, False, 0, 0
    )
    tokens = mixed_qkv.shape[0]
    A_log = torch.randn(heads, device=DEVICE) * 0.5
    dt_bias = torch.randn(heads * D, device=DEVICE) * 0.5
    conv_weight = torch.randn(3 * heads * D, 4, device=DEVICE) * 0.5
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    ones = torch.ones(batch, dtype=torch.int32, device=DEVICE)
    pages = _pages(batch + 1, heads, window, True)
    initial = _clone_pages(pages)
    _, conv, state = pages
    records = _records(batch, heads, window)
    verify = _compiled_verify(heads, window, True, torch.float32)
    out = torch.empty(tokens, heads * D, device=DEVICE, dtype=torch.bfloat16)

    def run():
        assert verify(
            mixed_qkv=mixed_qkv,
            raw_gate=raw_gate,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=LOWER_BOUND,
            state=state,
            records=records,
            query_start_loc=query_start,
            state_indices=slots,
            num_accepted_tokens=ones,
            conv_state=conv,
            conv_weight=conv_weight,
            out=out,
        )

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run()
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            run()
    for replay in range(2):
        pages[0].copy_(initial[0])
        mixed_qkv.normal_()
        graph.replay()
        torch.accelerator.synchronize()
        ref = _clone_pages(initial)
        ref_records = _records(batch, heads, window)
        conv_out = torch.empty_like(mixed_qkv)
        causal_conv1d_update(
            mixed_qkv,
            ref[1],
            conv_weight,
            None,
            activation="silu",
            conv_state_indices=slots,
            num_accepted_tokens=ones,
            query_start_loc=query_start,
            max_query_len=window,
            validate_data=False,
            out=conv_out,
        )
        q, k, v = (
            x.unflatten(-1, (heads, D)).unsqueeze(0)
            for x in conv_out.split(heads * D, -1)
        )
        ref_out = kda_recoverssm_verify(
            q=q,
            k=k,
            v=v,
            raw_g=raw_gate.view(1, tokens, heads, D),
            raw_beta=raw_beta.unsqueeze(0),
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=LOWER_BOUND,
            checkpoint_state=ref[2],
            records=ref_records,
            query_start_loc=query_start,
            state_indices=slots,
            spec_query_len=window,
        )
        torch.testing.assert_close(
            out.float(), ref_out.view(tokens, -1).float(), atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            records.correction, ref_records.correction, atol=3e-2, rtol=3e-2
        )
        assert int(verify.counter) == 0, replay


@pytest.mark.parametrize("window", [2, 5, 8])
@pytest.mark.parametrize("batch", [1, 13])
@torch.inference_mode()
def test_cute_state_commit_align_matches_triton(window, batch, monkeypatch):
    """With prefix caching the commit stores the window's final state to the
    block holding its last token and, when the window crosses a block boundary,
    the state at the boundary to the boundary's block."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
        lambda: "DS",
    )
    torch.manual_seed(window * 31 + batch)
    heads, block_size, width = 12, 8, 4
    slots = 1 + batch * width
    layers = [
        _layer(
            _pages(slots, heads, window, dim_first=True),
            _records(batch, heads, window),
            heads,
        )
        for _ in range(2)
    ]
    query_start = torch.arange(
        0, (batch + 1) * window, window, dtype=torch.int32, device=DEVICE
    )
    accepted = torch.randint(0, window + 1, (batch,), dtype=torch.int32, device=DEVICE)
    # Each request owns blocks 1 + b * width ...; its state is in the block of
    # its last computed token.
    block_table = (
        1 + torch.arange(batch * width, dtype=torch.int32, device=DEVICE)
    ).view(batch, width)
    num_computed = torch.randint(
        1, 2 * block_size, (batch,), dtype=torch.int32, device=DEVICE
    )
    # Request 0 crosses a block boundary with every accepted token.
    num_computed[0] = block_size - 1
    accepted[0] = window
    state_indices = block_table.gather(
        1, ((num_computed - 1) // block_size).long().unsqueeze(1)
    ).squeeze(1)
    if batch > 3:
        state_indices[3] = 0
    compile_state_commit(heads, window, DEVICE, align=True)
    initial = [_clone_pages(layer.pages) for layer in layers]
    results = {}
    for cute in (False, True):
        for layer, before in zip(layers, initial):
            layer.pages[0].copy_(before[0])
        context = KDARecoverSSMCommitContext.create(
            layers,
            spec_query_len=window,
            max_num_reqs=batch,
            backend="cute" if cute else "triton",
        )
        context.commit(
            accepted,
            state_indices,
            query_start,
            block_table=block_table,
            num_computed_tokens=num_computed,
            mamba_block_size=block_size,
        )
        torch.accelerator.synchronize()
        results[cute] = [layer.kv_cache[1].clone() for layer in layers]
    for triton_state, cute_state in zip(results[False], results[True]):
        torch.testing.assert_close(
            cute_state.float(), triton_state.float(), atol=2e-2, rtol=2e-2
        )


@torch.inference_mode()
@torch.inference_mode()
def test_triton_records_at_record_rows_match_records_at_batch_rows(monkeypatch):
    """A non-last PP rank keeps a step's verify records at its requests' slots
    (``record_rows``) until the commit, which runs after the next step's verify:
    verify + commit through any record rows give the outputs and states of the
    batch-row records, and write each request's records only at its slot."""
    # The pages are dim-first (the production DS conv state layout).
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
        lambda: "DS",
    )
    heads, window, batch, num_layers = 16, 5, 6, 2
    torch.manual_seed(0)
    slots = 2 * batch + 1
    lens = torch.full((batch,), window, dtype=torch.int32)
    lens[1] = 2
    query_start = torch.zeros(batch + 1, dtype=torch.int32)
    query_start[1:] = torch.cumsum(lens, 0)
    query_start = query_start.to(DEVICE)
    tokens = int(query_start[-1])
    q, k, v, raw_g = (
        torch.randn(1, tokens, heads, D, device=DEVICE, dtype=torch.bfloat16)
        for _ in range(4)
    )
    raw_beta = torch.randn(1, tokens, heads, device=DEVICE, dtype=torch.bfloat16)
    state_indices = torch.randperm(slots - 1, device=DEVICE)[:batch].to(torch.int32) + 1
    state_indices[3] = 0  # a CUDA graph padding row
    accepted = torch.randint(0, window + 1, (batch,), dtype=torch.int32, device=DEVICE)
    num_rows = batch + 4
    slot_rows = torch.randperm(num_rows, device=DEVICE)[:batch].to(torch.int32)
    base_pages = [_pages(slots, heads, window, dim_first=True) for _ in range(2)]
    params = [
        (torch.randn(heads, device=DEVICE) * 0.5, torch.randn(heads * D, device=DEVICE))
        for _ in range(num_layers)
    ]

    def run(record_rows):
        layers, outs = [], []
        for pages, (A_log, dt_bias) in zip(base_pages, params):
            pages = _clone_pages(pages)
            records = _records(num_rows, heads, window)
            # Any read of a row the verify did not write turns a state NaN.
            records.correction.fill_(float("nan"))
            records.key_gate.fill_(float("nan"))
            outs.append(
                kda_recoverssm_verify(
                    q=q,
                    k=k,
                    v=v,
                    raw_g=raw_g,
                    raw_beta=raw_beta,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    lower_bound=LOWER_BOUND,
                    checkpoint_state=pages[2],
                    records=records,
                    query_start_loc=query_start,
                    state_indices=state_indices,
                    spec_query_len=window,
                    record_rows=record_rows,
                )
            )
            layer = _layer(pages, records, heads)
            layer.A_log, layer.dt_bias = A_log, dt_bias
            layers.append(layer)
        context = KDARecoverSSMCommitContext.create(
            layers, spec_query_len=window, max_num_reqs=batch, backend="triton"
        )
        context.commit(accepted, state_indices, query_start, record_rows=record_rows)
        torch.accelerator.synchronize()
        return outs, layers

    batch_outs, batch_layers = run(None)
    slot_outs, slot_layers = run(slot_rows)

    live = [b for b in range(batch) if int(state_indices[b]) > 0]
    for batch_out, slot_out in zip(batch_outs, slot_outs):
        assert torch.equal(batch_out, slot_out)
    for batch_layer, slot_layer in zip(batch_layers, slot_layers):
        for batch_cache, slot_cache in zip(batch_layer.kv_cache, slot_layer.kv_cache):
            assert torch.equal(batch_cache, slot_cache)
        for b in live:
            row, n = int(slot_rows[b]), int(lens[b])
            for batch_rec, slot_rec in zip(
                (
                    batch_layer.recoverssm_records.correction,
                    batch_layer.recoverssm_records.key_gate,
                ),
                (
                    slot_layer.recoverssm_records.correction,
                    slot_layer.recoverssm_records.key_gate,
                ),
            ):
                assert torch.equal(slot_rec[row, :, :n], batch_rec[b, :, :n])


# KdaWide in the regular decode against the Triton conv + verify (+ o_norm) it
# replaces. The KDA layer's conv1d parameter is FP32 (params_dtype=float32): the
# Triton convolution then multiplies in FP32 like KdaWide (with a BF16 weight it
# rounds every product to BF16, and ~60% of the outputs differ; such layers keep
# the Triton verify). The F_B / F_A gate inputs sit on a 2^-6 / integer grid,
# as in the specialized decode's tests, so KdaWide's mma.sync gate projection
# and the F_B GEMM agree bit for bit; test_wide_verify_random_gates covers
# arbitrary gates.


def _wide_layer(heads, window, conv_weight, f_b):
    """The KDA layer attributes ``WideRecoverSSMVerify`` and the commit read."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.prepared import (
        decode_conv_weight,
    )

    return SimpleNamespace(
        local_num_heads=heads,
        head_dim=D,
        spec_query_len=window,
        conv1d=SimpleNamespace(bias=None),
        decode_conv1d_weight=decode_conv_weight(conv_weight.unsqueeze(1)),
        decode_norm_weight=torch.randn(D, device=DEVICE),
        dt_bias=torch.randn(heads * D, device=DEVICE) * 0.5,
        A_log=torch.randn(heads, device=DEVICE) * 0.5,
        f_b_proj=SimpleNamespace(weight=f_b),
        gate_lower_bound=LOWER_BOUND,
        o_norm=SimpleNamespace(eps=1e-6),
    )


def _wide_weights(heads: int, grid: bool = True):
    """F_B ([H*128, 128] BF16) and the FP32 conv1d weight [3*H*128, 4]."""
    f_b = torch.randn(heads * D, D, device=DEVICE)
    if grid:
        f_b = (f_b * 3).round().clamp(-8, 8).bfloat16() / 64
    else:
        f_b = (f_b * D**-0.5).bfloat16()
    return f_b, torch.randn(3 * heads * D, 4, device=DEVICE) * 0.5


def _wide_step_inputs(tokens: int, heads: int, f_b: torch.Tensor, grid: bool = True):
    """One in-projection row per token (q|k|v, output gate, f_a, beta, padding)
    and the raw gate the regular path's F_B projection feeds the Triton verify."""
    dim = heads * D
    width = 4 * dim + D + heads + 4
    projected = torch.randn(tokens, width, device=DEVICE, dtype=torch.bfloat16)
    if grid:
        f_a = torch.randn(tokens, D, device=DEVICE) * 1.5
        projected[:, 4 * dim : 4 * dim + D] = f_a.round().clamp(-3, 3)
    x, gate = projected[:, : 3 * dim], projected[:, 3 * dim : 4 * dim]
    f_a = projected[:, 4 * dim : 4 * dim + D]
    raw_beta = projected[:, 4 * dim + D : 4 * dim + D + heads]
    raw_gate = (f_a.float() @ f_b.float().T).to(torch.bfloat16)
    return x, gate, f_a, raw_beta, raw_gate


def _triton_regular_verify(
    layer, conv_weight, x, raw_gate, raw_beta, conv, state, records, slots, rows
):
    """The regular path's Triton convolution + verify of q-token requests."""
    heads, window = layer.local_num_heads, layer.spec_query_len
    tokens, dim = x.shape[0], heads * D
    query_start = torch.arange(0, tokens + 1, window, dtype=torch.int32, device=DEVICE)
    conv_out = torch.empty(tokens, 3 * dim, device=DEVICE, dtype=torch.bfloat16)
    causal_conv1d_update(
        x,
        conv,
        conv_weight,
        None,
        activation="silu",
        conv_state_indices=slots,
        num_accepted_tokens=torch.ones_like(slots),
        query_start_loc=query_start,
        max_query_len=window,
        validate_data=False,
        out=conv_out,
    )
    q, k, v = (
        t.unflatten(-1, (heads, D)).unsqueeze(0) for t in conv_out.split(dim, -1)
    )
    out = torch.empty(1, tokens, heads, D, device=DEVICE, dtype=torch.bfloat16)
    kda_recoverssm_verify(
        q=q,
        k=k,
        v=v,
        raw_g=raw_gate.view(1, tokens, heads, D),
        raw_beta=raw_beta.unsqueeze(0),
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        lower_bound=LOWER_BOUND,
        checkpoint_state=state,
        records=records,
        query_start_loc=query_start,
        state_indices=slots,
        spec_query_len=window,
        out=out,
        record_rows=rows,
    )
    return out[0]


def _gated_rms_norm(out, gate, layer):
    """The regular path's o_norm on the Triton verify output [T, H, 128]."""
    out = out.float()
    rstd = torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + layer.o_norm.eps)
    gate = torch.sigmoid(gate.float().view(out.shape))
    return (out * rstd * layer.decode_norm_weight * gate).to(torch.bfloat16)


def _run_wide(layer, x, f_a, raw_beta, conv, state, records, slots, rows, gate):
    """``WideRecoverSSMVerify`` as the regular path calls it; it must launch."""
    from vllm.models.kimi_k3.nvidia.recoverssm.wide_verify import (
        WideRecoverSSMVerify,
    )

    out = torch.full(
        (x.shape[0], layer.local_num_heads * D),
        7.0,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    WideRecoverSSMVerify._failed = None
    assert WideRecoverSSMVerify(layer)(
        x=x,
        f_a=f_a,
        raw_beta=raw_beta,
        state_indices=slots,
        num_accepted_tokens=torch.ones_like(slots),
        conv_state=conv,
        state=state,
        records=records,
        record_rows=rows,
        out=out,
        output_gate=gate,
    ), WideRecoverSSMVerify._failed
    return out


def _assert_matches(got, want, same=0.99, rtol=2e-2, rms_atol=1e-3, rms_max=None):
    """KdaWide vs Triton over [..., 128] rows (a head's channels). Tolerances
    scale with each row's RMS: the gated RMSNorm divides a head by RMS(o), and
    o is ~3e-3 here, so an absolute atol would either hide everything before the
    norm or fail on ulps of large elements after it.

    ``same``: least bit-identical fraction (only FP32 reduction order and
    approximate exp/rsqrt differ: a few rounding ties). Every element within
    ``rtol`` of itself (two BF16 ulps: a tie in o, then one in the norm) plus
    ``rms_atol`` of its row's RMS; ``rms_max``: instead a bound on |d| / RMS
    (gates that differ by an ulp move whole heads a little)."""
    identical = (got.view(torch.int16) == want.view(torch.int16)).float().mean()
    got, want = got.float(), want.float()
    rms = want.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
    diff = (got - want).abs()
    assert identical.item() >= same, f"bit-identical fraction {identical.item():.4f}"
    if rms_max is not None:
        worst = (diff / rms).max().item()
        assert worst < rms_max, f"max |d| / RMS {worst:.3e}"
        return
    bad = diff > rtol * want.abs() + rms_atol * rms
    assert not bad.any(), (
        f"{int(bad.sum())} of {bad.numel()} elements off; worst |d| / RMS "
        f"{(diff / rms).max().item():.3e}"
    )


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("slot_rows", [False, True])
@pytest.mark.parametrize("batch", [1, 16, 48, 96])
@torch.inference_mode()
def test_wide_verify_matches_triton(batch, slot_rows, dim_first, norm):
    """KdaWide in the regular decode (FP32 SSM state, BF16 conv state, FP32
    conv1d weight; TP8 shapes: 12 heads, q 5) against the Triton convolution +
    verify it replaces, and with ``norm`` the gated RMSNorm the regular path
    applies after it: outputs, records (by batch row, or by request slot on
    non-last PP ranks), the conv window and the untouched checkpoint, with a
    CUDA graph padding row (null slot)."""
    heads, window = 12, 5
    torch.manual_seed(batch + 2 * slot_rows + 4 * dim_first + 8 * norm)
    slots, tokens = 3 * batch + 1, batch * window
    f_b, conv_weight = _wide_weights(heads)
    layer = _wide_layer(heads, window, conv_weight, f_b)
    x, gate, f_a, raw_beta, raw_gate = _wide_step_inputs(tokens, heads, f_b)
    state_indices = torch.randperm(slots - 1, device=DEVICE)[:batch].to(torch.int32)
    state_indices += 1
    if batch > 4:
        state_indices[3] = 0  # a CUDA graph padding row
    rows = batch + 4
    record_rows = (
        torch.randperm(rows, device=DEVICE)[:batch].to(torch.int32)
        if slot_rows
        else None
    )
    pages = _pages(slots, heads, window, dim_first, state_dtype=torch.float32)
    ref = _clone_pages(pages)
    records = _records(rows, heads, window)
    ref_records = _clone_records(records)
    initial_records = _clone_records(records)

    def conv_view(conv):
        return conv if dim_first else conv.transpose(-1, -2)

    _, ref_conv, ref_state = ref
    expected = _triton_regular_verify(
        layer,
        conv_weight,
        x,
        raw_gate,
        raw_beta,
        conv_view(ref_conv),
        ref_state,
        ref_records,
        state_indices,
        record_rows,
    )
    if norm:
        expected = _gated_rms_norm(expected, gate, layer)

    _, conv, state = pages
    out = _run_wide(
        layer,
        x,
        f_a,
        raw_beta,
        conv_view(conv),
        state,
        records,
        state_indices,
        record_rows,
        gate if norm else None,
    )
    torch.accelerator.synchronize()

    live = state_indices > 0
    live_tokens = live.repeat_interleave(window)
    _assert_matches(out.unflatten(-1, (heads, D))[live_tokens], expected[live_tokens])
    assert torch.all(out[~live_tokens] == 0)
    # The checkpoint stays as it was; live requests' conv windows are rewritten.
    assert torch.equal(state, ref_state)
    for slot in state_indices[live].tolist():
        assert torch.equal(conv[slot], ref_conv[slot])
    written = record_rows if slot_rows else torch.arange(batch, device=DEVICE)
    live_rows = written[live].long()
    # Keys and raw gates are BF16 copies; corrections are FP32 (V-wide rows).
    _assert_matches(
        records.key_gate[live_rows], ref_records.key_gate[live_rows], same=0.999
    )
    torch.testing.assert_close(
        records.correction[live_rows],
        ref_records.correction[live_rows],
        rtol=1e-2,
        atol=1e-4,
    )
    for row in sorted(set(range(rows)) - set(written.tolist())):
        assert torch.equal(records.correction[row], initial_records.correction[row])
        assert torch.equal(records.key_gate[row], initial_records.key_gate[row])


@pytest.mark.parametrize("norm", [False, True])
@torch.inference_mode()
def test_wide_verify_random_gates(norm):
    """Arbitrary F_A / F_B: KdaWide's mma.sync gate projection rounds to the F_B
    GEMM's BF16 gate except at rounding ties (one ulp), which move a head's
    later outputs slightly."""
    heads, window, batch = 12, 5, 96
    torch.manual_seed(11 + norm)
    slots, tokens = batch + 1, batch * window
    f_b, conv_weight = _wide_weights(heads, grid=False)
    layer = _wide_layer(heads, window, conv_weight, f_b)
    x, gate, f_a, raw_beta, raw_gate = _wide_step_inputs(tokens, heads, f_b, grid=False)
    state_indices = torch.randperm(slots - 1, device=DEVICE)[:batch].to(torch.int32)
    state_indices += 1
    pages = _pages(slots, heads, window, True, state_dtype=torch.float32)
    _, ref_conv, ref_state = _clone_pages(pages)
    records, ref_records = (
        _records(batch, heads, window),
        _records(batch, heads, window),
    )
    expected = _triton_regular_verify(
        layer,
        conv_weight,
        x,
        raw_gate,
        raw_beta,
        ref_conv,
        ref_state,
        ref_records,
        state_indices,
        None,
    )
    if norm:
        expected = _gated_rms_norm(expected, gate, layer)
    _, conv, state = pages
    out = _run_wide(
        layer,
        x,
        f_a,
        raw_beta,
        conv,
        state,
        records,
        state_indices,
        None,
        gate if norm else None,
    )
    torch.accelerator.synchronize()

    # Raw gates: KdaWide's (records) against the F_B GEMM's.
    ours = records.key_gate[..., D:].reshape(batch, heads, window, D)
    gemm = raw_gate.view(batch, window, heads, D).transpose(1, 2)
    ulps = (ours.view(torch.int16).int() - gemm.view(torch.int16).int()).abs()
    assert ulps.max().item() <= 1
    assert (ulps > 0).float().mean().item() < 1e-2
    _assert_matches(out.unflatten(-1, (heads, D)), expected, same=0.95, rms_max=0.1)


@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("slot_rows", [False, True])
@torch.inference_mode()
def test_wide_verify_steps_match_triton(slot_rows, dim_first, conv_dtype, monkeypatch):
    """Verify -> sample -> commit steps with the KdaWide verify and with the
    Triton verify. The Triton commit folds 1..q accepted tokens per request
    (varlen acceptance) from the verify's records, by batch row or by request
    slot (the deferred commit of non-last PP ranks), and compacts the conv
    window the next step's verify reads: conv states stay identical, and SSM
    states and outputs within the verify tolerance, step after step."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
        lambda: "DS" if dim_first else "SD",
    )
    heads, window, batch, steps = 12, 5, 16, 3
    torch.manual_seed(slot_rows + 2 * dim_first + 4 * (conv_dtype == torch.float32))
    slots, tokens = 2 * batch + 1, batch * window
    f_b, conv_weight = _wide_weights(heads)
    layer = _wide_layer(heads, window, conv_weight, f_b)
    state_indices = torch.randperm(slots - 1, device=DEVICE)[:batch].to(torch.int32)
    state_indices += 1
    state_indices[3] = 0  # a CUDA graph padding row
    live = state_indices > 0
    live_tokens = live.repeat_interleave(window)
    live_slots = state_indices[live].long()
    rows = batch + 4
    record_rows = (
        torch.randperm(rows, device=DEVICE)[:batch].to(torch.int32)
        if slot_rows
        else None
    )
    query_start = torch.arange(0, tokens + 1, window, dtype=torch.int32, device=DEVICE)
    base = _pages(
        slots,
        heads,
        window,
        dim_first,
        state_dtype=torch.float32,
        conv_dtype=conv_dtype,
    )
    paths = {}
    for wide in (False, True):
        pages = _clone_pages(base)
        records = _records(rows, heads, window)
        commit_layer = SimpleNamespace(
            kv_cache=(pages[1], pages[2]),
            recoverssm_records=records,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            local_num_heads=heads,
            head_dim=D,
            gate_lower_bound=LOWER_BOUND,
        )
        context = KDARecoverSSMCommitContext.create(
            [commit_layer], spec_query_len=window, max_num_reqs=batch, backend="triton"
        )
        paths[wide] = (pages, records, context)

    for _ in range(steps):
        x, _, f_a, raw_beta, raw_gate = _wide_step_inputs(tokens, heads, f_b)
        # Sampled after the verify: 1 (bonus only) to q tokens per request.
        accepted = torch.randint(
            1, window + 1, (batch,), dtype=torch.int32, device=DEVICE
        )
        accepted[0], accepted[1] = 1, window
        outs = {}
        for wide, ((_, conv, state), records, context) in paths.items():
            conv = conv if dim_first else conv.transpose(-1, -2)
            if wide:
                outs[wide] = _run_wide(
                    layer,
                    x,
                    f_a,
                    raw_beta,
                    conv,
                    state,
                    records,
                    state_indices,
                    record_rows,
                    None,
                ).unflatten(-1, (heads, D))
            else:
                outs[wide] = _triton_regular_verify(
                    layer,
                    conv_weight,
                    x,
                    raw_gate,
                    raw_beta,
                    conv,
                    state,
                    records,
                    state_indices,
                    record_rows,
                )
            context.commit(
                accepted, state_indices, query_start, record_rows=record_rows
            )
        torch.accelerator.synchronize()
        _assert_matches(outs[True][live_tokens], outs[False][live_tokens])
        (_, ref_conv, ref_state), (_, conv, state) = paths[False][0], paths[True][0]
        assert torch.equal(conv[live_slots], ref_conv[live_slots])
        torch.testing.assert_close(
            state[live_slots], ref_state[live_slots], rtol=1e-2, atol=1e-3
        )


@pytest.mark.parametrize("slot_rows", [False, True])
@torch.inference_mode()
def test_wide_verify_cuda_graph_replay(slot_rows):
    """FULL CUDA graphs capture the regular path's ``_forward``: the TVM-FFI
    KdaWide launch (on the capture stream, compiled beforehand by an eager
    step, as ``WideRecoverSSMVerify.compile`` does at layer init) replays with
    new activations and slots, as the Triton verify computes them."""
    from vllm.models.kimi_k3.nvidia.recoverssm.wide_verify import (
        WideRecoverSSMVerify,
    )

    heads, window, batch = 12, 5, 16
    torch.manual_seed(5 + slot_rows)
    slots, tokens = 2 * batch + 1, batch * window
    f_b, conv_weight = _wide_weights(heads)
    layer = _wide_layer(heads, window, conv_weight, f_b)
    x, gate, f_a, raw_beta, raw_gate = _wide_step_inputs(tokens, heads, f_b)
    projected = x.as_strided((tokens, x.stride(0)), (x.stride(0), 1))
    state_indices = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    ones = torch.ones(batch, dtype=torch.int32, device=DEVICE)
    rows = batch + 4
    record_rows = (
        torch.randperm(rows, device=DEVICE)[:batch].to(torch.int32)
        if slot_rows
        else None
    )
    pages = _pages(slots, heads, window, True, state_dtype=torch.float32)
    initial = _clone_pages(pages)
    _, conv, state = pages
    records = _records(rows, heads, window)
    out = torch.empty(tokens, heads * D, device=DEVICE, dtype=torch.bfloat16)
    verify = WideRecoverSSMVerify(layer)
    WideRecoverSSMVerify._failed = None

    def run():
        assert verify(
            x=x,
            f_a=f_a,
            raw_beta=raw_beta,
            state_indices=state_indices,
            num_accepted_tokens=ones,
            conv_state=conv,
            state=state,
            records=records,
            record_rows=record_rows,
            out=out,
            output_gate=gate,
        ), WideRecoverSSMVerify._failed

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        state_indices.copy_(torch.arange(1, batch + 1, device=DEVICE))
        run()  # compiles (eagerly)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            run()
    for replay in range(2):
        pages[0].copy_(initial[0])
        projected.normal_()
        f_a.copy_(
            (torch.randn_like(f_a, dtype=torch.float32) * 1.5).round().clamp(-3, 3)
        )
        state_indices.copy_(torch.randperm(slots - 1, device=DEVICE)[:batch] + 1)
        state_indices[3] = 0  # a padding row this time
        graph.replay()
        torch.accelerator.synchronize()
        ref = _clone_pages(initial)
        ref_records = _records(rows, heads, window)
        raw_gate = (f_a.float() @ f_b.float().T).to(torch.bfloat16)
        expected = _triton_regular_verify(
            layer,
            conv_weight,
            x,
            raw_gate,
            raw_beta,
            ref[1],
            ref[2],
            ref_records,
            state_indices,
            record_rows,
        )
        expected = _gated_rms_norm(expected, gate, layer)
        live_tokens = (state_indices > 0).repeat_interleave(window)
        got = out.unflatten(-1, (heads, D))
        _assert_matches(got[live_tokens], expected[live_tokens])
        assert torch.all(out[~live_tokens] == 0), replay
