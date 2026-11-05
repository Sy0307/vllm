# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which KDA layers and verify steps the KdaWide RecoverSSM verify serves.

Layers it cannot serve keep the Triton verify; so do steps whose operands it
cannot take (not exactly q tokens per request, misaligned rows or caches).
These checks run before any kernel is compiled, so they are CPU tests; the
kernel itself is compared with the Triton verify in test_recoverssm_kernels.py
(SM10x)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.recoverssm.wide_verify import (
    WideRecoverSSMVerify,
    wide_verify_unsupported,
)

HEADS, WINDOW, D = 12, 5, 128


def _eligibility(**overrides):
    args = dict(
        head_dim=D,
        conv_width=4,
        window=WINDOW,
        activation_dtype=torch.bfloat16,
        conv_state_dtype=torch.bfloat16,
        conv_weight_dtype=torch.float32,
        state_dtype=torch.float32,
        lower_bound=-5.0,
        f_b_weight=torch.empty(HEADS * D, D, dtype=torch.bfloat16),
        enabled=True,
        sm100=True,
    )
    args.update(overrides)
    return wide_verify_unsupported(**args)


def test_production_layer_takes_kdawide():
    """TP8 K3.1: 12 local heads, q 5, FP32 SSM state, BF16 (or FP32) conv state,
    the FP32 conv1d parameter."""
    assert _eligibility() is None
    assert _eligibility(conv_state_dtype=torch.float32) is None


@pytest.mark.parametrize(
    "override",
    [
        dict(enabled=False),
        dict(sm100=False),
        dict(head_dim=64),
        dict(conv_width=3),
        dict(window=1),
        dict(window=9),
        dict(activation_dtype=torch.float16),
        # The CuTe verify's BF16 state; KdaWide holds an FP32 one.
        dict(state_dtype=torch.bfloat16),
        dict(conv_state_dtype=torch.float16),
        # The Triton convolution rounds BF16 x BF16 products to BF16; KdaWide
        # multiplies in FP32, which matches it only for an FP32 weight.
        dict(conv_weight_dtype=torch.bfloat16),
        dict(lower_bound=None),
        # A quantized F_B projection cannot feed the gate MMA.
        dict(f_b_weight=torch.empty(HEADS * D, D, dtype=torch.float8_e4m3fn)),
        dict(f_b_weight=torch.empty(HEADS * D, 64, dtype=torch.bfloat16)),
    ],
)
def test_unsupported_layers_keep_triton(override):
    assert _eligibility(**override) is not None


def _layer():
    dim = HEADS * D
    return SimpleNamespace(
        local_num_heads=HEADS,
        spec_query_len=WINDOW,
        conv1d=SimpleNamespace(bias=None),
        decode_conv1d_weight=torch.zeros(3, 4, dim),
        decode_norm_weight=torch.zeros(D),
        dt_bias=torch.zeros(dim),
        A_log=torch.zeros(HEADS),
        f_b_proj=SimpleNamespace(weight=torch.zeros(dim, D, dtype=torch.bfloat16)),
        gate_lower_bound=-5.0,
        o_norm=SimpleNamespace(eps=1e-6),
    )


def _step(requests, tokens_per_request=WINDOW, x_rows=None):
    """A verify step's operands as the regular path passes them (16-byte
    aligned rows of the packed projection, dim-first pages, FP32 state)."""
    tokens = requests * tokens_per_request if x_rows is None else x_rows
    dim, slots = HEADS * D, requests + 1
    projected = torch.zeros(tokens, 4 * dim + D + HEADS + 4, dtype=torch.bfloat16)
    return dict(
        x=projected[:, : 3 * dim],
        f_a=projected[:, 4 * dim : 4 * dim + D],
        raw_beta=projected[:, 4 * dim + D : 4 * dim + D + HEADS],
        state_indices=torch.arange(1, requests + 1, dtype=torch.int32),
        num_accepted_tokens=torch.ones(requests, dtype=torch.int32),
        conv_state=torch.zeros(slots, 3 * dim, WINDOW + 2, dtype=torch.bfloat16),
        state=torch.zeros(slots, HEADS, D, D),
        records=SimpleNamespace(
            correction=torch.zeros(requests, HEADS, WINDOW, D),
            key_gate=torch.zeros(requests, HEADS, WINDOW, 2 * D, dtype=torch.bfloat16),
        ),
        record_rows=None,
        out=torch.zeros(tokens, dim, dtype=torch.bfloat16),
    )


@pytest.fixture
def fresh(monkeypatch):
    """Start from a KdaWide that has not failed; on this CPU-only host a step
    that reaches the kernel fails to build and leaves ``_failed`` set."""
    monkeypatch.setattr(WideRecoverSSMVerify, "_failed", None)


def _declined(verify, step) -> bool:
    """Declined before any build: the Triton verify runs, nothing is disabled."""
    return not verify(**step) and WideRecoverSSMVerify._failed is None


def test_rows_not_a_multiple_of_q_keep_triton(fresh):
    """A step whose verify rows are not whole q-token requests (a request with
    fewer drafts) cannot be split into KdaWide's (request, head) items."""
    assert _declined(WideRecoverSSMVerify(_layer()), _step(3, x_rows=3 * WINDOW - 2))


def test_too_narrow_conv_window_keeps_triton(fresh):
    step = _step(2)
    step["conv_state"] = step["conv_state"][..., : WINDOW + 1]
    assert _declined(WideRecoverSSMVerify(_layer()), step)


def test_too_few_record_rows_keep_triton(fresh):
    step = _step(4)
    step["record_rows"] = torch.zeros(3, dtype=torch.int32)
    assert _declined(WideRecoverSSMVerify(_layer()), step)


def test_misaligned_rows_keep_triton(fresh):
    """An odd row offset into a projection whose rows are not 16-byte multiples
    (the overlap path's beta-like layouts) is declined, not assumed aligned."""
    step = _step(2)
    base = torch.zeros(2 * WINDOW + 1, 3 * HEADS * D + 4, dtype=torch.bfloat16)
    step["x"] = base[1:, : 3 * HEADS * D]
    assert _declined(WideRecoverSSMVerify(_layer()), step)


def test_misaligned_beta_keeps_triton(fresh):
    """TVM-FFI assumes 16-byte aligned tensor starts, beta included."""
    step = _step(2)
    step["raw_beta"] = step["raw_beta"][:, 1:]
    assert _declined(WideRecoverSSMVerify(_layer()), step)


def test_unaligned_state_slots_keep_triton(fresh):
    """State pages whose 4-key chunks are not 16 bytes apart (cp.async)."""
    step = _step(2)
    pages = torch.zeros(3 * (HEADS * D * D + 2))
    step["state"] = pages.as_strided((3, HEADS, D, D), (HEADS * D * D + 2, D * D, D, 1))
    assert _declined(WideRecoverSSMVerify(_layer()), step)


def test_failed_build_falls_back_for_good(fresh):
    """A kernel that fails to build (here: no SM10x toolchain, CPU tensors) is
    logged once; every later step keeps the Triton verify."""
    verify = WideRecoverSSMVerify(_layer())
    assert not verify(**_step(2))
    reason = WideRecoverSSMVerify._failed
    assert reason is not None
    assert not verify(**_step(2))
    assert WideRecoverSSMVerify._failed == reason
