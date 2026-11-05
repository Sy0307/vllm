# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which KDA layers and verify steps the KdaWide RecoverSSM verify serves.

Layers it cannot serve keep the Triton verify; so do steps whose operands it
cannot take (not exactly q tokens per request, misaligned rows). These checks
run before any kernel is compiled, so they are CPU tests; the kernel itself is
compared with the Triton verify in test_recoverssm_kernels.py (SM10x)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.recoverssm.wide_verify import (
    WideRecoverSSMVerify,
    wide_verify_unsupported,
)

HEADS, WINDOW = 12, 5


def _eligibility(**overrides):
    args = dict(
        head_dim=128,
        conv_width=4,
        window=WINDOW,
        activation_dtype=torch.bfloat16,
        conv_state_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        lower_bound=-5.0,
        f_b_weight=torch.empty(HEADS * 128, 128, dtype=torch.bfloat16),
        enabled=True,
        sm100=True,
    )
    args.update(overrides)
    return wide_verify_unsupported(**args)


def test_production_layer_takes_kdawide():
    """TP8 K3.1: 12 local heads, q 5, FP32 SSM state, BF16 conv state."""
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
        dict(lower_bound=None),
        # A quantized F_B projection cannot feed the gate MMA.
        dict(f_b_weight=torch.empty(HEADS * 128, 128, dtype=torch.float8_e4m3fn)),
        dict(f_b_weight=torch.empty(HEADS * 128, 64, dtype=torch.bfloat16)),
    ],
)
def test_unsupported_layers_keep_triton(override):
    assert _eligibility(**override) is not None


def _layer():
    return SimpleNamespace(
        local_num_heads=HEADS,
        spec_query_len=WINDOW,
        conv1d=SimpleNamespace(bias=None),
    )


def _step(requests, tokens_per_request=WINDOW, x_rows=None):
    tokens = requests * tokens_per_request
    dim = HEADS * 128
    projected = torch.zeros(tokens if x_rows is None else x_rows, 4 * dim + 256)
    return dict(
        x=projected[:, : 3 * dim].bfloat16(),
        f_a=torch.zeros(projected.shape[0], 128, dtype=torch.bfloat16),
        raw_beta=torch.zeros(projected.shape[0], HEADS, dtype=torch.bfloat16),
        state_indices=torch.arange(1, requests + 1, dtype=torch.int32),
        num_accepted_tokens=torch.ones(requests, dtype=torch.int32),
        conv_state=torch.zeros(requests + 1, 3 * dim, WINDOW + 2),
        state=torch.zeros(1),
        records=SimpleNamespace(correction=None, key_gate=None),
        record_rows=None,
        out=torch.zeros(projected.shape[0], dim, dtype=torch.bfloat16),
    )


@pytest.fixture
def fresh(monkeypatch):
    """Start from a KdaWide that has not failed; on this CPU-only host any step
    that reaches the kernel fails to build and leaves ``_failed`` set."""
    monkeypatch.setattr(WideRecoverSSMVerify, "_failed", None)


def _declined(verify, step) -> bool:
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
    base = torch.zeros(2 * WINDOW + 1, 3 * HEADS * 128 + 4, dtype=torch.bfloat16)
    step["x"] = base[1:, : 3 * HEADS * 128]
    assert _declined(WideRecoverSSMVerify(_layer()), step)


def test_failed_build_falls_back_for_good(fresh):
    """A kernel that fails to build (here: no SM10x, CPU tensors) is logged once;
    every later step keeps the Triton verify."""
    layer = _layer()
    layer.decode_conv1d_weight = torch.zeros(3, 4, HEADS * 128)
    layer.decode_norm_weight = torch.zeros(128)
    layer.dt_bias = torch.zeros(HEADS * 128)
    layer.A_log = torch.zeros(HEADS)
    layer.f_b_proj = SimpleNamespace(weight=torch.zeros(HEADS * 128, 128))
    layer.gate_lower_bound = -5.0
    layer.o_norm = SimpleNamespace(eps=1e-6)
    verify = WideRecoverSSMVerify(layer)
    assert not verify(**_step(2))
    reason = WideRecoverSSMVerify._failed
    assert reason is not None
    assert not verify(**_step(2))
    assert WideRecoverSSMVerify._failed == reason
