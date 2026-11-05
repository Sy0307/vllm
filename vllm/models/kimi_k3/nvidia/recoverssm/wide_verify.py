# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM verify of the regular Kimi-K3 KDA decode on the specialized
decode's CuTe ``KdaWide`` core.

The regular path verifies speculative KDA steps with the Triton causal
convolution update followed by ``kda_recoverssm_verify`` (the CuTe verify in
``recoverssm.cutedsl`` needs a BF16 SSM state, and only the Triton kernels
address records by request slot on non-last PP ranks). ``KdaWide`` keeps the
FP32 state of a (request, head) in registers, fuses the convolution, the F_B
gate projection, the recurrence and the record writes, and addresses records by
row or by request slot (``record_rows``). The commit after sampling stays the
Triton commit, which reads the same records.

Numerics follow the Triton formulas (BF16-rounded convolution outputs, raw
gates and recurrence outputs); FP32 reduction orders differ, and the raw gate
is recomputed from F_A on mma.sync instead of read from the F_B GEMM, so it
can differ from it by one BF16 ulp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMRecords

logger = init_logger(__name__)

HEAD_DIM = 128
CONV_WIDTH = 4
GATE_RANK = 128
# KdaWide verifies 1-8 tokens per request; a RecoverSSM verify has q >= 2.
MIN_WINDOW, MAX_WINDOW = 2, 8


def wide_verify_unsupported(
    *,
    head_dim: int,
    conv_width: int,
    window: int,
    activation_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    state_dtype: torch.dtype,
    lower_bound: float | None,
    f_b_weight: torch.Tensor,
    enabled: bool | None = None,
    sm100: bool | None = None,
) -> str | None:
    """Why ``KdaWide`` cannot run a layer's RecoverSSM verify, or None."""
    if enabled is None:
        enabled = envs.VLLM_KIMI_K3_KDA_WIDE_VERIFY
    if not enabled:
        return "VLLM_KIMI_K3_KDA_WIDE_VERIFY=0"
    if sm100 is None:
        sm100 = current_platform.is_cuda() and (
            current_platform.is_device_capability_family(100)
        )
    if not sm100:
        return "it needs SM10x"
    if head_dim != HEAD_DIM or conv_width != CONV_WIDTH:
        return f"it needs head_dim {HEAD_DIM} and a width-{CONV_WIDTH} convolution"
    if not MIN_WINDOW <= window <= MAX_WINDOW:
        return f"it verifies {MIN_WINDOW} to {MAX_WINDOW} tokens per request"
    if activation_dtype != torch.bfloat16:
        return "it needs bfloat16 activations"
    if state_dtype != torch.float32:
        return "it holds an FP32 SSM state"
    if conv_state_dtype not in (torch.bfloat16, torch.float32):
        return "it needs a bfloat16 or float32 convolution state"
    if lower_bound is None:
        return "it needs a bounded KDA gate (gate_lower_bound)"
    if (
        f_b_weight.dtype != torch.bfloat16
        or f_b_weight.dim() != 2
        or f_b_weight.shape[1] != GATE_RANK
        or f_b_weight.shape[0] % HEAD_DIM != 0
        or not f_b_weight.is_contiguous()
    ):
        return f"it projects the gate from a BF16 [H*{HEAD_DIM}, {GATE_RANK}] F_B"
    return None


def _aligned(tensor: torch.Tensor, align: int) -> bool:
    """Base pointer and every non-unit row pitch aligned to ``align`` bytes."""
    size = tensor.element_size()
    return tensor.data_ptr() % align == 0 and all(
        (stride * size) % align == 0
        for stride, extent in zip(tensor.stride()[:-1], tensor.shape[:-1])
        if extent > 1
    )


class WideRecoverSSMVerify:
    """``KdaWide`` (record mode) on the regular decode's operands of one layer.

    Built for layers ``wide_verify_unsupported`` accepts; ``__call__`` returns
    False (the caller runs the Triton verify) for steps it cannot serve: not
    exactly ``window`` tokens per request, misaligned operands, or a kernel
    that failed to compile (logged once, then never retried).
    """

    _compiled: dict[tuple, Any] = {}
    _failed: str | None = None

    def __init__(self, layer: torch.nn.Module) -> None:
        self.layer = layer
        self.heads: int = layer.local_num_heads
        self.window: int = layer.spec_query_len
        self.has_bias = layer.conv1d.bias is not None
        # (static data pointers, their layouts, CuTe views, scalar operands)
        self._static: tuple | None = None

    def __call__(
        self,
        *,
        x: torch.Tensor,
        f_a: torch.Tensor,
        raw_beta: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        conv_state: torch.Tensor,
        state: torch.Tensor,
        records: KDARecoverSSMRecords,
        record_rows: torch.Tensor | None,
        out: torch.Tensor,
        output_gate: torch.Tensor | None = None,
    ) -> bool:
        """Verify ``x.shape[0] // window`` requests of ``window`` tokens each.

        Args:
            x: Pre-convolution Q|K|V rows [B*q, 3*H*128].
            f_a: Gate low-rank inputs [B*q, 128].
            raw_beta: Pre-sigmoid beta [B*q, H].
            state_indices: Checkpoint slots [>= B] (0 or less: padding).
            num_accepted_tokens: [>= B], all ones under RecoverSSM.
            conv_state: Dim-first convolution state [slots, 3*H*128, >= q+2].
            state: FP32 SSM state [slots, H, 128, 128].
            records: The layer's verify records (row- or slot-addressed).
            record_rows: Request-slot record rows [>= B] (non-last PP ranks).
            out: Output rows [B*q, H*128].
            output_gate: With it, the gated RMSNorm is applied to ``out``.

        """
        cls = type(self)
        if cls._failed is not None:
            return False
        tokens = x.shape[0]
        if tokens == 0 or tokens % self.window:
            return False
        rows = tokens // self.window
        if (
            state_indices.shape[0] < rows
            or num_accepted_tokens.shape[0] < rows
            or conv_state.shape[-1] < self.window + 2
            or (record_rows is not None and record_rows.shape[0] < rows)
        ):
            return False
        batch = (x, f_a, out) + ((output_gate,) if output_gate is not None else ())
        if not all(_aligned(t, 16) for t in batch):
            return False
        layer = self.layer
        conv_weight = layer.decode_conv1d_weight
        norm_weight = layer.decode_norm_weight
        if conv_weight is None or (output_gate is not None and norm_weight is None):
            return False
        dt_bias = layer.dt_bias
        bias = layer.conv1d.bias if self.has_bias else dt_bias
        # [B, 1] checkpoint slots (any row stride): KdaWide reads
        # slots[b, accepted[b] - 1] with every accepted count one.
        slots = state_indices[:rows].unsqueeze(1)
        accepted = num_accepted_tokens[:rows]
        dynamic = [
            x,
            f_a,
            raw_beta,
            slots,
            out,
            output_gate if output_gate is not None else out,
            accepted,
        ]
        if record_rows is not None:
            dynamic.append(record_rows[:rows])
        static = (
            conv_weight,
            bias,
            conv_state,
            layer.A_log,
            dt_bias,
            state,
            norm_weight if norm_weight is not None else dt_bias,
            layer.f_b_proj.weight,
            records.correction,
            records.key_gate,
        )
        try:
            return self._launch(dynamic, static, output_gate is not None)
        except Exception as error:  # noqa: BLE001
            cls._failed = f"{type(error).__name__}: {error}"
            logger.warning(
                "The KdaWide RecoverSSM verify failed (%s); using the Triton "
                "verify from now on.",
                cls._failed,
            )
            return False

    def _launch(
        self, dynamic: list[torch.Tensor], static: tuple[torch.Tensor, ...], norm: bool
    ) -> bool:
        import cutlass
        import cutlass.cute as cute

        from vllm.models.common.ops.moe_tail_fusion.primitives import to_cute
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            stream,
        )

        # The verify runs eagerly every step (``_forward`` breaks the CUDA
        # graph), so the CuTe views of the layer's weights, caches and records
        # are built once per binding, not per launch.
        ptrs = tuple(t.data_ptr() for t in static)
        if self._static is None or self._static[0] != ptrs:
            layouts = tuple(
                (tuple(t.shape), tuple(t.stride()), t.dtype) for t in static
            )
            layer = self.layer
            scalars = tuple(
                cutlass.Float32(v)
                for v in (layer.gate_lower_bound, HEAD_DIM**-0.5, layer.o_norm.eps)
            )
            self._static = (ptrs, layouts, [to_cute(t, 16) for t in static], scalars)
        _, layouts, views, scalars = self._static
        conv_w, bias, conv, a_log, dt, state, norm_w, f_b, corr, key_gate = views
        x, f_a, beta, slots, out, gate, accepted, *rows = (
            to_cute(t, align).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t, align in zip(dynamic, (16, 16, 2, 4, 16, 16, 4, 4))
        )
        operands = (
            *(x, conv_w, bias, conv, f_a, beta, a_log, dt, slots, state, out, gate),
            norm_w,
            *scalars,
            *(f_b, accepted, corr, key_gate, *rows),
        )
        device = dynamic[0].device
        cuda_stream = stream(device)
        key = (
            device.index,
            self.heads,
            self.window,
            norm,
            bool(rows),
            self.has_bias,
            layouts,
            tuple(t.dtype for t in dynamic),
        )
        cls = type(self)
        compiled = cls._compiled.get(key)
        if compiled is None:
            compiled = self._compile(cute, key, operands, cuda_stream, norm)
        compiled(*operands, cuda_stream)
        return True

    def _compile(self, cute: Any, key: tuple, operands: tuple, cuda_stream, norm):
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.wide import (
            KdaWide,
        )

        device_index = key[0]
        sms = torch.cuda.get_device_properties(device_index).multi_processor_count
        kernel = KdaWide(
            self.heads,
            self.window,
            record=True,
            gate_mode=0,
            has_bias=self.has_bias,
            ahead=2 * sms,
            norm=norm,
        )
        with torch.accelerator.device_index(device_index):
            compiled = cute.compile(kernel, *operands, cuda_stream)
        type(self)._compiled[key] = compiled
        logger.info_once(
            "Compiled the KdaWide RecoverSSM verify (%d heads, q %d, %s).",
            self.heads,
            self.window,
            "gated RMSNorm" if norm else "pre-norm output",
        )
        return compiled
