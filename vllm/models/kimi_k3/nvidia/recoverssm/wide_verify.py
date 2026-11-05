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

Numerics follow the Triton formulas with the layer's FP32 conv1d weight (FP32
products, BF16-rounded convolution outputs, raw gates and recurrence outputs);
FP32 reduction orders differ, and the raw gate is recomputed from F_A on
mma.sync instead of read from the F_B GEMM: it differs from it by one BF16 ulp
at rounding ties, and by the FP32 reduction-order error (about 2^-24 of
sum |F_A F_B|) where the products cancel. With a BF16 conv weight the Triton
convolution rounds each product to BF16 and KdaWide would not match it, so such
layers keep the Triton verify.

Launches go through TVM-FFI straight from the torch tensors on the current
stream (as the CuTe verify does): pure speculative steps run inside FULL CUDA
graphs, but mixed and piecewise steps run ``_forward`` eagerly, where per-launch
host cost counts.
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
    conv_weight_dtype: torch.dtype,
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
    if conv_weight_dtype != torch.float32:
        # The Triton convolution rounds BF16 x BF16 products to BF16.
        return "it matches the Triton convolution only with an FP32 conv1d weight"
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


def _rows_aligned(tensor: torch.Tensor) -> bool:
    """16-byte aligned start and row pitch, unit last stride (activations)."""
    return (
        tensor.data_ptr() % 16 == 0
        and tensor.stride(-1) == 1
        and (tensor.shape[0] <= 1 or tensor.stride(0) * tensor.element_size() % 16 == 0)
    )


def _static_layout_ok(
    conv_w: torch.Tensor,
    conv: torch.Tensor,
    state: torch.Tensor,
    others: tuple[torch.Tensor, ...],
) -> bool:
    """The layer's weights, caches and records as the kernel addresses them:
    16-byte aligned starts (TVM-FFI's assumed alignment), the state's 4-key
    chunks 16 bytes apart (cp.async), a dim-first or token-major conv state."""
    tensors = (conv_w, conv, state, *others)
    if any(t.data_ptr() % 16 for t in tensors):
        return False
    if (
        state.dim() != 4
        or state.stride(-1) != 1
        or any(s % 4 for s in state.stride()[:-1])
    ):
        return False
    if conv.dim() != 3 or 1 not in conv.stride()[1:]:
        return False
    return conv_w.is_contiguous() and all(t.stride(-1) == 1 for t in others)


class WideRecoverSSMVerify:
    """``KdaWide`` (record mode) on the regular decode's operands of one layer.

    Built for layers ``wide_verify_unsupported`` accepts; ``__call__`` returns
    False (the caller runs the Triton verify) for steps it cannot serve: not
    exactly ``window`` tokens per request, misaligned operands, a variant not
    compiled yet under CUDA graph capture, or a kernel that failed (logged once,
    then never retried).
    """

    # (device, heads, q, bias, conv layout, dtypes, gated RMSNorm, record rows)
    _compiled: dict[tuple, Any] = {}
    _failed: str | None = None

    def __init__(self, layer: torch.nn.Module) -> None:
        self.layer = layer
        self.heads: int = layer.local_num_heads
        self.window: int = layer.spec_query_len
        self.has_bias = layer.conv1d.bias is not None
        # Per KV cache binding: (data pointers, static operands, key, scalars).
        self._binding: tuple | None = None

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
            conv_state: Convolution state as [slots, 3*H*128, >= q+2] (a
                dim-first page, or the transposed view of a token-major one).
            state: FP32 SSM state [slots, H, 128, 128].
            records: The layer's verify records (row- or slot-addressed).
            record_rows: Request-slot record rows [>= B] (non-last PP ranks).
            out: Output rows [B*q, H*128].
            output_gate: With it, the gated RMSNorm is applied to ``out``.

        """
        cls = type(self)
        if cls._failed is not None:
            return False
        rows, rest = divmod(x.shape[0], self.window)
        if (
            rest
            or rows == 0
            or state_indices.shape[0] < rows
            or num_accepted_tokens.shape[0] < rows
            or conv_state.shape[-1] < self.window + 2
            or (record_rows is not None and record_rows.shape[0] < rows)
        ):
            return False
        gate = out if output_gate is None else output_gate
        # TVM-FFI assumes 16-byte aligned starts; activation rows are read in
        # 8- and 16-byte words.
        if (
            not (
                _rows_aligned(x)
                and _rows_aligned(f_a)
                and _rows_aligned(out)
                and _rows_aligned(gate)
                and raw_beta.stride(-1) == 1
            )
            or (
                raw_beta.data_ptr()
                | state_indices.data_ptr()
                | num_accepted_tokens.data_ptr()
            )
            % 16
        ):
            return False
        if record_rows is not None and record_rows.data_ptr() % 16:
            return False
        binding = self._bind(conv_state, state, records)
        if binding is None:
            return False
        _, static, key, scalars = binding
        key = (*key, output_gate is not None, record_rows is not None)
        conv_w, bias, conv, a_log, dt, state, norm_w, f_b, corr, key_gate = static
        tail = (f_b, num_accepted_tokens[:rows], corr, key_gate)
        if record_rows is not None:
            tail += (record_rows[:rows],)
        try:
            compiled = cls._compiled.get(key)
            if compiled is None:
                # Compile in an eager step (warmup), never inside graph capture.
                if x.is_cuda and torch.cuda.is_current_stream_capturing():
                    return False
                compiled = self._compile(key)
            # [B, 1] checkpoint slots (any row stride): KdaWide reads
            # slots[b, accepted[b] - 1] with every accepted count one.
            compiled(
                x, conv_w, bias, conv, f_a, raw_beta, a_log, dt,
                state_indices[:rows].unsqueeze(1), state, out, gate, norm_w,
                *scalars, *tail,
            )  # fmt: skip
            return True
        except Exception as error:  # noqa: BLE001
            cls._failed = f"{type(error).__name__}: {error}"
            logger.warning(
                "The KdaWide RecoverSSM verify failed (%s); using the Triton "
                "verify from now on.",
                cls._failed,
            )
            return False

    def compile(
        self,
        *,
        conv_state_dtype: torch.dtype,
        conv_dim_first: bool,
        slot_records: bool,
    ) -> None:
        """Compile both variants (pure speculative steps with the gated RMSNorm,
        mixed steps without) for the layer's KV cache before CUDA graph capture:
        FULL graphs capture ``_forward``, and with no eager warmup of a size a
        lazy compile would fall inside capture. A failure is logged and keeps
        the Triton verify."""
        layer = self.layer
        records = layer.recoverssm_records
        bias = layer.conv1d.bias if self.has_bias else layer.dt_bias
        device = layer.conv1d.weight.device
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        key = (
            index,
            self.heads,
            self.window,
            self.has_bias,
            conv_dim_first,
            layer.decode_conv1d_weight.dtype,
            bias.dtype,
            conv_state_dtype,
            layer.A_log.dtype,
            layer.dt_bias.dtype,
            torch.float32,  # the SSM state
            layer.decode_norm_weight.dtype,
            layer.f_b_proj.weight.dtype,
            records.correction.dtype,
            records.key_gate.dtype,
        )
        cls = type(self)
        try:
            for norm in (True, False):
                variant = (*key, norm, slot_records)
                if variant not in cls._compiled:
                    self._compile(variant)
        except Exception as error:  # noqa: BLE001
            cls._failed = f"{type(error).__name__}: {error}"
            logger.warning(
                "The KdaWide RecoverSSM verify failed to compile (%s); using the "
                "Triton verify.",
                cls._failed,
            )

    def _bind(
        self,
        conv_state: torch.Tensor,
        state: torch.Tensor,
        records: KDARecoverSSMRecords,
    ) -> tuple | None:
        """The static operands, compile key base and scalars of this KV cache
        binding, checked once per binding (None: the layout does not fit)."""
        layer = self.layer
        conv_w, norm_w = layer.decode_conv1d_weight, layer.decode_norm_weight
        if conv_w is None or norm_w is None:
            return None
        dt = layer.dt_bias
        bias = layer.conv1d.bias if self.has_bias else dt
        static = (
            conv_w,
            bias,
            conv_state,
            layer.A_log,
            dt,
            state,
            norm_w,
            layer.f_b_proj.weight,
            records.correction,
            records.key_gate,
        )
        ptrs = tuple(t.data_ptr() for t in static) + (conv_state.stride(-1),)
        binding = self._binding
        if binding is not None and binding[0] == ptrs:
            return binding
        others = (bias, layer.A_log, dt, norm_w, layer.f_b_proj.weight)
        others += (records.correction, records.key_gate)
        if not _static_layout_ok(conv_w, conv_state, state, others):
            self._binding = None
            return None
        key = (
            state.device.index,
            self.heads,
            self.window,
            self.has_bias,
            conv_state.stride(-1) == 1,  # dim-first conv state
            *(t.dtype for t in static),
        )
        scalars = (
            float(layer.gate_lower_bound),
            HEAD_DIM**-0.5,
            float(layer.o_norm.eps),
        )
        self._binding = (ptrs, static, key, scalars)
        return self._binding

    def _compile(self, key: tuple) -> Any:
        """KdaWide behind TVM-FFI: every tensor a symbolic layout (16-byte
        aligned), the stream TVM-FFI's environment stream (the current one)."""
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream

        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
            architecture,
            fake,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.wide import (
            KdaWide,
        )

        device, heads, window, has_bias, dim_first, *dtypes, norm, rows = key
        conv_w_t, bias_t, conv_t, a_log_t, dt_t, state_t, norm_t, fb_t, corr_t, kg_t = (
            dtypes
        )

        def symbolic(dtype: torch.dtype, rank: int, unit: int = -1) -> Any:
            unit %= rank
            shape = tuple(cute.sym_int() for _ in range(rank))
            stride = tuple(1 if i == unit else cute.sym_int(64) for i in range(rank))
            return fake(shape, stride, dtype)

        bf16, i32 = torch.bfloat16, torch.int32
        operands = (
            symbolic(bf16, 2),  # x [T, 3HK]
            symbolic(conv_w_t, 3),  # conv weight [3, 4, HK]
            symbolic(bias_t, 1),
            symbolic(conv_t, 3, -1 if dim_first else 1),  # [slots, 3HK, q + 2]
            symbolic(bf16, 2),  # f_a [T, 128]
            symbolic(bf16, 2),  # beta [T, H]
            symbolic(a_log_t, 1),
            symbolic(dt_t, 1),
            symbolic(i32, 2),  # checkpoint slots [B, 1]
            symbolic(state_t, 4),  # [slots, H, V, K]
            symbolic(bf16, 2),  # out [T, HV]
            symbolic(bf16, 2),  # output gate [T, HV]
            symbolic(norm_t, 1),
            *(cutlass.Float32(0) for _ in range(3)),  # bound, scale, eps
            symbolic(fb_t, 2),  # F_B [HK, 128]
            symbolic(i32, 1),  # accepted [B]
            symbolic(corr_t, 4),  # corrections [rows, H, q, V]
            symbolic(kg_t, 4),  # keys and raw gates [rows, H, q, 2K]
            *((symbolic(i32, 1),) if rows else ()),  # record rows [B]
        )
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        kernel = KdaWide(
            heads,
            window,
            record=True,
            gate_mode=0,
            has_bias=has_bias,
            ahead=2 * sms,
            norm=norm,
        )
        with torch.accelerator.device_index(device):
            compiled = cute.compile(
                kernel,
                *operands,
                make_fake_stream(use_tvm_ffi_env_stream=True),
                options=f"--gpu-arch={architecture(device)} --enable-tvm-ffi",
            )
        type(self)._compiled[key] = compiled
        logger.info_once(
            "Compiled the KdaWide RecoverSSM verify (%d heads, q %d, %s, %s).",
            heads,
            window,
            "gated RMSNorm" if norm else "pre-norm output",
            "slot-addressed records" if rows else "row-addressed records",
        )
        return compiled
