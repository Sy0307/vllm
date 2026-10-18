# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TRT-LLM Ragged backend for MLA prefill."""

from typing import TYPE_CHECKING, ClassVar

import torch

import vllm.envs as envs
from vllm.v1.attention.backends.mla.prefill.base import (
    MLADimensions,
    MLAPrefillBackend,
)
from vllm.v1.attention.backends.utils import log2_lse_to_ln
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability


import os as _os

_K3_SPLIT_KV = _os.environ.get("VLLM_K3_CONTEXT_SPLIT_KV", "0") == "1"
_K3_SPLIT_MEMO = _os.environ.get("VLLM_K3_SPLIT_MEMO", "0") == "1"
_K3_SPLIT_MERGE_FUSED = _os.environ.get("VLLM_K3_SPLIT_MERGE_FUSED", "0") == "1"


# Experimental B200 history-only backend. Existing split-KV and causal paths
# retain their native kernels. Same input/output precision; arithmetic differs.
_K3_CUTE_CONTEXT = _os.environ.get("VLLM_K3_CUTE_CONTEXT", "0") == "1"
_K3_CUTE_CONTEXT_DIAG = _os.environ.get("VLLM_K3_CUTE_CONTEXT_DIAG", "0") == "1"
_k3_cute_context_counts = {}


def _k3_context_backend(q, k, v, out):
    if not _K3_CUTE_CONTEXT:
        return "trtllm-gen"
    supported = (
        q.is_cuda and torch.cuda.get_device_capability(q.device) == (10, 0)
        and q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
        and out.dtype == torch.bfloat16
        and q.shape[1:] == k.shape[1:] == (12, 192)
        and v.shape[1:] == (12, 128)
        and out.shape == (q.shape[0], 12, 128)
        and all(t.is_contiguous() for t in (q, k, v, out))
        and all(t.data_ptr() % 16 == 0 for t in (q, k, v))
        and out.data_ptr() % 32 == 0
    )
    return "cute-dsl" if supported else "trtllm-gen"


def _k3_cute_indptr(tensor):
    # Contiguous slices may still have a 4/8/12-byte offset. .to(int32) and
    # .contiguous() do not copy an already-contiguous int32 view. No memoization:
    # each invocation must observe this batch's current metadata contents.
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def k3_cute_context_diagnostics(reset=False):
    data = dict(enabled=_K3_CUTE_CONTEXT, diagnostic=_K3_CUTE_CONTEXT_DIAG,
                counts=dict(_k3_cute_context_counts))
    if reset:
        _k3_cute_context_counts.clear()
    return data

from vllm.triton_utils import tl, triton  # noqa: E402


@triton.jit
def _k3_split_merge_kernel(
    o_ptr, lse_ptr, rows_ptr, out_ptr, lse_out_ptr,
    Tq, H,
    o_s0, o_s1, lse_s0, lse_s1, out_s0, out_s1,
    S: tl.constexpr, D: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """Merge S split-KV partials of one query row: LSE-weighted average of the
    partial outputs (base-2 LSE in), natural-log merged LSE out ([H, Tq])."""
    t = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    d = tl.arange(0, D)
    m = tl.full([BLOCK_H], float("-inf"), tl.float32)
    for s in tl.static_range(S):
        r = tl.load(rows_ptr + s * Tq + t).to(tl.int64)
        lse = tl.load(lse_ptr + r * lse_s0 + h * lse_s1, mask=hm, other=float("-inf"))
        m = tl.maximum(m, lse.to(tl.float32))
    denom = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)
    for s in tl.static_range(S):
        r = tl.load(rows_ptr + s * Tq + t).to(tl.int64)
        lse = tl.load(lse_ptr + r * lse_s0 + h * lse_s1, mask=hm, other=float("-inf"))
        w = tl.exp2(lse.to(tl.float32) - m)
        denom += w
        o = tl.load(o_ptr + r * o_s0 + h[:, None] * o_s1 + d[None, :],
                    mask=hm[:, None], other=0.0).to(tl.float32)
        acc += o * w[:, None]
    merged = acc / denom[:, None]
    tl.store(out_ptr + t * out_s0 + h[:, None] * out_s1 + d[None, :],
             merged.to(out_ptr.dtype.element_ty), mask=hm[:, None])
    lse_out = (m + tl.log2(denom)) * 0.6931471805599453
    tl.store(lse_out_ptr + h * Tq + t, lse_out, mask=hm)


def _k3_split_kv_factor(max_query_len: int, num_requests: int, max_seq_len: int) -> int:
    """Segments per request; 1 disables. Tuned on B200 (148 SMs, 12 heads)."""
    if not _K3_SPLIT_KV or max_seq_len < 16384:
        return 1
    # Measured: two ~1K queries already fill enough CTAs; splitting regressed.
    if max_query_len > 512 and num_requests > 1:
        return 1
    if max_query_len <= 256:
        s = 8
    elif max_query_len <= 512:
        s = 4
    elif max_query_len <= 1024:
        s = 2
    else:
        return 1
    # Several requests already add CTAs; keep total virtual requests modest.
    while s > 1 and num_requests * s > 32:
        s //= 2
    # Keep segments long enough to amortize per-CTA setup.
    while s > 1 and max_seq_len // s < 8192:
        s //= 2
    return s


def _k3_split_kv_indices(query_start_loc, cu_seq_lens, num_tokens, splits):
    """GPU-built index tensors for the request-major split layout."""
    qsl = query_start_loc.long()
    qlen = qsl[1:] - qsl[:-1]
    kvlen = (cu_seq_lens[1:] - cu_seq_lens[:-1]).long()
    num_req = qlen.shape[0]
    seg = torch.arange(splits, device=qsl.device)
    base = torch.div(kvlen, splits, rounding_mode="floor")
    rem = kvlen - base * splits
    seg_len = base[:, None] + (seg[None, :] < rem[:, None]).long()  # [R, S]
    vq = qlen.repeat_interleave(splits)
    zero = torch.zeros(1, dtype=torch.long, device=qsl.device)
    vqsl = torch.cat([zero, vq.cumsum(0)])
    vkv = torch.cat([zero, seg_len.flatten().cumsum(0)])
    # Row t of the original chunk -> its row in each segment's partial.
    t = torch.arange(num_tokens, device=qsl.device)
    r = torch.searchsorted(qsl[1:], t, right=True).clamp_(max=num_req - 1)
    off = t - qsl[r]
    part_rows = vqsl[(r * splits)[None, :] + seg[:, None]] + off[None, :]  # [S, Tq]
    # Replicated-query gather: output row j -> source row.
    j = torch.arange(num_tokens * splits, device=qsl.device)
    v = torch.searchsorted(vqsl[1:], j, right=True)
    src = qsl[torch.div(v, splits, rounding_mode="floor")] + (j - vqsl[v])
    return (
        src,
        part_rows,
        vq.to(torch.int32),
        seg_len.flatten().to(torch.int32),
        vqsl.to(torch.int32),
        vkv.to(torch.int32),
    )


class TrtllmRaggedPrefillBackend(MLAPrefillBackend):
    """TRT-LLM Ragged backend for MLA prefill."""

    supported_mla_dimensions: ClassVar[list[MLADimensions]] = [
        MLADimensions(
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ),
        MLADimensions(
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
        ),
    ]

    @staticmethod
    def get_name() -> str:
        return "TRTLLM_RAGGED"

    @classmethod
    def supports_compute_capability(
        cls,
        device_capability: "DeviceCapability | None",
    ) -> bool:
        return device_capability is not None and device_capability.major == 10

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flashinfer.prefill import (
                trtllm_ragged_attention_deepseek,  # noqa: F401
            )

            return True
        except ImportError:
            return False

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )
        self._workspace_spec = (
            (envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,),
            torch.uint8,
        )
        # Reserve capacity without retaining a view: a retained view would pin
        # the old buffer when another user grows the shared workspace.
        current_workspace_manager().get_simultaneous(self._workspace_spec)

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        super().prepare_metadata(prefill_metadata)
        (self._workspace_buffer,) = current_workspace_manager().get_simultaneous(
            self._workspace_spec
        )
        self._query_seq_lens = (
            prefill_metadata.query_start_loc[1:] - prefill_metadata.query_start_loc[:-1]
        )
        query_lens_cpu = prefill_metadata.query_lens_cpu
        if query_lens_cpu is None:
            raise ValueError("TRTLLM ragged prefill requires CPU query lengths")
        if query_lens_cpu.device.type != "cpu":
            raise ValueError("TRTLLM ragged prefill query lengths must be on CPU")
        if query_lens_cpu.ndim != 1 or query_lens_cpu.numel() == 0:
            raise ValueError(
                "TRTLLM ragged prefill requires a non-empty 1D query-length tensor"
            )
        min_query_len = int(torch.min(query_lens_cpu).item())
        if min_query_len < 0:
            raise ValueError("TRTLLM ragged prefill query lengths must be non-negative")
        self._has_active_rows = min_query_len > 0
        if not self._has_active_rows:
            has_mixed_rows = bool(torch.any(query_lens_cpu > 0).item())
            if has_mixed_rows:
                raise ValueError(
                    "TRTLLM ragged prefill contains mixed active and empty query "
                    f"rows: query_lens={query_lens_cpu.tolist()}"
                )

    def supports_out(self) -> bool:
        # Output head dim is v.shape[-1] == v_head_dim, so `out` is unpadded.
        return True

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        if out is None:
            out = torch.empty(
                q.shape[0],
                q.shape[1],
                v.shape[2],
                device=q.device,
                dtype=self._prefill_metadata.output_dtype,
            )

        if not self._has_active_rows:
            out.zero_()
            if return_softmax_lse:
                lse = torch.empty(
                    self.num_heads,
                    q.shape[0],
                    dtype=torch.float32,
                    device=q.device,
                )
                return out, lse.fill_(-float("inf"))
            return out

        ret = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self._workspace_buffer,
            seq_lens=self._query_seq_lens,
            max_q_len=self._prefill_metadata.max_query_len,
            max_kv_len=self._prefill_metadata.max_query_len,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=self._query_seq_lens.shape[0],
            window_left=-1,
            cum_seq_lens_q=self._prefill_metadata.query_start_loc,
            cum_seq_lens_kv=self._prefill_metadata.query_start_loc,
            enable_pdl=False,
            is_causal=True,
            return_lse=return_softmax_lse,
            out=out,
            skip_all_rows_active_check=True,
        )

        if isinstance(ret, tuple):
            # Convert from (q_len, num_heads) to (num_heads, q_len)
            return ret[0], log2_lse_to_ln(ret[1].transpose(0, 1))
        return ret

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        if not chunk.all_rows_active:
            raise ValueError(
                "TRTLLM ragged context prefill contains an empty query or KV row"
            )

        if out is None:
            out = torch.empty(
                q.shape[0],
                q.shape[1],
                v.shape[2],
                device=q.device,
                dtype=self._prefill_metadata.output_dtype,
            )

        splits = _k3_split_kv_factor(
            chunk.max_query_len, chunk.num_requests, chunk.max_seq_len
        )
        if splits > 1 and int(chunk.seq_lens.min().item()) < splits:
            # Empty segments leave output rows unwritten; NaN * zero is NaN.
            splits = 1
        if splits > 1:
            return self._k3_split_context(chunk, q, k, v, out, splits)

        backend = _k3_context_backend(q, k, v, out)
        query_start_loc, cu_seq_lens = chunk.query_start_loc, chunk.cu_seq_lens
        if backend == "cute-dsl":
            query_start_loc = _k3_cute_indptr(query_start_loc)
            cu_seq_lens = _k3_cute_indptr(cu_seq_lens)
        attn_out, lse = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self._workspace_buffer,
            seq_lens=chunk.seq_lens,
            max_q_len=chunk.max_query_len,
            max_kv_len=chunk.max_seq_len,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=chunk.num_requests,
            window_left=-1,
            cum_seq_lens_q=query_start_loc,
            cum_seq_lens_kv=cu_seq_lens,
            enable_pdl=False,
            is_causal=False,
            return_lse=True,
            out=out,
            skip_all_rows_active_check=True,
            backend=backend,
        )

        if _K3_CUTE_CONTEXT_DIAG:
            key = f"{backend}:q{chunk.max_query_len}:kv{chunk.max_seq_len}:b{chunk.num_requests}"
            # Diagnostic host counts are reset after health, read after drain.
            # Prefill executes eagerly; decode CUDA graph paths are unchanged.
            _k3_cute_context_counts[key] = _k3_cute_context_counts.get(key, 0) + 1

        # Convert from (q_len, num_heads) to (num_heads, q_len)
        return attn_out, log2_lse_to_ln(lse.transpose(0, 1))


    def _k3_split_context(self, chunk, q, k, v, out, splits):
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        num_tokens = q.shape[0]
        memo = chunk.__dict__.setdefault("_k3_split_memo", {}) if _K3_SPLIT_MEMO else None
        idx = memo.get((splits, num_tokens)) if memo is not None else None
        if idx is None:
            idx = _k3_split_kv_indices(
                chunk.query_start_loc, chunk.cu_seq_lens, num_tokens, splits
            )
            if memo is not None:
                memo[(splits, num_tokens)] = idx
        src, part_rows, _vq, vkv_len, vqsl, vkv = idx
        q_rep = q.index_select(0, src)
        o_rep = torch.empty(
            q_rep.shape[0], q.shape[1], v.shape[2],
            device=q.device, dtype=self._prefill_metadata.output_dtype,
        )
        o_rep, lse_rep = trtllm_ragged_attention_deepseek(
            query=q_rep,
            key=k,
            value=v,
            workspace_buffer=self._workspace_buffer,
            seq_lens=vkv_len,
            max_q_len=chunk.max_query_len,
            max_kv_len=-(-chunk.max_seq_len // splits),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=chunk.num_requests * splits,
            window_left=-1,
            cum_seq_lens_q=vqsl,
            cum_seq_lens_kv=vkv,
            enable_pdl=False,
            is_causal=False,
            return_lse=True,
            out=o_rep,
            skip_all_rows_active_check=True,
        )
        if _K3_SPLIT_MERGE_FUSED and o_rep.stride(-1) == 1:
            if out is None:
                out = torch.empty(
                    num_tokens, q.shape[1], v.shape[2],
                    device=q.device, dtype=self._prefill_metadata.output_dtype,
                )
            num_heads, head_dim = o_rep.shape[1], o_rep.shape[2]
            lse_out = torch.empty(num_heads, num_tokens, dtype=torch.float32,
                                  device=q.device)
            rows = part_rows if part_rows.is_contiguous() else part_rows.contiguous()
            block_h = min(triton.next_power_of_2(num_heads), 16)
            _k3_split_merge_kernel[(num_tokens, triton.cdiv(num_heads, block_h))](
                o_rep, lse_rep, rows, out, lse_out,
                num_tokens, num_heads,
                o_rep.stride(0), o_rep.stride(1), lse_rep.stride(0), lse_rep.stride(1),
                out.stride(0), out.stride(1),
                S=splits, D=head_dim, BLOCK_H=block_h,
            )
            return out, lse_out
        # LSE merge (base-2 LSE from the kernel) over the S partials.
        lse_s = lse_rep.float()[part_rows]  # [S, Tq, H]
        m = lse_s.amax(0)
        w = torch.exp2(lse_s - m)  # [S, Tq, H]
        denom = w.sum(0)
        o_s = o_rep[part_rows].float()  # [S, Tq, H, D]
        merged = (o_s * w[..., None]).sum(0) / denom[..., None]
        if out is None:
            out = torch.empty(
                num_tokens, q.shape[1], v.shape[2],
                device=q.device, dtype=self._prefill_metadata.output_dtype,
            )
        out.copy_(merged)
        lse = m + torch.log2(denom)
        return out, log2_lse_to_ln(lse.transpose(0, 1))
