# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
CuTe DSL MLA Decode Kernel Integration
=======================================

Wraps NVIDIA's CuTe DSL MLA decode kernels (FP16/BF16/FP8) for Blackwell SM100
and exposes them via a PyTorch API compatible with FlashInfer's MLA backend.
"""

import functools
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
)
from tokenspeed_mla.mla_decode_fp16 import (
    BlackwellMultiHeadLatentAttentionForwardFP16,
)
from tokenspeed_mla.mla_helpers import (
    get_mla_decode_fold_sq_factor,
    select_mla_decode_tilers,
)

import os as _os
_K3_VAR_SPLIT = _os.environ.get('VLLM_K3_TS_VAR_SPLIT', '0') == '1'
_K3_MAX_SPLITS = int(_os.environ.get('VLLM_K3_TS_MAX_SPLITS', '32'))
_K3_VAR_SPLIT_Q = _os.environ.get('VLLM_K3_TS_VAR_SPLIT_Q', '0') == '1'
_K3_VS_ALPHA = float(_os.environ.get('VLLM_K3_TS_VS_ALPHA', '1.0'))
_K3_VS_CEIL = _os.environ.get('VLLM_K3_TS_VS_CEIL', '1') == '1'
_K3_SKIP_CORR = float(_os.environ.get('VLLM_K3_TS_SKIP_CORR', '0.0') or 0.0)
# Static work order for the var-split persistent scheduler (needs the K3
# tokenspeed_mla overlay whose scheduler reads block_split_kvs[B:]).
_K3_WORK_PERM = _os.environ.get('VLLM_K3_TS_WORK_PERM', '0') == '1'
# 'snake': longest-first snake over clusters; 'lpt': batched longest-processing-
# time (least-loaded cluster first, Triton, falls back to snake if it cannot
# place every item); 'auto': lpt from 56 requests of 96 heads x 5 tokens (B=64:
# 1.11-1.15x over snake; at B<=48 it only adds ~45 us of ordering per step).
_K3_WORK_ORDER = _os.environ.get('VLLM_K3_TS_WORK_ORDER', 'auto')
_K3_LPT_MIN_ROWS = int(_os.environ.get('VLLM_K3_TS_LPT_MIN_ROWS', str(56 * 5 * 96)))
# One Triton launch for split counts + work order (instead of ~15 torch ops).
_K3_FUSED_ORDER = _os.environ.get('VLLM_K3_TS_FUSED_ORDER', '1') == '1'
# Single-split requests write O/LSE from the split kernel (needs the overlay kernel).
_K3_DIRECT_SINGLE = _os.environ.get('VLLM_K3_TS_DIRECT_SINGLE', '0') == '1'
_K3_ORDER_MAX_ITEMS = 8192
from tokenspeed_mla.utils import (
    get_max_active_clusters,
    get_num_sm,
    torch_to_cutlass_dtype,
)


import triton
import triton.language as tl


@triton.jit
def _k3_lpt_kernel(
    cost_ptr,  # fp32 [N] item costs in descending order (0 = empty item)
    order_ptr,  # int32 [N] item ids in the same order
    nne_ptr,  # int32 [1] number of non-empty items
    perm_ptr,  # int32 [N] out: slot -> item
    ok_ptr,  # int32 [1] out: 1 if every item was placed
    N,
    n_cl,
    K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    MAX_CAP: tl.constexpr,
):
    cl = tl.arange(0, BLOCK_C)
    in_grid = cl < n_cl
    # cluster cl owns slots cl, cl + n_cl, ... < N
    cap = tl.where(in_grid, N // n_cl + (cl < N % n_cl).to(tl.int32), 0)
    loads = tl.zeros([BLOCK_C], dtype=tl.float32)
    cnt = tl.zeros([BLOCK_C], dtype=tl.int32)
    kk = tl.arange(0, K)
    n_ne = tl.load(nne_ptr)
    for r in range(0, (n_ne + K - 1) // K):
        idx = r * K + kk
        valid = idx < n_ne
        c = tl.load(cost_ptr + idx, mask=valid, other=0.0)
        it = tl.load(order_ptr + idx, mask=valid, other=0)
        open_ = cnt < cap
        eff = tl.where(open_, loads, float("inf"))
        less = (eff[None, :] < eff[:, None]) | (
            (eff[None, :] == eff[:, None]) & (cl[None, :] < cl[:, None])
        )
        rank = tl.sum(less.to(tl.int32), axis=1)
        # the j-th least-loaded open cluster takes the j-th largest item
        sel = (rank[:, None] == kk[None, :]) & valid[None, :] & open_[:, None]
        got = tl.sum(sel.to(tl.int32), axis=1) > 0
        item = tl.sum(tl.where(sel, it[None, :], 0), axis=1)
        tl.store(perm_ptr + cl + n_cl * cnt, item, mask=got)
        loads += tl.sum(tl.where(sel, c[None, :], 0.0), axis=1)
        cnt += got.to(tl.int32)
    placed = tl.sum(cnt, axis=0)
    tl.store(ok_ptr, (placed == n_ne).to(tl.int32))
    # empty items fill the remaining slots of each cluster
    free = cap - cnt
    off = tl.cumsum(free, 0) - free
    for t in range(0, MAX_CAP):
        e = n_ne + off + t
        m = (t < free) & (e < N)
        item = tl.load(order_ptr + e, mask=m, other=0)
        tl.store(perm_ptr + cl + n_cl * (cnt + t), item, mask=m)


@triton.jit
def _k3_order_kernel(
    seqs_ptr,  # int32 [B] per-request cache lengths
    out_ptr,  # int32 [B + N] out: split counts, then slot -> item
    order_ptr,  # int32 [N] out (LPT only): items by descending cost
    cost_ptr,  # fp32 [N] out (LPT only): their costs
    nne_ptr,  # int32 [1] out (LPT only): non-empty items
    B,
    S,
    s_cap,
    n_cl,
    split_num,  # q_len > 1: round(n_sm * alpha * 1024); q_len == 1: n_sm
    split_den,  # q_len > 1: q_len_eff * 1024; q_len == 1: 1
    k_tile,
    CEIL: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WRITE_SORTED: tl.constexpr,
):
    # Same integer math as the torch path: split = clamp(ceil(len * num / (sum * den)), 1, s_cap)
    ob = tl.arange(0, BLOCK_B)
    lens_b = tl.load(seqs_ptr + ob, mask=ob < B, other=0).to(tl.int64)
    tot = tl.maximum(tl.sum(lens_b, axis=0), 1)
    den = tot * split_den
    nb = lens_b * split_num
    sb = (nb + den - 1) // den if CEIL else nb // den
    sb = tl.minimum(tl.maximum(sb, 1), s_cap)
    tl.store(out_ptr + ob, sb.to(tl.int32), mask=ob < B)
    # work items in cluster-batch order: index = (split * B + b) * S + s
    N = s_cap * B * S
    i = tl.arange(0, BLOCK_N)
    valid = i < N
    b = (i // S) % B
    sp = (i // (S * B)).to(tl.int64)
    li = tl.load(seqs_ptr + b, mask=valid, other=0).to(tl.int64)
    ni = li * split_num
    si = (ni + den - 1) // den if CEIL else ni // den
    si = tl.minimum(tl.maximum(si, 1), s_cap)
    kt = (li + k_tile - 1) // k_tile
    per = (kt + si - 1) // si
    cnt = tl.maximum(tl.minimum(kt, (sp + 1) * per) - sp * per, 0)
    cnt = tl.where(sp < si, cnt, 0)
    cost = tl.where(cnt > 0, cnt * 8 + 24, 0)
    # stable descending sort by cost: key = cost << 14 | (16383 - i)
    key = tl.where(valid, (tl.minimum(cost, 131071) << 14) | (16383 - i), -1).to(tl.int32)
    key = tl.sort(key, descending=True)
    item = 16383 - (key & 16383)
    r = i
    j = r // n_cl
    p = r % n_cl
    rev = (j % 2 == 1) & (j < N // n_cl)
    slot = j * n_cl + tl.where(rev, n_cl - 1 - p, p)
    tl.store(out_ptr + B + slot, item, mask=valid)
    if WRITE_SORTED:
        tl.store(order_ptr + r, item, mask=valid)
        tl.store(cost_ptr + r, (key >> 14).to(tl.float32), mask=valid)
        tl.store(nne_ptr, tl.sum((cnt > 0).to(tl.int32), axis=0))


def _k3_split_and_order(
    seqs: torch.Tensor,
    B: int,
    S: int,
    s_cap: int,
    n_cl: int,
    k_tile: int,
    split_num: int,
    split_den: int,
    use_lpt: bool,
) -> torch.Tensor:
    """[B split counts | slot -> item permutation] in one launch (two with LPT)."""
    N = s_cap * B * S
    n_cl = max(1, min(n_cl, N))
    dev = seqs.device
    out = torch.empty(B + N, dtype=torch.int32, device=dev)
    lpt = use_lpt and n_cl >= 2
    if lpt:
        order = torch.empty(N, dtype=torch.int32, device=dev)
        cost = torch.empty(N, dtype=torch.float32, device=dev)
        nne = torch.empty(1, dtype=torch.int32, device=dev)
    else:
        order = cost = nne = out
    _k3_order_kernel[(1,)](
        seqs, out, order, cost, nne, B, S, s_cap, n_cl, split_num, split_den, k_tile,
        CEIL=_K3_VS_CEIL, BLOCK_B=triton.next_power_of_2(B),
        BLOCK_N=triton.next_power_of_2(N), WRITE_SORTED=lpt, num_warps=8,
    )
    if lpt:
        perm = torch.empty(N, dtype=torch.int32, device=dev)
        ok = torch.empty(1, dtype=torch.int32, device=dev)
        _k3_lpt_kernel[(1,)](
            cost, order, nne, perm, ok, N, n_cl,
            K=16, BLOCK_C=triton.next_power_of_2(n_cl), MAX_CAP=N // n_cl + 1,
            num_warps=4,
        )
        tail = out[B:]
        tail.copy_(torch.where(ok.bool(), perm, tail))
    return out


@functools.cache
def _k3_num_clusters(cluster_size: int) -> int:
    # Persistent grid: cluster c takes cluster-batch slots c, c + n, ...
    return get_max_active_clusters(cluster_size)


def _k3_work_perm(
    split: torch.Tensor,
    seqs: torch.Tensor,
    B: int,
    S: int,
    s_cap: int,
    n_cl: int,
    k_tile: int,
    use_lpt: bool = False,
) -> torch.Tensor:
    """Longest-first snake order of the (split, b, s) work items over the
    persistent clusters. Returns int32 [s_cap * B * S], a permutation of the
    cluster-batch indices; device-only ops, so it is graph-capturable."""
    dev = split.device
    kt = (seqs.to(torch.int64) + k_tile - 1) // k_tile
    n = split.to(torch.int64)
    per = (kt + n - 1) // n
    sp = torch.arange(s_cap, device=dev, dtype=torch.int64).view(-1, 1)
    cnt = (torch.minimum(kt, (sp + 1) * per) - sp * per).clamp_(min=0)
    cnt = torch.where(sp < n, cnt, 0)
    # cluster-batch index = (split * B + b) * S + s, s fastest
    cnt = cnt.unsqueeze(-1).expand(s_cap, B, S).reshape(-1)
    # fixed per-item cost ~3 k-tiles (pipeline fill + epilogue); empty items 0
    cost = torch.where(cnt > 0, cnt * 8 + 24, 0)
    order = torch.argsort(cost, descending=True, stable=True)
    N = cnt.numel()
    n_cl = max(1, min(n_cl, N))
    r = torch.arange(N, device=dev, dtype=torch.int64)
    j, p = r // n_cl, r % n_cl
    rev = (j % 2 == 1) & (j < N // n_cl)
    slot = j * n_cl + torch.where(rev, n_cl - 1 - p, p)
    perm = torch.empty(N, dtype=torch.int32, device=dev)
    perm[slot] = order.to(torch.int32)
    if not use_lpt or n_cl < 2:
        return perm
    order32 = order.to(torch.int32)
    cost_sorted = cost[order].to(torch.float32)
    n_ne = (cnt > 0).sum().to(torch.int32).view(1)
    lpt = torch.empty(N, dtype=torch.int32, device=dev)
    ok = torch.empty(1, dtype=torch.int32, device=dev)
    _k3_lpt_kernel[(1,)](
        cost_sorted, order32, n_ne, lpt, ok, N, n_cl,
        K=16, BLOCK_C=triton.next_power_of_2(n_cl), MAX_CAP=N // n_cl + 1,
        num_warps=4,
    )
    return torch.where(ok.bool(), lpt, perm)


@functools.cache
def _get_split_kv_and_workspace_size(
    B: int,
    q_len: int,
    H: int,
    kv_lora_rank: int,
    max_active_blocks: int,
    max_seq_len: int,
    torch_dtype: torch.dtype,
    mma_qk_tiler_mn: tuple[int, int],
) -> Tuple[int, int]:
    """Cache split_kv and workspace_size since they are deterministic for the same params."""
    is_fp8 = torch_dtype == torch.float8_e4m3fn
    if is_fp8 and mma_qk_tiler_mn[0] == 64:
        # M64 launches one CTA per M tile. Reuse the FP8 kernel's wave-aware
        # split heuristic so serving matches the standalone mla_decode_fp8.py
        # benchmark path instead of under-splitting like the 2-CTA M128 path.
        split_kv = BlackwellMultiHeadLatentAttentionForwardFP8.get_split_kv(
            B,
            q_len,
            max_seq_len,
            mma_qk_tiler_mn,
            max_active_blocks,
            1,
        )
        workspace_size = BlackwellMultiHeadLatentAttentionForwardFP8.get_workspace_size(
            H, q_len, kv_lora_rank, B, split_kv, cutlass.Float32
        )
    else:
        split_kv = BlackwellMultiHeadLatentAttentionForwardFP16.get_split_kv_simplified(
            B, q_len, max_active_blocks
        )
        workspace_size = (
            BlackwellMultiHeadLatentAttentionForwardFP16.get_workspace_size(
                H, q_len, kv_lora_rank, B, split_kv, cutlass.Float32
            )
        )
    return split_kv, workspace_size


@functools.cache
def _check_can_implement(
    torch_dtype: torch.dtype,
    page_size: int,
    num_heads: int,
    seq_len_q: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
    compute_capability: tuple[int, int],
) -> None:
    """Check if the kernel supports the given configuration (cached)."""
    is_fp8 = torch_dtype == torch.float8_e4m3fn
    mma_qk_tiler_mn, mma_pv_tiler_mn = select_mla_decode_tilers(
        num_heads,
        seq_len_q,
        is_fp8=is_fp8,
        compute_capability=compute_capability,
    )
    KernelClass = (
        BlackwellMultiHeadLatentAttentionForwardFP8
        if is_fp8
        else BlackwellMultiHeadLatentAttentionForwardFP16
    )
    cutlass_dtype = torch_to_cutlass_dtype(torch_dtype)
    if not KernelClass.can_implement(
        1,  # B (runtime, use placeholder)
        seq_len_q,
        1,  # K (runtime, use placeholder)
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        cutlass_dtype,
        cutlass_dtype,
        cutlass.Float32,
        cutlass.Float32,
        mma_qk_tiler_mn,
        mma_pv_tiler_mn,
        1,  # split_kv placeholder; actual value is selected at runtime
        is_persistent,
        is_var_seq,
        is_var_split_kv,
        page_size,
    ):
        raise ValueError(
            f"tokenspeed_mla_decode: unsupported configuration "
            f"(q_len={seq_len_q}, num_heads={num_heads}, page_size={page_size}, "
            f"dtype={torch_dtype})"
        )


@functools.cache
def _get_compiled_mla_kernel(
    torch_dtype: torch.dtype,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
    skip_correction_threshold: float = 0.0,
    is_workspace_size_zero: bool = False,
    fold_sq_factor: int = 1,
    causal_mask: bool = True,
    num_heads: int = 128,
    seq_len_q: int = 1,
    cp_world: int = 1,  # DCP world size; >1 enables strided global-coord causal masking
    use_pdl: bool = False,
    return_lse: bool = False,  # DCP: enable LSE output
    compute_capability: tuple[int, int] = (0, 0),
    use_work_perm: bool = False,
    direct_single: bool = False,
) -> Callable:
    """Compile and cache an MLA decode kernel.

    Returns a callable that accepts (q_latent, q_rope, c_latent, c_rope,
    page_table, o, lse (None to skip), workspace, split_kv_scalar, cache_seqs,
    block_split_kvs, softmax_scale_scalar, output_scale_scalar).

    All scalar arguments must be pre-wrapped as Int32/Float32.
    """
    is_fp8 = torch_dtype == torch.float8_e4m3fn
    mma_qk_tiler_mn, mma_pv_tiler_mn = select_mla_decode_tilers(
        num_heads,
        seq_len_q,
        is_fp8=is_fp8,
        compute_capability=compute_capability,
    )
    # 2 CTAs for M=128 path; 1 CTA for M=64 path.
    cluster_shape_mnk = (2, 1, 1) if mma_qk_tiler_mn[0] == 128 else (1, 1, 1)
    KernelClass = (
        BlackwellMultiHeadLatentAttentionForwardFP8
        if is_fp8
        else BlackwellMultiHeadLatentAttentionForwardFP16
    )
    cutlass_dtype = torch_to_cutlass_dtype(torch_dtype)
    cutlass_out_dtype = cutlass.BFloat16 if is_fp8 else cutlass_dtype

    kernel_kwargs = dict(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=mma_qk_tiler_mn,
        mma_pv_tiler_mn=mma_pv_tiler_mn,
        max_active_clusters=get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1]
        ),
        page_size=page_size,
        skip_correction_threshold=skip_correction_threshold,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        fold_sq_factor=fold_sq_factor,
    )
    # Causal masking is now supported by both the FP8 and FP16/BF16 kernels.
    kernel_kwargs["is_causal"] = causal_mask
    kernel_kwargs["num_heads"] = num_heads
    kernel_kwargs["seq_len_q"] = seq_len_q
    if is_fp8:
        # DCP (cp_world) strided global-coordinate causal masking is fp8-only.
        kernel_kwargs["cp_world"] = cp_world
    if use_work_perm:
        kernel_kwargs["use_work_perm"] = True
    if direct_single:
        kernel_kwargs["k3_direct_single"] = True
    kernel_obj = KernelClass(**kernel_kwargs)

    # All dimensions as sym_int — this matches the original kernel's use of
    # mark_compact_shape_dynamic, which makes ALL shapes dynamic CuTe Integers.
    # Static Python ints would cause cute.assume() to fail with AttributeError
    # inside initialize_workspace() since it expects DSL Integer types.
    sym_heads = cute.sym_int()
    sym_latent = cute.sym_int(divisibility=16)
    sym_seq_q = cute.sym_int()
    sym_rope = cute.sym_int(divisibility=16)
    sym_batch = cute.sym_int()  # query/output batch dimension
    sym_kv_batch = cute.sym_int()  # KV cache batch dim (flat pool, =1 in paged mode)
    sym_seq_kv = cute.sym_int()
    sym_page_count = cute.sym_int()
    sym_workspace_size = cute.sym_int()

    # q_latent, q_rope, c_latent, c_rope are slices of contiguous tensors on
    # the last dim (e.g. query[..., :kv_lora_rank]), so they are NOT contiguous:
    #   stride[-2] = D_qk (original full last dim), not the sliced shape.
    # Use make_fake_tensor with fully dynamic strides so the compiled kernel
    # reads actual strides from the runtime tensor.  Last-dim stride is always 1.

    # q_latent: [batch_size, seq_len_q, num_heads, latent_dim] — non-contiguous slice
    q_latent_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_latent),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # q_rope: [batch_size, seq_len_q, num_heads, rope_dim] — non-contiguous slice
    q_rope_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_rope),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # c_latent: [kv_batch, seq_len_k, latent_dim] — non-contiguous slice
    # kv_batch is a separate sym_int from query batch: paged KV cache uses a flat
    # pool so kv_batch=num_pages at runtime, while query batch can be any value.
    c_latent_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_latent),
        stride=(cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # c_rope: [kv_batch, seq_len_k, rope_dim] — non-contiguous slice
    c_rope_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_rope),
        stride=(cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # page_table: [batch_size, page_count] — contiguous
    page_table_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch, sym_page_count),
        stride_order=(1, 0),
        assumed_align=4,
    )
    # o: [batch_size, seq_len_q, num_heads, latent_dim] — contiguous
    o_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_out_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_latent),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    if is_workspace_size_zero:
        workspace_fake = None
    else:
        # workspace: 1-D int8 buffer. 32-byte alignment because workspace stores
        # fp32 partial sums internally, requiring stricter alignment than tensors.
        workspace_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8,
            (sym_workspace_size,),
            assumed_align=32,
        )
    # cache_seqs: [batch_size] — int32
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=4,
    )
    # DCP: per-request causal bound tensor [batch] int32
    causal_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=4,
    )
    # block_split_kvs: [batch_size] — int32 (only needed for is_var_split_kv=True)
    if is_var_split_kv:
        block_split_kvs_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            # work order: [B split counts | cluster-batch permutation]
            (cute.sym_int() if use_work_perm else sym_batch,),
            assumed_align=4,
        )
    else:
        block_split_kvs_fake = None

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # DCP: build an lse_fake when return_lse=True so the compiled kernel
    # is shaped to emit LSE. Shape per mla_decode_fp8.py:359 = [B, q_len, H] fp32.
    if return_lse:
        lse_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (sym_batch, sym_seq_q, sym_heads),
            stride_order=(2, 1, 0),
            assumed_align=4,
        )
    else:
        lse_fake = None

    compile_args = [
        kernel_obj,
        q_latent_fake,
        q_rope_fake,
        c_latent_fake,
        c_rope_fake,
        page_table_fake,
        o_fake,
        lse_fake,
        workspace_fake,
        Int32(1),  # split_kv placeholder
        cache_seqs_fake,
    ]
    # DCP causal bound is an fp8-decode-kernel-only argument (see the call site).
    if is_fp8:
        compile_args.append(causal_seqs_fake)
    compile_args += [
        block_split_kvs_fake,
        Float32(1.0),  # softmax_scale placeholder
        Float32(1.0),  # output_scale placeholder
        stream_fake,
        use_pdl,
    ]
    compiled_kernel = cute.compile(
        *compile_args, options="--enable-tvm-ffi --opt-level 2"
    )

    return compiled_kernel


def tokenspeed_mla_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    output_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    is_var_seq: bool = True,
    causal_mask: bool = True,
    enable_pdl: bool = False,
    return_lse: bool = False,  # also return log-sum-exp (DCP cross-rank merge)
    causal_seqs: Optional[
        torch.Tensor
    ] = None,  # per-request global causal bound; None = local (non-DCP)
    cp_world: int = 1,  # decode-context-parallel world size (1 = no DCP)
    cp_rank: int = 0,  # this rank's index in [0, cp_world); subtracted from causal_seqs
    k3_memo: Optional[dict] = None,  # K3: per-step split memo (VLLM_K3_TS_VS_MEMO)
) -> torch.Tensor:
    """CuTe DSL MLA decode kernel for Blackwell SM100.

    Parameters
    ----------
    query : torch.Tensor
        [B, q_len, H, D_qk] where D_qk = kv_lora_rank + qk_rope_head_dim
    kv_cache : torch.Tensor
        [num_pages, page_size, D_ckv + D_kpe] (3D) or [num_pages, 1, page_size, D_ckv + D_kpe] (4D)
    workspace_buffer : torch.Tensor
        Pre-allocated workspace buffer (uint8). Required size depends on batch size
        and split_kv (auto-computed from B, q_len, and number of SMs):

        - Formula: ``B * H * q_len * split_kv * (kv_lora_rank + 1) * 4`` bytes
          (0 when split_kv == 1, which happens when B >= num_SMs / 2)
        - The TokenSpeed runtime backend grows this buffer from the actual
          q_len before each decode launch.
    kv_lora_rank : int
        Latent dimension (e.g. 512).
    qk_rope_head_dim : int
        RoPE dimension (e.g. 64).
    block_tables : torch.Tensor
        [B, max_pages] — page table indices.
    seq_lens : torch.Tensor
        [B] — per-request KV sequence lengths.
    max_seq_len : int
        Maximum sequence length across the batch.
    softmax_scale : float
        Scale factor for QK^T before softmax.
    output_scale : float
        Scale factor applied to the output.
    out : Optional[torch.Tensor]
        Pre-allocated output tensor [B, q_len, H, kv_lora_rank].
    is_var_seq : bool
        Whether the sequence length is variable.
        If True, the sequence length is variable.
        Otherwise,the sequence length is fixed for all the requests in the batch.
    causal_mask : bool
        Whether to enable causal masking in the CuTe DSL kernel.
        Currently this is effective for the FP8 kernel path.
    enable_pdl : bool
        When True, enables Programmatic Dependent Launch (PDL) on the
        underlying CuTe DSL decode kernel. Tokenspeed callers wire this from
        ``pdl_enabled()`` so ``--disable-pdl`` propagates through to the
        kernel binary; ``use_pdl`` is part of the kernel cache key.
    return_lse : bool
        Decode-context-parallel (DCP) support. When True, also return the
        per-(B, q_len, head) log-sum-exp so callers can merge the partial
        attention each rank computes over its KV slice into the full result.
        Default False keeps the original single-tensor return and emits no
        LSE code in the compiled kernel.
    causal_seqs : Optional[torch.Tensor]
        DCP support. Per-request *global* causal bound [B] (int32) -- the full
        context length, the SAME value on every rank. Under DCP each rank holds a
        strided 1/cp_world slice of the context, so the causal cutoff is expressed
        in global coordinates; this function subtracts ``cp_rank`` and the kernel
        divides by ``cp_world`` to recover this rank's local bound. Required when
        ``cp_world > 1``. ``None`` (default, non-DCP) uses the local ``seq_lens``.
    cp_world : int
        DCP world size. ``1`` (default) is the non-DCP path and compiles to the
        original masking with no extra work. ``>1`` enables strided
        global-coordinate causal masking (``causal_seqs`` is then required).
    cp_rank : int
        This rank's index in ``[0, cp_world)``. Local key ``c`` on this rank maps
        to global position ``c*cp_world + cp_rank``, so the rank-local causal
        bound is ``ceil((causal_seqs - cp_rank) / cp_world)``; the wrapper folds
        ``cp_rank`` in for you, so callers pass the same global ``causal_seqs`` on
        every rank. Must be 0 when ``cp_world == 1``.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Output tensor [B, q_len, H, kv_lora_rank]. When ``return_lse=True``,
        returns ``(output, lse)`` with ``lse`` of shape [B, q_len, H] (fp32).
    """
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    assert (
        query.dtype in supported_dtypes
    ), f"tokenspeed_mla_decode only supports {supported_dtypes}, got {query.dtype}"
    assert (
        kv_cache.dtype == query.dtype
    ), f"kv_cache dtype {kv_cache.dtype} must match query dtype {query.dtype}"
    B, q_len, H, D_qk = query.shape
    assert D_qk == kv_lora_rank + qk_rope_head_dim

    q_dtype = query.dtype

    # Handle 3D vs 4D kv_cache: normalize to 3D [num_pages, page_size, D_total]
    if kv_cache.dim() == 4:
        kv_cache = kv_cache.squeeze(1)
    page_size = kv_cache.shape[1]

    # Split query into latent and rope components — keep contiguous [B, q_len, H, D].
    # The kernel's __call__ reinterprets to [H, D, q_len, B] via zero-cost make_tensor.
    q_latent_k = query[..., :kv_lora_rank]
    q_rope_k = query[..., kv_lora_rank:]

    # KV cache slices — keep contiguous [num_pages, page_size, D].
    # The kernel reinterprets to [page_size, D, num_pages] internally.
    c_latent_k = kv_cache[:, :, :kv_lora_rank]
    c_rope_k = kv_cache[:, :, kv_lora_rank:]

    # Page table: [B, max_pages]: passed directly, kernel reinterprets.
    page_table_k = block_tables

    # Runtime validation (int comparisons only, negligible overhead)
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
    is_fp8 = q_dtype == torch.float8_e4m3fn
    compute_capability = torch.cuda.get_device_capability(query.device)
    mma_qk_tiler_mn, _ = select_mla_decode_tilers(
        H,
        q_len,
        is_fp8=is_fp8,
        compute_capability=compute_capability,
    )
    # Fold only by a factor that exactly divides q_len; otherwise leave q_len
    # on the scheduler dimension.
    mma_m_tile = mma_qk_tiler_mn[0]
    fold_sq_factor = get_mla_decode_fold_sq_factor(H, q_len, mma_m_tile)

    # Effective dimensions used by split_kv/workspace accounting.
    H_eff = H * fold_sq_factor
    q_len_eff = q_len // fold_sq_factor

    # Cached split_kv and workspace_size computation
    max_active_blocks = get_num_sm(query.device)
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        B,
        q_len_eff,
        H_eff,
        kv_lora_rank,
        max_active_blocks,
        max_seq_len,
        q_dtype,
        mma_qk_tiler_mn,
    )

    # Prepare workspace: slice of contiguous 1D buffer is already contiguous
    assert (
        workspace_buffer.dtype == torch.int8
    ), f"workspace_buffer must be torch.int8, got {workspace_buffer.dtype}"
    assert workspace_buffer.numel() >= workspace_size, (
        f"workspace_buffer too small: {workspace_buffer.numel()} bytes, "
        f"need {workspace_size} bytes"
    )
    is_workspace_size_zero = workspace_size == 0
    if is_workspace_size_zero:
        workspace_bytes = None
    else:
        workspace_bytes = workspace_buffer[:workspace_size]
    # Output buffer: contiguous [B, q_len, H, D].
    # Kernel reinterprets to [H, D, q_len, B] internally via zero-cost make_tensor.
    # FP8 kernel writes BF16 output for better downstream precision.
    out_dtype = torch.bfloat16 if q_dtype == torch.float8_e4m3fn else q_dtype
    if out is not None:
        o_k = out
    else:
        o_k = torch.empty(
            (B, q_len, H, kv_lora_rank), dtype=out_dtype, device=query.device
        )

    # cache_seqs: per-batch sequence lengths (skip .to() if already int32)
    cache_seqs = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
    # cp_world is the DCP world size; reject non-positive values up front. Otherwise
    # cp_world=0 slips past the cp_rank check below (max(cp_world, 1) == 1) and reaches
    # the kernel, which treats cp_world != 1 as DCP and divides the causal bound by
    # self.cp_world -> div-by-zero / nonsensical local bounds deep in JIT.
    if cp_world < 1:
        raise ValueError(f"cp_world must be >= 1, got cp_world={cp_world}")
    # DCP (strided global-coordinate causal masking) is implemented on the fp8
    # decode kernel only; the fp16 kernel masks against the local cache length.
    is_fp8 = q_dtype == torch.float8_e4m3fn
    if not is_fp8 and (cp_world > 1 or causal_seqs is not None):
        raise ValueError(
            "decode-context-parallel (cp_world > 1 / causal_seqs) is only supported "
            f"on the fp8 decode path, got query dtype {q_dtype}"
        )
    # causal_seqs is mandatory for DCP: the kernel divides the causal bound by
    # cp_world, so falling back to the rank-local cache_seqs here would mask each
    # rank to ~1/cp_world of its slice and produce a wrong partial. Require the
    # caller to pass the global bound explicitly rather than fail silently.
    if cp_world > 1 and causal_seqs is None:
        raise ValueError(
            "causal_seqs (per-request global causal bound) is required when "
            f"cp_world > 1, got cp_world={cp_world} and causal_seqs=None"
        )
    if not 0 <= cp_rank < max(cp_world, 1):
        raise ValueError(
            f"cp_rank must be in [0, cp_world), got cp_rank={cp_rank} "
            f"cp_world={cp_world}"
        )
    # DCP: derive this rank's local causal bound from the global one. causal_seqs
    # is the global cutoff (same on every rank); local key c maps to global
    # position c*cp_world + cp_rank, so the kernel computes
    # k_bound = ceil((causal_seqs - cp_rank - (q_len-1) + q_tok) / cp_world). We
    # fold cp_rank in here so callers pass the same global bound on every rank.
    # None (non-DCP) => cache_seqs (local_K, cp_world==1 => bound unchanged).
    if causal_seqs is None:
        causal_seqs_dev = cache_seqs
    else:
        causal_seqs_dev = (
            causal_seqs
            if causal_seqs.dtype == torch.int32
            else causal_seqs.to(torch.int32)
        )
        if cp_rank:
            causal_seqs_dev = causal_seqs_dev - cp_rank

    is_var_split_kv = False
    block_split_kvs = None
    use_work_perm = False
    skip_correction_threshold = _K3_SKIP_CORR

    # For fixed-length input, set is_persistent to True; otherwise, set to False.
    is_persistent = not is_var_seq

    # K3: length-balanced variable split-KV (see module docstring).
    # q_len == 1: measured 2.2-2.4x on real skewed batches (1h-validated path).
    # q_len > 1 (spec verify) is opt-in via VLLM_K3_TS_VAR_SPLIT_Q=1.
    # q_len > 1 var-split only for verify-sized batches (B >= 8): single-request
    # small-q prefill pieces keep the stock split (the measured fast path).
    if is_var_seq and is_fp8 and _K3_VAR_SPLIT and (q_len == 1 or (_K3_VAR_SPLIT_Q and B >= 8)):
        per_split = B * H_eff * q_len_eff * (kv_lora_rank + 1) * 4
        s_cap = min(_K3_MAX_SPLITS, workspace_buffer.numel() // max(1, per_split))
        # initialize_workspace places acc_lse at cosize(acc_o) * 32 // 8 in Int32:
        # acc_o >= 2^26 floats (256 MB) wraps the offset negative and the LSE
        # partials land before the workspace (seen as KV-cache corruption with a
        # 512 MB workspace). The production workspace (222 MB at H=96) stays below.
        s_cap = min(s_cap, ((1 << 31) - 1) // 32 // max(1, B * H_eff * q_len_eff * kv_lora_rank))
        if s_cap > split_kv:
            n_sm = max_active_blocks
            # k3_memo: per-step dict owned by the caller's decode metadata (one
            # object per step, shared by every MLA layer of the stage).
            memo_key = (id(cache_seqs), B, q_len_eff, s_cap, n_sm, _K3_VS_ALPHA, _K3_VS_CEIL)
            block_split_kvs = k3_memo.get(memo_key) if k3_memo is not None else None
            k3_fused = (_K3_WORK_PERM and _K3_FUSED_ORDER
                        and s_cap * B * q_len_eff <= _K3_ORDER_MAX_ITEMS)
            if block_split_kvs is None and k3_fused:
                k3_rows = B * q_len * H
                block_split_kvs = _k3_split_and_order(
                    cache_seqs, B, q_len_eff, s_cap,
                    _k3_num_clusters(2 if mma_qk_tiler_mn[0] == 128 else 1),
                    mma_qk_tiler_mn[1],
                    n_sm if q_len == 1 else int(round(n_sm * _K3_VS_ALPHA * 1024)),
                    1 if q_len == 1 else q_len_eff * 1024,
                    _K3_WORK_ORDER == 'lpt' or (
                        _K3_WORK_ORDER == 'auto' and k3_rows >= _K3_LPT_MIN_ROWS))
            tot = None if block_split_kvs is not None else torch.clamp(cache_seqs.sum(), min=1)
            if block_split_kvs is not None:
                pass
            elif q_len == 1:
                # floor share of #SM: total tiles <= #SM (never an extra wave);
                # uniform lengths reproduce the stock split.
                block_split_kvs = torch.clamp((cache_seqs.to(torch.int64) * n_sm) // tot, 1, s_cap).to(torch.int32)
            else:
                # q_len_eff query tiles per request already share the SMs:
                # target per-tile work <= alpha * (q_len_eff * sum K / #SM).
                num = cache_seqs.to(torch.int64) * int(round(n_sm * _K3_VS_ALPHA * 1024))
                den = tot * (q_len_eff * 1024)
                q_split = (num + den - 1) // den if _K3_VS_CEIL else num // den
                block_split_kvs = torch.clamp(q_split, 1, s_cap).to(torch.int32)
            if k3_memo is not None and memo_key not in k3_memo:
                # Reuse across layers: eager steps skip the recompute; a captured
                # graph records one computation that all layers read.
                k3_memo[memo_key] = block_split_kvs
            split_kv = s_cap
            if _K3_WORK_PERM and not k3_fused:
                perm_key = ('k3_work_perm',) + memo_key
                ordered = k3_memo.get(perm_key) if k3_memo is not None else None
                if ordered is None:
                    ordered = torch.cat((block_split_kvs, _k3_work_perm(
                        block_split_kvs, cache_seqs, B, q_len_eff, s_cap,
                        _k3_num_clusters(2 if mma_qk_tiler_mn[0] == 128 else 1),
                        mma_qk_tiler_mn[1],
                        use_lpt=_K3_WORK_ORDER == 'lpt' or (
                            _K3_WORK_ORDER == 'auto' and B * q_len * H >= _K3_LPT_MIN_ROWS))))
                    if k3_memo is not None:
                        k3_memo[perm_key] = ordered
                block_split_kvs = ordered
            use_work_perm = _K3_WORK_PERM
            workspace_size = BlackwellMultiHeadLatentAttentionForwardFP8.get_workspace_size(
                H_eff, q_len_eff, kv_lora_rank, B, split_kv, cutlass.Float32
            )
            assert workspace_buffer.numel() >= workspace_size
            workspace_bytes = workspace_buffer[:workspace_size]
            is_workspace_size_zero = False
            is_var_split_kv = True
            is_persistent = True

    # Validate configuration (cached, negligible overhead after first call)
    _check_can_implement(
        torch_dtype=q_dtype,
        page_size=page_size,
        num_heads=H,
        seq_len_q=q_len,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        compute_capability=compute_capability,
    )

    # Get compiled kernel (cached after first compile)
    # Note: when is_workspace_size_zero is True, workspace_bytes is None and it will launch one kernel without workspace.
    # Otherwise, workspace_bytes is not None and it will launch two kernels.
    compiled_kernel = _get_compiled_mla_kernel(
        torch_dtype=q_dtype,
        page_size=page_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        skip_correction_threshold=skip_correction_threshold,
        is_workspace_size_zero=is_workspace_size_zero,
        fold_sq_factor=fold_sq_factor,
        causal_mask=causal_mask,
        num_heads=H,
        seq_len_q=q_len,
        cp_world=cp_world,
        use_pdl=enable_pdl,
        return_lse=return_lse,
        compute_capability=compute_capability,
        use_work_perm=use_work_perm,
        direct_single=_K3_DIRECT_SINGLE and is_var_split_kv,
    )

    # DCP: allocate real LSE tensor when return_lse=True (DCP path). torch.zeros
    # (not empty) so any kernel-unwritten positions are a known value, not garbage.
    if return_lse:
        lse_real = torch.zeros((B, q_len, H), dtype=torch.float32, device=query.device)
    else:
        lse_real = None

    # TVM FFI env stream must be set to PyTorch's current stream so the kernel
    # runs on the same stream as upstream PyTorch ops. CuTe tensors flow through
    # __tvm_ffi_object__ which does NOT auto-infer PyTorch's current stream;
    # use_torch_stream() binds it explicitly. Symmetric with mla_prefill.py.
    import tvm_ffi

    call_args = [
        q_latent_k,
        q_rope_k,
        c_latent_k,
        c_rope_k,
        page_table_k,
        o_k,
        lse_real,
        workspace_bytes,
        Int32(split_kv),
        cache_seqs,
    ]
    # DCP: the per-request global causal bound is an fp8-decode-kernel-only
    # argument; the fp16 kernel masks against the local cache length.
    if is_fp8:
        call_args.append(causal_seqs_dev)
    call_args += [
        block_split_kvs,
        Float32(softmax_scale),
        Float32(output_scale),
    ]

    with tvm_ffi.use_torch_stream():
        compiled_kernel(*call_args)

    # DCP: return (o, lse) tuple when LSE was requested.
    o_result = out if out is not None else o_k
    if return_lse:
        return o_result, lse_real
    return o_result
