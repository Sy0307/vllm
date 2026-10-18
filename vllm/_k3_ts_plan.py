# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3: per-step split plans for TokenSpeed MLA decode (tokenspeed-mla >= 2.0.1
with the ``split_plan`` extension).

``decode(ts_decode, ...)`` wraps ``tokenspeed_mla.tokenspeed_mla_decode``:

- ``VLLM_K3_TS_PLAN=1``: per-request split counts plus a static work order for
  the persistent clusters (``tokenspeed_mla.k3_plan``), built once per step and
  shared by every MLA layer through the decode metadata's memo dict.
- ``VLLM_K3_TS_MIN_SPLIT=N`` (plan off): upstream's uniform split floor, using
  the largest power of two <= N whose partials fit the workspace.
"""

import os

import torch

_PLAN = os.environ.get("VLLM_K3_TS_PLAN", "0") == "1"
_ALPHA = float(os.environ.get("VLLM_K3_TS_VS_ALPHA", "1.0"))
# 'snake' | 'lpt' | 'auto' (lpt from 56 requests of 96 heads x 5 tokens)
_ORDER = os.environ.get("VLLM_K3_TS_WORK_ORDER", "auto")
_LPT_MIN_ROWS = int(os.environ.get("VLLM_K3_TS_LPT_MIN_ROWS", str(56 * 5 * 96)))
# Largest split per request; independent of the workspace size (more empty
# persistent slots cost more than the extra balance buys).
_SPLIT_CAP = int(os.environ.get("VLLM_K3_TS_SPLIT_CAP", "8"))
_MIN_SPLIT = int(os.environ.get("VLLM_K3_TS_MIN_SPLIT", "1"))


def _layout(H: int, q_len: int, device: torch.device) -> tuple[int, int, int, int]:
    """(H_eff, q_len_eff, mma_m, k_tile) exactly as the upstream wrapper picks
    them for the non-packed path."""
    from tokenspeed_mla.mla_helpers import (
        get_mla_decode_fold_sq_factor,
        select_mla_decode_tilers,
    )

    cc = torch.cuda.get_device_capability(device)
    (mma_m, k_tile), _ = select_mla_decode_tilers(
        H, q_len, is_fp8=True, compute_capability=cc
    )
    fold = get_mla_decode_fold_sq_factor(H, q_len, mma_m)
    return H * fold, q_len // fold, mma_m, k_tile


def decode(ts_decode, *, query, seq_lens, workspace_buffer, kv_lora_rank,
           k3_memo=None, **kw):
    B, q_len, H, _ = query.shape
    fp8 = query.dtype == torch.float8_e4m3fn
    if fp8 and q_len > 1 and B >= 8 and (_PLAN or _MIN_SPLIT > 1):
        H_eff, S, mma_m, k_tile = _layout(H, q_len, query.device)
        per_split = B * H_eff * S * (kv_lora_rank + 1) * 4
        fit = workspace_buffer.numel() // max(1, per_split)
        if _PLAN and k3_memo is not None and min(_SPLIT_CAP, fit) >= 2:
            s_cap = min(_SPLIT_CAP, fit)
            key = ("k3_ts_plan", id(seq_lens), B, S, s_cap)
            plan = k3_memo.get(key)
            if plan is None:
                from tokenspeed_mla import k3_plan
                from tokenspeed_mla.utils import get_num_sm

                order = _ORDER
                if order == "auto":
                    order = "lpt" if B * q_len * H >= _LPT_MIN_ROWS else "snake"
                seqs = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
                plan = k3_plan.make_plan(
                    seqs, B, S, s_cap, alpha=_ALPHA, n_sm=get_num_sm(query.device),
                    order=order, k_tile=k_tile, cluster_size=2 if mma_m == 128 else 1,
                )
                k3_memo[key] = plan
            return ts_decode(query=query, seq_lens=seq_lens,
                             workspace_buffer=workspace_buffer,
                             kv_lora_rank=kv_lora_rank, split_plan=plan,
                             split_plan_cap=s_cap, **kw)
        if not _PLAN and _MIN_SPLIT > 1:
            m = _MIN_SPLIT
            while m > 1 and m > fit:
                m //= 2
            if m > 1:
                kw["min_split_kv"] = m
    return ts_decode(query=query, seq_lens=seq_lens, workspace_buffer=workspace_buffer,
                     kv_lora_rank=kv_lora_rank, **kw)
