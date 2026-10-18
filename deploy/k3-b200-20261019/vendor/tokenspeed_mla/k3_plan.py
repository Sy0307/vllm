# K3: per-request variable-split planner and static work order for the persistent
# decode kernel. make_plan() returns int32 [B split counts | slot -> item order]
# (or just the split counts), to pass as split_plan= to tokenspeed_mla_decode.
import functools

import torch
import triton
import triton.language as tl

from tokenspeed_mla.utils import get_max_active_clusters

_K3_VS_CEIL = True


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
def _num_clusters(cluster_size: int) -> int:
    return get_max_active_clusters(cluster_size)


def make_plan(seqs, B, S, s_cap, alpha=1.0, n_sm=148, order="snake", k_tile=128, cluster_size=2):
    """seqs: int32 [B] cache lengths (one entry per kernel batch row); S: q tiles per
    request; s_cap: largest split; order: 'none' | 'snake' | 'lpt'."""
    out = _k3_split_and_order(
        seqs, B, S, s_cap, _num_clusters(cluster_size), k_tile,
        int(round(n_sm * alpha * 1024)), S * 1024, order == "lpt",
    )
    return out[:B] if order == "none" else out
