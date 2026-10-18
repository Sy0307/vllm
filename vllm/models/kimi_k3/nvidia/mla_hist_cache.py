# SPDX-License-Identifier: Apache-2.0
"""Cross-step reuse of the DCP chunked-context K/V of a multi-chunk prefill.

A prompt that needs several scheduler steps (more new tokens than the step
budget) re-processes its whole context on every chunk: under DCP each MLA
layer gathers the context latent shard, all-gathers it, dequantizes it,
up-projects it with ``kv_b_proj`` and packs fp8 K/V, although all of that is
identical to what the previous chunk computed except for the tokens that
chunk added.

This module keeps, per request and MLA layer, the packed fp8 K/V of the
context computed so far, indexed by (DCP rank, local row) -- the layout the
impl's all-gather produces before ``reorg_kvcache`` -- so a later chunk can
rebuild exactly the K/V tensors the impl loop would have fed the prefill
backend, after processing only the tokens added since. The cached rows come
from the same gather / dequant / ``kv_b_proj`` / pack kernels as the impl
loop, so reuse is exact as long as each up-projected row does not depend on
which other rows share the GEMM; ``VLLM_K3_MLA_HIST_CACHE_VERIFY=1``
recomputes every reused chunk and counts byte mismatches.

Only requests whose prompt continues after the current step are cached, the
total size is bounded by ``VLLM_K3_MLA_HIST_CACHE_GIB`` per GPU (whole
requests are evicted LRU), and entries of requests that stop appearing are
dropped.
"""

from __future__ import annotations

import collections
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_K3_MLA_HIST_CACHE", "0") == "1"
VERIFY = os.environ.get("VLLM_K3_MLA_HIST_CACHE_VERIFY", "0") == "1"
# Optional runtime switch for VERIFY: on while this file contains "1" (re-read
# every step), so one server can run the same prompts with and without reuse.
_VERIFY_FILE = os.environ.get("VLLM_K3_MLA_HIST_CACHE_VERIFY_FILE", "")
BUDGET_BYTES = int(float(os.environ.get("VLLM_K3_MLA_HIST_CACHE_GIB", "6")) * (1 << 30))
# Drop a request's entries after this many steps without seeing it.
_IDLE_STEPS = 64
_LOG_EVERY = int(os.environ.get("VLLM_K3_MLA_HIST_CACHE_LOG_EVERY", "4096"))


@dataclass
class _StepInfo:
    req_ids: list[str]
    num_computed: np.ndarray  # context length per batch request
    continues: np.ndarray  # prompt still has tokens after this step


_step: _StepInfo | None = None
_step_index = 0
# Prompt length of the requests in the current batch (sizes new entries).
_prefill_len: dict[str, int] = {}


@dataclass
class Entry:
    # Packed context K/V per (DCP rank, local row); rows below ``local_valid``
    # on every rank hold the request's first ``h`` context tokens.
    k: torch.Tensor  # [dcp, L, H, Dk] fp8
    v: torch.Tensor  # [dcp, L, H, Dv] fp8
    h: int = 0
    local_valid: list[int] = field(default_factory=list)
    # (offset, bytes) of the pool blocks backing k and v.
    pool_blocks: list = field(default_factory=list)


class _ReqState:
    __slots__ = ("layers", "nbytes", "last_step")

    def __init__(self) -> None:
        self.layers: dict[str, Entry] = {}
        # Budget reserved for the request (all its layers), charged on admission.
        self.nbytes = 0
        self.last_step = 0


_reqs: "collections.OrderedDict[str, _ReqState]" = collections.OrderedDict()
_total_bytes = 0
stats = collections.Counter()

# Entries live in one pool reserved when the model is built, so vLLM's memory
# profile (and hence the KV cache size) accounts for it and nothing is
# allocated at run time. Allocation is a deterministic first-fit over the pool:
# every rank sees the same request sequence and makes the same decisions, which
# the chunk loop relies on (a reused chunk skips the DCP all-gather).
_pool: torch.Tensor | None = None
_free: list[list[int]] = []  # sorted [offset, size]
_ALIGN = 256
# MLA layers on this rank (each calls init_pool once); a request is admitted
# for all of them at once so it is never cached in only some layers.
_n_layers = 0


def init_pool(device) -> None:
    global _pool, _n_layers
    if not ENABLED:
        return
    _n_layers += 1
    if _pool is not None:
        return
    _pool = torch.empty(BUDGET_BYTES, dtype=torch.uint8, device=device)
    _free[:] = [[0, BUDGET_BYTES]]
    logger.info("K3 MLA hist cache: reserved %.2f GiB pool", BUDGET_BYTES / (1 << 30))


def _pool_alloc(nbytes: int) -> int | None:
    nbytes = -(-nbytes // _ALIGN) * _ALIGN
    for i, (off, size) in enumerate(_free):
        if size >= nbytes:
            if size == nbytes:
                _free.pop(i)
            else:
                _free[i] = [off + nbytes, size - nbytes]
            return off
    return None


def _pool_release(off: int, nbytes: int) -> None:
    nbytes = -(-nbytes // _ALIGN) * _ALIGN
    import bisect

    i = bisect.bisect_left(_free, [off, 0])
    _free.insert(i, [off, nbytes])
    # coalesce with neighbours
    if i + 1 < len(_free) and _free[i][0] + _free[i][1] == _free[i + 1][0]:
        _free[i][1] += _free.pop(i + 1)[1]
    if i > 0 and _free[i - 1][0] + _free[i - 1][1] == _free[i][0]:
        _free[i - 1][1] += _free.pop(i)[1]


def set_step(
    req_ids: list[str],
    num_computed_np: np.ndarray,
    num_scheduled_np: np.ndarray,
    prefill_len_np: np.ndarray,
) -> None:
    """Record the batch of the forward about to run (called by the runner)."""
    global _step, _step_index, VERIFY
    _step_index += 1
    if _VERIFY_FILE:
        try:
            with open(_VERIFY_FILE) as f:
                VERIFY = f.read().strip() == "1"
        except OSError:
            VERIFY = False
    n = len(req_ids)
    num_computed = np.asarray(num_computed_np[:n], dtype=np.int64)
    sched = np.asarray(num_scheduled_np[:n], dtype=np.int64)
    plen = np.asarray(prefill_len_np[:n], dtype=np.int64)
    _step = _StepInfo(list(req_ids), num_computed, (num_computed + sched) < plen)
    _prefill_len.clear()
    _prefill_len.update(zip(req_ids, plen.tolist()))
    for rid in req_ids:
        st = _reqs.get(rid)
        if st is not None:
            st.last_step = _step_index
            _reqs.move_to_end(rid)
    stale = [rid for rid, st in _reqs.items() if _step_index - st.last_step > _IDLE_STEPS]
    for rid in stale:
        drop(rid)
    if _LOG_EVERY and _step_index % _LOG_EVERY == 0 and stats:
        logger.info(
            "K3 MLA hist cache: %s, %d reqs, %.2f GiB",
            dict(stats), len(_reqs), _total_bytes / (1 << 30),
        )


def clear_step() -> None:
    global _step
    _step = None


def prefill_requests(num_decodes: int, context_lens: list[int]):
    """Map the attention batch's prefill requests to (req_id, continues).

    Returns None (no caching this step) unless the runner's batch matches the
    MLA prefill layout: decodes first, then prefills whose context length is
    the request's computed-token count.
    """
    st = _step
    if st is None:
        return None
    out = []
    for i, ctx in enumerate(context_lens):
        j = num_decodes + i
        if j >= len(st.req_ids) or int(st.num_computed[j]) != int(ctx):
            stats["mapping_mismatch"] += 1
            return None
        out.append((st.req_ids[j], bool(st.continues[j])))
    return out


def max_local_rows(req_id: str, virtual_block: int, local_block: int) -> int:
    """Per-rank local rows of the request's whole prompt (0 if unknown)."""
    plen = _prefill_len.get(req_id, 0)
    return -(-plen // virtual_block) * local_block


def get(layer: str, req_id: str) -> Entry | None:
    st = _reqs.get(req_id)
    if st is None:
        return None
    return st.layers.get(layer)


def _free_entry(e: Entry) -> None:
    for off, nbytes in e.pool_blocks:
        _pool_release(off, nbytes)
    e.pool_blocks = []


def drop(req_id: str) -> None:
    global _total_bytes
    st = _reqs.pop(req_id, None)
    if st is not None:
        _total_bytes -= st.nbytes
        for e in st.layers.values():
            _free_entry(e)


def drop_layer(layer: str, req_id: str) -> None:
    global _total_bytes
    st = _reqs.get(req_id)
    if st is None:
        return
    e = st.layers.pop(layer, None)
    if e is not None:
        _free_entry(e)
        if not st.layers:
            # Last cached layer gone: release the request's reservation.
            _reqs.pop(req_id, None)
            _total_bytes -= st.nbytes


def create(
    layer: str,
    req_id: str,
    dcp: int,
    local_rows: int,
    heads: int,
    dk: int,
    dv: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Entry | None:
    """Allocate a request's entry for one layer.

    The first layer to cache a request admits it for every MLA layer on this
    rank: the whole reservation must fit in half the budget, and LRU requests
    are evicted to make room. Later layers then only carve their blocks.
    """
    global _total_bytes
    esize = torch.empty((), dtype=dtype).element_size()
    kb = dcp * local_rows * heads * dk * esize
    vb = dcp * local_rows * heads * dv * esize
    st = _reqs.get(req_id)
    if st is None:
        total = (kb + vb) * max(1, _n_layers)
        if total > BUDGET_BYTES // 2:
            stats["too_large"] += 1
            return None
        while _total_bytes + total > BUDGET_BYTES and _reqs:
            drop(next(iter(_reqs)))
            stats["evicted"] += 1
        if _total_bytes + total > BUDGET_BYTES:
            stats["no_room"] += 1
            return None
        st = _reqs[req_id] = _ReqState()
        st.nbytes = total
        _total_bytes += total
        stats["admitted"] += 1
    st.last_step = _step_index
    if _pool is not None:
        k_off = _pool_alloc(kb)
        v_off = _pool_alloc(vb) if k_off is not None else None
        if k_off is None or v_off is None:
            if k_off is not None:
                _pool_release(k_off, kb)
            stats["pool_full"] += 1
            return None
        k = _pool[k_off : k_off + kb].view(dtype).view(dcp, local_rows, heads, dk)
        v = _pool[v_off : v_off + vb].view(dtype).view(dcp, local_rows, heads, dv)
        e = Entry(k=k, v=v, h=0, local_valid=[0] * dcp, pool_blocks=[(k_off, kb), (v_off, vb)])
    else:
        e = Entry(
            k=torch.empty((dcp, local_rows, heads, dk), dtype=dtype, device=device),
            v=torch.empty((dcp, local_rows, heads, dv), dtype=dtype, device=device),
            h=0,
            local_valid=[0] * dcp,
        )
    st.layers[layer] = e
    return e
