# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step host and device timestamps for pipeline analysis.

``VLLM_STEP_TIMELINE=<dir>`` makes TP rank 0 of every PP stage append one CSV
row per ``execute_model`` / ``sample_tokens`` call to
``<dir>/tl_pp<pp>_tp0.csv``: the host wall-clock interval of the call, the
step's composition and the wall-clock times at which the device reached named
points on the current stream (``g0`` call start, ``fs`` forward launch,
``g1`` call end). Device times come from CUDA events read back without
synchronizing the stream, anchored to wall clock through an event recorded on
an idle side stream, so they line up across stages and nodes up to NTP skew.
"""

import os
import time
from collections import deque

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Re-anchor device time to wall clock this often (steps), bounding drift and
# the float32 precision loss of long event intervals.
_ANCHOR_EVERY = 2000
_FLUSH_EVERY = 64


class StepTimeline:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w")  # noqa: SIM115
        self._file.write("kind,host_start,host_end,info,device\n")
        self._side = torch.cuda.Stream()
        self._pending: deque[dict] = deque()
        self._cur: dict | None = None
        self._num_ended = 0
        self._unflushed = 0
        self._anchor()
        logger.info("Writing the per-step timeline to %s", path)

    def _anchor(self) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(self._side)
        ev.synchronize()
        self._ref = (ev, time.time())
        self._anchored_at = self._num_ended

    def begin(self, kind: str, info: str) -> None:
        self._drain()
        self._cur = {"kind": kind, "info": info, "h0": time.time(), "ev": []}
        self.mark("g0")

    def mark(self, name: str) -> None:
        if self._cur is None:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._cur["ev"].append((name, ev))

    def end(self) -> None:
        cur = self._cur
        if cur is None:
            return
        self.mark("g1")
        cur["h1"] = time.time()
        cur["ref"] = self._ref
        self._pending.append(cur)
        self._cur = None
        self._num_ended += 1
        if self._num_ended - self._anchored_at >= _ANCHOR_EVERY:
            self._anchor()

    def _drain(self) -> None:
        # Marks may sit on different streams, so every event must be done.
        while self._pending and all(ev.query() for _, ev in self._pending[0]["ev"]):
            r = self._pending.popleft()
            ref, ref_t = r["ref"]
            dev = ";".join(
                f"{name}={ref_t + ref.elapsed_time(ev) / 1e3:.6f}"
                for name, ev in r["ev"]
            )
            self._file.write(
                f"{r['kind']},{r['h0']:.6f},{r['h1']:.6f},{r['info']},{dev}\n"
            )
            self._unflushed += 1
        if self._unflushed >= _FLUSH_EVERY:
            self._file.flush()
            self._unflushed = 0


def step_info(scheduler_output) -> str:
    """tokens|reqs|ctx_tokens|ctx_reqs|spec_tokens of one scheduled step.

    A request is decode when it was scheduled exactly one token plus its draft
    tokens; everything else counts as context (prefill) work.
    """
    spec = scheduler_output.scheduled_spec_decode_tokens
    ctx_tokens = ctx_reqs = spec_tokens = 0
    for req_id, n in scheduler_output.num_scheduled_tokens.items():
        k = len(spec.get(req_id, ()))
        spec_tokens += k
        if n != 1 + k:
            ctx_tokens += n
            ctx_reqs += 1
    return (
        f"{scheduler_output.total_num_scheduled_tokens}|"
        f"{len(scheduler_output.num_scheduled_tokens)}|"
        f"{ctx_tokens}|{ctx_reqs}|{spec_tokens}"
    )


def make_step_timeline(pp_rank: int, tp_rank: int) -> StepTimeline | None:
    import vllm.envs as envs

    directory = envs.VLLM_STEP_TIMELINE
    if not directory or tp_rank != 0:
        return None
    return StepTimeline(os.path.join(directory, f"tl_pp{pp_rank}_tp{tp_rank}.csv"))
