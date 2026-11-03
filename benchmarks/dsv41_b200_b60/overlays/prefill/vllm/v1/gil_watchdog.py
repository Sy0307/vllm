# SPDX-License-Identifier: Apache-2.0
"""VLLM_GIL_WATCHDOG=period_s: per-process host stall diagnostics (P EngineCore and worker).

A daemon thread sleeps 5 ms in a loop and records how late it wakes up. Pure-Python threads hand the GIL over every
switch interval (5 ms), so lateness far above that means the GIL was held without a switch: a garbage-collection
pass or a C extension call that does not release it. gc.callbacks time every collection per generation, so the two
causes can be told apart. Every period_s one line: late wakeups > 20 / > 100 ms, and GC count / sum / max per
generation. VLLM_GIL_WATCHDOG_STACKS=ms[,after_s]: on a wake-up more than ms late, also log every Python thread's
innermost frames (at most 30 times, none in the first after_s seconds: start-up compiles hold the GIL); the thread that
held the GIL has just released it, so it is at or after that call."""
import gc
import os
import sys
import threading
import time
import traceback

from vllm.logger import init_logger

logger = init_logger(__name__)


def start(tag: str) -> None:
    period = float(os.environ.get("VLLM_GIL_WATCHDOG", "0"))
    if not period:
        return
    gcs = [[0, 0.0, 0.0] for _ in range(3)]  # per generation: count, sum ms, max ms
    t_gc = [0.0]

    def on_gc(phase: str, info: dict) -> None:
        if phase == "start":
            t_gc[0] = time.perf_counter()
        else:
            ms = (time.perf_counter() - t_gc[0]) * 1e3
            g = gcs[info.get("generation", 0)]
            g[0] += 1
            g[1] += ms
            g[2] = max(g[2], ms)

    gc.callbacks.append(on_gc)
    stacks_spec = os.environ.get("VLLM_GIL_WATCHDOG_STACKS", "0").split(",")
    stacks_ms = float(stacks_spec[0])
    stacks_from = time.monotonic() + (float(stacks_spec[1]) if len(stacks_spec) > 1 else 0.0)
    dumps = [0]

    def dump_stacks(late: float) -> None:
        dumps[0] += 1
        frames = sys._current_frames()
        lines = []
        for th in threading.enumerate():
            frame = frames.get(th.ident)
            if frame is None or th is threading.current_thread():
                continue
            stack = traceback.extract_stack(frame)[-4:]
            lines.append(f"{th.name}: " + " < ".join(
                f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(stack)))
        logger.warning("[gil-watchdog] %s late %.0f ms (dump %d), threads: %s", tag, late, dumps[0],
                       " || ".join(lines))

    def run() -> None:
        n20 = n100 = 0
        s20 = worst = 0.0
        t_log = time.monotonic() + period
        while True:
            t = time.perf_counter()
            time.sleep(0.005)
            late = (time.perf_counter() - t - 0.005) * 1e3
            if stacks_ms and late > stacks_ms and dumps[0] < 30 and time.monotonic() >= stacks_from:
                dump_stacks(late)
            if late > 20:
                n20 += 1
                s20 += late
                n100 += late > 100
                worst = max(worst, late)
            if time.monotonic() >= t_log:
                logger.info(
                    "[gil-watchdog] %s %.0f s: late >20 ms %d (sum %.0f ms), >100 ms %d, max %.0f ms | "
                    "gc gen0 %d/%.0f/%.0f gen1 %d/%.0f/%.0f gen2 %d/%.0f/%.0f (count/sum ms/max ms)",
                    tag, period, n20, s20, n100, worst, *[x for g in gcs for x in g],
                )
                n20 = n100 = 0
                s20 = worst = 0.0
                for g in gcs:
                    g[:] = [0, 0.0, 0.0]
                t_log += period

    threading.Thread(target=run, name="gil-watchdog", daemon=True).start()
