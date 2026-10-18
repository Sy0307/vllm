# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker extension for K3 optimization diagnostics (loaded via --worker-extension-cls).

k3_sync_debug("1") turns on torch.cuda sync-debug "warn" mode and records the
Python stack of every implicit host<->device synchronization; k3_sync_debug("0",
OUT_DIR) turns it off and writes the aggregated stacks per rank.
"""

import collections
import json
import os
import socket
import time
import traceback
import warnings

import torch


class K3OptExt:
    def k3_sync_debug(self, on: str, out_dir: str = "") -> str:
        if on == "1":
            stacks = collections.Counter()
            orig = warnings.showwarning
            self._k3_sync_state = (stacks, orig, time.time())

            def show(message, category, filename, lineno, file=None, line=None):
                if "synchroniz" in str(message).lower():
                    frames = traceback.extract_stack()[:-2]
                    frames = [f for f in frames if "warnings.py" not in f.filename][
                        -12:
                    ]
                    key = " <- ".join(
                        f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                        for f in reversed(frames)
                    )
                    stacks[(str(message)[:120], key)] += 1
                else:
                    orig(message, category, filename, lineno, file, line)

            warnings.showwarning = show
            warnings.simplefilter("always", UserWarning)
            torch.cuda.set_sync_debug_mode(1)
            return "on"
        torch.cuda.set_sync_debug_mode(0)
        stacks, orig, t0 = self._k3_sync_state
        warnings.showwarning = orig
        rank = getattr(self, "rank", None)
        rows = [
            {"count": c, "message": m, "stack": k} for (m, k), c in stacks.most_common()
        ]
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(
                os.path.join(out_dir, f"sync-{socket.gethostname()}-rank{rank}.json"),
                "w",
            ) as f:
                json.dump(
                    {"rank": rank, "seconds": time.time() - t0, "rows": rows},
                    f,
                    indent=1,
                )
        return json.dumps(
            {"rank": rank, "total": sum(stacks.values()), "sites": len(rows)}
        )

    # CPU profile of the worker thread to locate host overhead.
    def k3_cprofile(self, on: str, out_dir: str = "", label: str = "cprof") -> str:
        import cProfile
        import io
        import pstats

        if on == "1":
            self._k3_cprof = cProfile.Profile()
            self._k3_cprof.enable()
            return json.dumps({"rank": getattr(self, "rank", None), "cprofile": "on"})
        prof = self._k3_cprof
        prof.disable()
        rank = getattr(self, "rank", None)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            prof.dump_stats(os.path.join(out_dir, f"{label}-rank{rank}.prof"))
            s = io.StringIO()
            st = pstats.Stats(prof, stream=s).sort_stats("tottime")
            st.print_stats(60)
            s2 = io.StringIO()
            pstats.Stats(prof, stream=s2).sort_stats("cumulative").print_stats(80)
            with open(os.path.join(out_dir, f"{label}-rank{rank}.txt"), "w") as f:
                f.write(s.getvalue())
                f.write("\n\n==== cumulative ====\n")
                f.write(s2.getvalue())
        del self._k3_cprof
        return json.dumps({"rank": rank, "cprofile": "off"})

    # Per-step CUDA event timeline on TP0 of each stage.
    def k3_timeline_begin(
        self, label: str, out_dir: str, every: str = "1", limit: str = "100000"
    ) -> str:
        from collections import deque

        from vllm.distributed import get_pp_group, get_tp_group

        if get_tp_group().rank_in_group != 0:
            return json.dumps({"rank": self.rank, "enabled": False})
        assert not hasattr(self, "_tl_restore")
        self._tl_restore, self._tl_rows, self._tl_pending = [], [], deque()
        self._tl_seq, self._tl_on = 0, False
        self._tl_label, self._tl_out = label, out_dir
        every, limit = int(every), int(limit)
        anchor = torch.cuda.Event(enable_timing=True)
        anchor.record()
        self._tl_anchor = anchor
        runner = self.model_runner

        def collect():
            while self._tl_pending and self._tl_pending[0][1].query():
                a, b, row = self._tl_pending.popleft()
                row["gpu_start_ms"] = anchor.elapsed_time(a)
                row["gpu_end_ms"] = anchor.elapsed_time(b)
                self._tl_rows.append(row)

        def wrap(obj, name, gpu=False, outer=False, tag=None):
            if obj is None or not hasattr(obj, name):
                return
            original = getattr(obj, name)
            self._tl_restore.append((obj, name, original))
            method = tag or name

            def call(*args, **kwargs):
                if outer:
                    collect()
                    self._tl_seq += 1
                    self._tl_on = self._tl_seq % every == 0 and self._tl_seq <= limit
                if not self._tl_on:
                    return original(*args, **kwargs)
                row = dict(
                    seq=self._tl_seq,
                    method=("worker_execute" if outer else method),
                    cpu_start_ns=time.perf_counter_ns(),
                )
                if gpu:
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    a.record()
                result = original(*args, **kwargs)
                if gpu:
                    b.record()
                row["cpu_end_ns"] = time.perf_counter_ns()
                if (
                    name == "execute_model"
                    and not outer
                    and getattr(runner, "execute_model_state", None) is not None
                ):
                    batch = runner.execute_model_state.input_batch
                    row.update(
                        query_lens=batch.num_scheduled_tokens.tolist(),
                        contexts=batch.num_computed_tokens_np.tolist(),
                        num_tokens=int(batch.num_tokens),
                        padded_tokens=int(batch.num_tokens_after_padding),
                        has_prefill=bool(batch.has_prefill),
                        req_ids=list(batch.req_ids),
                    )
                if gpu:
                    self._tl_pending.append((a, b, row))
                else:
                    self._tl_rows.append(row)
                return result

            setattr(obj, name, call)

        wrap(self, "execute_model", outer=True)
        pp = get_pp_group()
        for name in (
            "recv_object",
            "irecv_tensor_dict",
            "isend_object",
            "isend_tensor_dict",
            "broadcast_object",
            "broadcast_tensor_dict",
        ):
            wrap(pp, name, gpu=True, tag="pp." + name)
        wrap(runner, "execute_model", gpu=True)
        wrap(runner, "sample_tokens", gpu=True)
        for name in (
            "prepare_inputs",
            "prepare_attn",
            "postprocess_sampled",
            "take_draft_token_ids",
        ):
            wrap(runner, name, gpu=True)
        wrap(
            getattr(runner, "speculator", None),
            "propose",
            gpu=True,
            tag="speculator.propose",
        )
        wrap(getattr(runner, "pp_handler", None), "get_prev_sampled_outputs", gpu=True)
        wrap(
            getattr(runner, "pp_handler", None),
            "receive",
            gpu=True,
            tag="pp_handler.receive",
        )
        wrap(getattr(runner, "cudagraph_manager", None), "run_fullgraph", gpu=True)
        wrap(getattr(runner, "cudagraph_manager", None), "run_pw_graph", gpu=True)
        return json.dumps({"rank": self.rank, "enabled": True})

    def k3_timeline_end(self) -> str:
        if not hasattr(self, "_tl_restore"):
            return json.dumps({"rank": self.rank, "enabled": False})
        for obj, name, original in reversed(self._tl_restore):
            setattr(obj, name, original)
        if self._tl_pending:
            self._tl_pending[-1][1].synchronize()
        anchor = self._tl_anchor
        while self._tl_pending:
            a, b, row = self._tl_pending.popleft()
            row["gpu_start_ms"] = anchor.elapsed_time(a)
            row["gpu_end_ms"] = anchor.elapsed_time(b)
            self._tl_rows.append(row)
        os.makedirs(os.path.join(self._tl_out, "timing"), exist_ok=True)
        p = os.path.join(
            self._tl_out, "timing", f"{self._tl_label}-rank{self.rank}.json"
        )
        with open(p, "w") as f:
            json.dump(
                {"rank": self.rank, "label": self._tl_label, "rows": self._tl_rows}, f
            )
        n = len(self._tl_rows)
        del self._tl_restore, self._tl_rows, self._tl_pending, self._tl_anchor
        return json.dumps({"rank": self.rank, "path": p, "rows": n})

    # DSpark local-argmax verification counter.
    def k3_draft_la_stats(self) -> str:
        spec = getattr(self.model_runner, "speculator", None)
        f = getattr(spec, "k3_draft_local_argmax_stats", None)
        return json.dumps({"rank": self.rank, "stats": f() if f is not None else None})
