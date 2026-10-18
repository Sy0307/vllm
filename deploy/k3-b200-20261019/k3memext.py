# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read-only GPU memory audit for K3 workers (--worker-extension-cls k3memext.K3MemExt).

k3_mem_audit(OUT_DIR, LABEL) attributes this rank's device memory to model
weights, KV cache tensors, MegaMoE / symmetric-memory buffers, CUDA graph pools
and other live CUDA tensors, and writes mem-LABEL-HOST-rankR.json.
"""

import collections
import gc
import json
import os
import re
import socket

import torch
from k3opt_ext import K3OptExt

GiB = 1 << 30


def _storage(t):
    try:
        s = t.untyped_storage()
        return s.data_ptr(), s.nbytes()
    except Exception:
        return None, 0


class K3MemExt(K3OptExt):
    def k3_mem_audit(self, out_dir: str = "", label: str = "mem") -> str:
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        runner = self.model_runner
        out = {
            "rank": self.rank,
            "host": socket.gethostname(),
            "device_total": total,
            "device_used": total - free,
            "torch_reserved": stats.get("reserved_bytes.all.current", 0),
            "torch_allocated": stats.get("allocated_bytes.all.current", 0),
            "torch_peak_allocated": stats.get("allocated_bytes.all.peak", 0),
        }
        out["non_torch"] = out["device_used"] - out["torch_reserved"]
        owner = {}  # storage ptr -> (bytes, owner name)

        def claim(t, name):
            if isinstance(t, torch.Tensor) and t.is_cuda:
                p, n = _storage(t)
                if p and p not in owner:
                    owner[p] = (n, name)

        weights, kv, symm, kv_bytes = {}, {}, [], 0
        out["pools"] = []
        out["other_tensors_total"] = 0
        try:
            # 1. Weights / buffers by module category.
            def module_rows(mod, prefix):
                cat = collections.Counter()
                per_param = []
                for kind, it in (
                    ("param", mod.named_parameters(remove_duplicate=False)),
                    ("buffer", mod.named_buffers(remove_duplicate=False)),
                ):
                    for name, t in it:
                        p, n = _storage(t)
                        if not p or p in owner:
                            continue
                        full = f"{prefix}.{name}"
                        owner[p] = (n, f"{kind}:{full}")
                        key = re.sub(r"\.\d+\.", ".N.", full)
                        key = re.sub(r"\.experts\.\d+", ".experts.N", key)
                        cat[key] += n
                        per_param.append((n, full, list(t.shape), str(t.dtype)))
                per_param.sort(reverse=True)
                return cat, per_param[:30]

            model = getattr(runner, "model", None)
            weights = {}
            if model is not None:
                cat, top = module_rows(model, "model")
                weights["model"] = {
                    "total": sum(cat.values()),
                    "top_categories": cat.most_common(60),
                    "top_params": top,
                }
            spec = getattr(runner, "speculator", None)
            draft = getattr(spec, "model", None) if spec is not None else None
            if isinstance(draft, torch.nn.Module):
                cat, top = module_rows(draft, "draft")
                weights["draft"] = {
                    "total": sum(cat.values()),
                    "top_categories": cat.most_common(40),
                    "top_params": top,
                }
            out["weights"] = weights
        except Exception as e:  # keep the audit going
            out.setdefault("errors", {})["1"] = repr(e)
        try:
            # 2. KV cache: planned config and the storages actually allocated.
            kv = {}
            cfg = getattr(runner, "kv_cache_config", None)
            if cfg is not None:
                kv["num_blocks"] = cfg.num_blocks
                kv["tensor_sizes"] = [t.size for t in cfg.kv_cache_tensors]
                groups = []
                for g in cfg.kv_cache_groups:
                    s = g.kv_cache_spec
                    groups.append(
                        {
                            "layers": len(g.layer_names),
                            "type": type(s).__name__,
                            "block_size": getattr(s, "block_size", None),
                            "page_size_bytes": getattr(s, "page_size_bytes", None),
                            "first_layers": g.layer_names[:3],
                        }
                    )
                kv["groups"] = groups
            kv_bytes = 0

            def walk_kv(x, name):
                nonlocal kv_bytes
                if isinstance(x, torch.Tensor):
                    p, n = _storage(x)
                    if p and p not in owner:
                        owner[p] = (n, name)
                        kv_bytes += n
                elif isinstance(x, (list, tuple)):
                    for i, y in enumerate(x):
                        walk_kv(y, f"{name}[{i}]")
                elif isinstance(x, dict):
                    for k, y in x.items():
                        walk_kv(y, f"{name}[{k}]")

            walk_kv(getattr(runner, "kv_caches", None), "kv_caches")
            kv["allocated_bytes"] = kv_bytes
            out["kv"] = kv
        except Exception as e:  # keep the audit going
            out.setdefault("errors", {})["2"] = repr(e)
        try:
            # 3. MegaMoE symmetric buffers and torch symm_mem workspaces (non-torch).
            symm = []
            try:
                import vllm.models.kimi_k3.nvidia.model as km

                for key, b in km.KimiK3MegaMoEExperts._kimi_symm_buffer_cache.items():
                    buf = getattr(b, "buffer", None)
                    claim(buf, "symm:k3_megamoe")
                    symm.append(
                        {
                            "what": "k3_megamoe",
                            "tokens_per_rank": getattr(
                                b, "num_max_tokens_per_rank", None
                            ),
                            "bytes": buf.numel() * buf.element_size()
                            if buf is not None
                            else None,
                        }
                    )
            except Exception as e:
                symm.append({"error": repr(e)})
            try:
                import torch.distributed._symmetric_memory as sm

                ws = getattr(sm, "_group_name_to_workspace_tensor", {}) or {}
                for gname, t in ws.items():
                    if t is not None:
                        claim(t, f"symm_ws:{gname}")
                        symm.append(
                            {
                                "what": f"symm_ws:{gname}",
                                "bytes": t.numel() * t.element_size(),
                            }
                        )
            except Exception as e:
                symm.append({"error": repr(e)})
            out["symmetric"] = symm
        except Exception as e:  # keep the audit going
            out.setdefault("errors", {})["3"] = repr(e)
        try:
            # 4. Distributed groups inventory.
            groups = []
            try:
                from vllm.distributed import parallel_state as ps

                for name in dir(ps):
                    g = getattr(ps, name)
                    if isinstance(g, ps.GroupCoordinator):
                        comm = g.device_communicator
                        groups.append(
                            {
                                "var": name,
                                "name": g.unique_name,
                                "size": g.world_size,
                                "ranks": g.ranks[:8],
                                "comm": type(comm).__name__ if comm else None,
                                "parts": sorted(
                                    k
                                    for k, v in (vars(comm).items() if comm else [])
                                    if v is not None
                                    and k.endswith(("_comm", "communicator"))
                                ),
                                "a2a": type(
                                    getattr(comm, "all2all_manager", None)
                                ).__name__,
                                "a2a_handles": len(
                                    getattr(
                                        getattr(
                                            getattr(comm, "all2all_manager", None),
                                            "handle_cache",
                                            None,
                                        ),
                                        "_cache",
                                        {},
                                    )
                                    or {}
                                ),
                            }
                        )
            except Exception as e:
                groups.append({"error": repr(e)})
            out["groups"] = groups
        except Exception as e:  # keep the audit going
            out.setdefault("errors", {})["4"] = repr(e)
        try:
            # 5. Name other runner-held tensors (input buffers, graph I/O, workspaces).
            seen_obj = set()

            def walk(obj, name, depth):
                if depth > 4 or id(obj) in seen_obj:
                    return
                seen_obj.add(id(obj))
                if isinstance(obj, torch.Tensor):
                    claim(obj, name)
                    return
                if isinstance(obj, (str, bytes, int, float, bool, type(None))):
                    return
                if isinstance(obj, (list, tuple, set)):
                    for i, y in enumerate(list(obj)[:4096]):
                        walk(y, f"{name}[{i}]", depth + 1)
                    return
                if isinstance(obj, dict):
                    for k, y in list(obj.items())[:4096]:
                        walk(y, f"{name}[{k}]", depth + 1)
                    return
                d = getattr(obj, "__dict__", None)
                if (
                    d is None
                    or type(obj).__module__.startswith(("torch", "builtins", "numpy"))
                    and not isinstance(obj, torch.nn.Module)
                ):
                    return
                for k, y in list(d.items()):
                    walk(y, f"{name}.{k}", depth + 1)

            walk(runner, "runner", 0)
        except Exception as e:  # keep the audit going
            out.setdefault("errors", {})["5"] = repr(e)
        try:
            # 6. Allocator view: pools (graph pools are private ids) and big blocks.
            snap = torch.cuda.memory_snapshot()
            pools = collections.defaultdict(lambda: [0, 0, 0])
            unknown_blocks = []
            for seg in snap:
                pid = str(tuple(seg.get("segment_pool_id", (0, 0))))
                pools[pid][0] += seg["total_size"]
                pools[pid][1] += seg["allocated_size"]
                pools[pid][2] += 1
                addr = seg["address"]
                for b in seg["blocks"]:
                    if (
                        b["state"] == "active_allocated"
                        and b["size"] >= (32 << 20)
                        and addr not in owner
                    ):
                        unknown_blocks.append((b["size"], hex(addr), pid))
                    addr += b["size"]
            out["pools"] = sorted(
                ([k] + v for k, v in pools.items()), key=lambda r: -r[1]
            )
            unknown_blocks.sort(reverse=True)
            out["unknown_big_blocks"] = unknown_blocks[:40]
        except Exception as e:  # keep the audit going
            out.setdefault("errors", {})["6"] = repr(e)
        try:
            # 7. Remaining live CUDA tensors and their referrers.
            rest = {}
            for o in gc.get_objects():
                try:
                    if isinstance(o, torch.Tensor) and o.is_cuda:
                        p, n = _storage(o)
                        if p and p not in owner and p not in rest:
                            rest[p] = (n, o)
                except Exception:
                    pass
            rows = sorted(rest.values(), key=lambda r: -r[0])
            named = []
            for n, t in rows[:40]:
                ref = ""
                try:
                    for r in gc.get_referrers(t):
                        if isinstance(r, dict):
                            for rr in gc.get_referrers(r):
                                if getattr(rr, "__dict__", None) is r:
                                    k = next((k for k, v in r.items() if v is t), "?")
                                    ref = (
                                        f"{type(rr).__module__}.{type(rr).__name__}.{k}"
                                    )
                                    break
                        elif isinstance(r, (list, tuple)):
                            ref = ref or f"<{type(r).__name__} len={len(r)}>"
                        if ref and not ref.startswith("<"):
                            break
                except Exception:
                    pass
                named.append((n, list(t.shape), str(t.dtype), ref))
            out["other_tensors_total"] = sum(r[0] for r in rows)
            out["other_tensors_top"] = named
            by = collections.Counter()
            for p, (n, name) in owner.items():
                if name.startswith("runner"):
                    by[re.sub(r"\[\d+\]", "[i]", name)] += n
            out["runner_tensors_top"] = by.most_common(40)
        except Exception as e:  # keep the audit going
            out.setdefault("errors", {})["7"] = repr(e)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(
                os.path.join(
                    out_dir, f"{label}-{socket.gethostname()}-rank{self.rank}.json"
                ),
                "w",
            ) as f:
                json.dump(out, f, indent=1, default=str)
        g = lambda x: round(x / GiB, 3)
        return json.dumps(
            {
                "rank": self.rank,
                "used": g(out["device_used"]),
                "reserved": g(out["torch_reserved"]),
                "non_torch": g(out["non_torch"]),
                "kv": g(kv_bytes),
                "weights": {k: g(v["total"]) for k, v in weights.items()},
                "symm": [
                    (s.get("what"), g(s["bytes"]) if s.get("bytes") else None)
                    for s in symm
                ],
                "pools": [(r[0], g(r[1])) for r in out["pools"][:4]],
                "other": g(out["other_tensors_total"]),
            }
        )

    # Light-weight per-group block usage of the running requests (safe mid-run).
    def k3_kv_usage(self) -> str:
        runner = self.model_runner
        bt = runner.block_tables
        rs = getattr(runner, "req_states", None)
        active = (
            sorted(getattr(rs, "req_id_to_index", {}).values())
            if rs is not None
            else []
        )
        nb = bt.num_blocks.np
        rows = []
        for g, t in enumerate(bt.block_tables):
            tab = t.gpu
            if not active:
                rows.append({"g": g, "entries": 0, "unique": 0})
                continue
            idx = torch.tensor(active, device=tab.device, dtype=torch.long)
            n = torch.from_numpy(nb[g][active].astype("int64")).to(tab.device)
            m = torch.arange(tab.shape[1], device=tab.device)[None, :] < n[:, None]
            ids = tab[idx][m]
            ids = ids[ids > 0]
            rows.append(
                {
                    "g": g,
                    "entries": int(m.sum()),
                    "unique": int(torch.unique(ids).numel()),
                }
            )
        cfg = runner.kv_cache_config
        return json.dumps(
            {
                "rank": self.rank,
                "reqs": len(active),
                "num_blocks": cfg.num_blocks,
                "groups": rows,
            }
        )
