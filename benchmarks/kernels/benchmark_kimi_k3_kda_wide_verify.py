# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KdaWide against the Triton RecoverSSM verify of the regular Kimi-K3 KDA
decode, per KDA layer, at TP8 shapes (12 local heads x 128, q 5, FP32 SSM
state, BF16 conv state; SM10x only).

  triton   causal_conv1d_update + kda_recoverssm_verify (+ the gated RMSNorm)
  kdawide  WideRecoverSSMVerify: one CuTe launch (+ the fused gated RMSNorm)

``+norm`` rows are a pure speculative step (KdaWide fuses o_norm and returns);
``pre-norm`` rows a mixed step's speculative rows (o_norm runs separately on
both paths, so it is left out). Both paths keep the F_B GEMM of forward(), so
it is not timed. ``slot rows``: records addressed by request slot (non-last PP
ranks).

graph: a CUDA graph with one call per layer (each layer its own weights, conv
and SSM state, records and inputs), replayed after an L2 flush (Triton
do_bench median), per layer. eager: wall time per layer of back-to-back eager
calls; the server runs this verify eagerly (``_forward`` breaks the CUDA
graph), so host overhead counts there.

From the vllm root, on an SM10x GPU:
    python bench_kdawide.py [--batches 1 16 48 96] [--layers 8] [--conv-layout DS]
"""

import argparse
import functools
import time
from types import SimpleNamespace

import torch

HEAD_DIM = 128
LOWER_BOUND = -5.0
EPS = 1e-5  # FusedRMSNormGated's default, as the KDA layer builds it


class Layer:
    """One KDA layer's weights, caches, records and step inputs."""

    def __init__(self, heads, window, batch, slots, record_rows, dim_first):
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMRecords
        from vllm.models.kimi_k3.nvidia.recoverssm.wide_verify import (
            WideRecoverSSMVerify,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.prepared import (  # noqa: E501
            decode_conv_weight,
        )

        d, dev, bf16 = HEAD_DIM, torch.device("cuda"), torch.bfloat16
        dim, tokens = heads * d, batch * window
        self.heads, self.window, self.tokens = heads, window, tokens
        self.conv_weight = (torch.randn(3 * dim, 4, device=dev) * 0.5).to(bf16)
        self.f_b = (torch.randn(dim, d, device=dev) * d**-0.5).to(bf16)
        self.norm_weight = torch.randn(d, device=dev).to(bf16)
        self.A_log = torch.randn(heads, device=dev) * 0.5
        self.dt_bias = torch.randn(dim, device=dev) * 0.5
        shape = (3 * dim, window + 2) if dim_first else (window + 2, 3 * dim)
        pages = torch.randn(slots, *shape, device=dev).to(bf16)
        self.conv = pages if dim_first else pages.transpose(-1, -2)
        self.state = torch.randn(slots, heads, d, d, device=dev) * 0.1
        self.records = KDARecoverSSMRecords.allocate(
            batch + 4 if record_rows else batch, heads, window, d, bf16, dev
        )
        # The in-projection rows: q|k|v, output gate, f_a, beta, padding.
        projected = torch.randn(tokens, 4 * dim + d + heads + 4, device=dev)
        projected = projected.to(bf16)
        self.x, self.gate = projected[:, : 3 * dim], projected[:, 3 * dim : 4 * dim]
        self.f_a = projected[:, 4 * dim : 4 * dim + d]
        self.beta = projected[:, 4 * dim + d : 4 * dim + d + heads]
        self.raw_gate = (self.f_a.float() @ self.f_b.float().T).to(bf16)
        self.conv_out = torch.empty(tokens, 3 * dim, device=dev, dtype=bf16)
        self.out = torch.empty(1, tokens, heads, d, device=dev, dtype=bf16)
        self.wide = WideRecoverSSMVerify(
            SimpleNamespace(
                local_num_heads=heads,
                spec_query_len=window,
                conv1d=SimpleNamespace(bias=None),
                decode_conv1d_weight=decode_conv_weight(self.conv_weight.unsqueeze(1)),
                decode_norm_weight=self.norm_weight.float(),
                dt_bias=self.dt_bias,
                A_log=self.A_log,
                f_b_proj=SimpleNamespace(weight=self.f_b),
                gate_lower_bound=LOWER_BOUND,
                o_norm=SimpleNamespace(eps=EPS),
            )
        )

    def triton(self, meta, norm):
        """The regular path's Triton verify (``_recoverssm_verify``) + o_norm."""
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update,
        )
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import kda_recoverssm_verify
        from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

        heads, tokens, dim = self.heads, self.tokens, self.heads * HEAD_DIM
        causal_conv1d_update(
            self.x,
            self.conv,
            self.conv_weight,
            None,
            activation="silu",
            conv_state_indices=meta.slots,
            num_accepted_tokens=meta.accepted,
            query_start_loc=meta.query_start,
            max_query_len=self.window,
            validate_data=False,
            out=self.conv_out,
        )
        q, k, v = (
            t.unflatten(-1, (heads, HEAD_DIM)).unsqueeze(0)
            for t in self.conv_out.split(dim, -1)
        )
        kda_recoverssm_verify(
            q=q,
            k=k,
            v=v,
            raw_g=self.raw_gate.view(1, tokens, heads, HEAD_DIM),
            raw_beta=self.beta.unsqueeze(0),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=LOWER_BOUND,
            checkpoint_state=self.state,
            records=self.records,
            query_start_loc=meta.query_start,
            state_indices=meta.slots,
            spec_query_len=self.window,
            out=self.out,
            record_rows=meta.rows,
        )
        if norm:
            gate = self.gate.view(tokens, heads, HEAD_DIM)
            self.out.copy_(
                rms_norm_gated(
                    self.out, gate, self.norm_weight, None, "sigmoid", eps=EPS
                )
            )

    def kdawide(self, meta, norm):
        from vllm.models.kimi_k3.nvidia.recoverssm.wide_verify import (
            WideRecoverSSMVerify,
        )

        if not self.wide(
            x=self.x,
            f_a=self.f_a,
            raw_beta=self.beta,
            state_indices=meta.slots,
            num_accepted_tokens=meta.accepted,
            conv_state=self.conv,
            state=self.state,
            records=self.records,
            record_rows=meta.rows,
            out=self.out.view(self.tokens, -1),
            output_gate=self.gate if norm else None,
        ):
            raise RuntimeError(f"KdaWide declined ({WideRecoverSSMVerify._failed})")


def _graph_us(calls) -> float:
    from vllm.triton_utils import triton

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    ms = triton.testing.do_bench(graph.replay, warmup=25, rep=200, return_mode="median")
    return ms * 1e3 / len(calls)


def _eager_us(calls, iters: int = 30) -> float:
    for call in calls:
        call()
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        for call in calls:
            call()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) * 1e6 / (iters * len(calls))


def _difference(layer, meta, norm, live_tokens) -> tuple[float, float]:
    """Largest and mean |KdaWide - Triton| over live rows, from the same caches."""
    records = (layer.records.correction, layer.records.key_gate)
    saved = [t.clone() for t in (layer.conv, *records)]
    layer.triton(meta, norm)
    expected = layer.out.clone()
    for tensor, copy in zip((layer.conv, *records), saved):
        tensor.copy_(copy)
    layer.kdawide(meta, norm)
    torch.accelerator.synchronize()
    diff = (layer.out - expected)[0, live_tokens].float().abs()
    return diff.max().item(), diff.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 48, 96])
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--window", type=int, default=5, help="q = drafts + 1")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--conv-layout", choices=("DS", "SD"), default="DS")
    args = parser.parse_args()

    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(100):
        raise SystemExit("KdaWide needs an SM10x GPU")
    torch.manual_seed(0)
    dev = torch.device("cuda")
    print(
        f"heads {args.heads}, q {args.window}, {args.layers} layers, conv "
        f"{args.conv_layout}; us per layer (graph: CUDA graph replay, L2 flushed)"
    )
    header = (
        f"{'batch':>5} {'step':>8} {'records':>9} | {'triton':>8} {'kdawide':>8} "
        f"{'speedup':>7} | {'eager tri':>9} {'eager kw':>9} | {'max|d|':>8} "
        f"{'mean|d|':>8}"
    )
    print(header)
    print("-" * len(header))
    for batch in args.batches:
        slots = 2 * batch + 1
        tokens = batch * args.window
        for record_rows in (False, True):
            meta = SimpleNamespace(
                slots=torch.randperm(slots - 1, device=dev)[:batch].int() + 1,
                accepted=torch.ones(batch, dtype=torch.int32, device=dev),
                query_start=torch.arange(
                    0, tokens + 1, args.window, dtype=torch.int32, device=dev
                ),
                rows=(
                    torch.randperm(batch + 4, device=dev)[:batch].int()
                    if record_rows
                    else None
                ),
            )
            if batch > 4:
                meta.slots[3] = 0  # a CUDA graph padding row
            live_tokens = (meta.slots > 0).repeat_interleave(args.window)
            layers = [
                Layer(
                    args.heads,
                    args.window,
                    batch,
                    slots,
                    record_rows,
                    args.conv_layout == "DS",
                )
                for _ in range(args.layers)
            ]
            for norm in (True, False):
                diff = _difference(layers[0], meta, norm, live_tokens)
                times = {}
                for path in ("triton", "kdawide"):
                    calls = [
                        functools.partial(getattr(layer, path), meta, norm)
                        for layer in layers
                    ]
                    times[path] = (_graph_us(calls), _eager_us(calls))
                (tri, tri_eager), (wide, wide_eager) = times["triton"], times["kdawide"]
                print(
                    f"{batch:>5} {'+norm' if norm else 'pre-norm':>8} "
                    f"{'slot' if record_rows else 'row':>9} | {tri:>8.1f} "
                    f"{wide:>8.1f} {tri / wide:>6.2f}x | {tri_eager:>9.1f} "
                    f"{wide_eager:>9.1f} | {diff[0]:>8.2e} {diff[1]:>8.2e}"
                )
            del layers
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
