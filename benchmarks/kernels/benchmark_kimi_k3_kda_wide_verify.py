# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KdaWide against the Triton RecoverSSM verify of the regular Kimi-K3 KDA
decode, per KDA layer, at TP8 shapes (12 local heads x 128, q 5, FP32 SSM
state, BF16 conv state, the layer's FP32 conv1d weight; SM10x only).

  triton   causal_conv1d_update + kda_recoverssm_verify (+ the gated RMSNorm)
  kdawide  WideRecoverSSMVerify: one TVM-FFI CuTe launch (+ the fused norm)

``+norm`` rows are a pure speculative step (KdaWide fuses o_norm and returns);
``pre-norm`` rows a mixed step's speculative rows (o_norm runs separately on
both paths, so it is left out). Both paths keep the F_B GEMM of forward(), so
it is not timed. ``slot`` records: addressed by request slot (non-last PP
ranks).

graph: a CUDA graph with one call per layer (each layer its own weights, conv
and SSM state, records and inputs), replayed after an L2 flush (Triton
do_bench median), per layer: uniform speculative steps run this way (FULL
graphs capture ``_forward``). eager: wall time per layer of back-to-back eager
calls, as piecewise / mixed steps run ``_forward``. host: CPU time per layer to
issue the launches (no synchronization), and the Triton path's split into its
three launches.

same / max|d|/rms: bit-identical output fraction and the largest difference
relative to its (token, head) row's RMS, KdaWide against Triton from the same
caches. ``--bf16-conv-weight`` reproduces the earlier mismatch (the Triton
convolution rounds BF16 x BF16 products to BF16).

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

    def __init__(self, heads, window, batch, slots, record_rows, dim_first, args):
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
        self.conv_weight = torch.randn(3 * dim, 4, device=dev) * 0.5
        if args.bf16_conv_weight:
            self.conv_weight = self.conv_weight.to(bf16)
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

    def conv1d(self, meta):
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update,
        )

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

    def verify(self, meta):
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import kda_recoverssm_verify

        heads, tokens, dim = self.heads, self.tokens, self.heads * HEAD_DIM
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

    def norm(self, meta):
        """The KDA layer's ``core_attn_out.copy_(self.o_norm(core_attn_out, g2))``."""
        from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

        gate = self.gate.view(self.tokens, self.heads, HEAD_DIM)
        self.out.copy_(
            rms_norm_gated(self.out, gate, self.norm_weight, None, "sigmoid", eps=EPS)
        )

    def triton(self, meta, norm):
        """The regular path's Triton verify (``_recoverssm_verify``) + o_norm."""
        self.conv1d(meta)
        self.verify(meta)
        if norm:
            self.norm(meta)

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
    """Wall time per call of back-to-back eager calls (host- or GPU-bound)."""
    for call in calls:
        call()
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        for call in calls:
            call()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) * 1e6 / (iters * len(calls))


def _host_us(calls, iters: int = 10) -> float:
    """CPU time per call to issue the launches: timed without synchronizing
    (``iters * len(calls)`` launches stay well inside the launch queue)."""
    for call in calls:
        call()
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        for call in calls:
            call()
    elapsed = time.perf_counter() - start
    torch.accelerator.synchronize()
    return elapsed * 1e6 / (iters * len(calls))


def _difference(layer, meta, norm, live_tokens) -> tuple[float, float]:
    """Bit-identical fraction and max |KdaWide - Triton| / RMS of the (token,
    head) row over live rows, both from the same caches."""
    records = (layer.records.correction, layer.records.key_gate)
    saved = [t.clone() for t in (layer.conv, *records)]
    layer.triton(meta, norm)
    expected = layer.out[0, live_tokens].clone()
    for tensor, copy in zip((layer.conv, *records), saved):
        tensor.copy_(copy)
    layer.kdawide(meta, norm)
    torch.accelerator.synchronize()
    got = layer.out[0, live_tokens]
    same = (got.view(torch.int16) == expected.view(torch.int16)).float().mean()
    got, expected = got.float(), expected.float()
    rms = expected.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
    return same.item(), ((got - expected).abs() / rms).max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 48, 96])
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--window", type=int, default=5, help="q = drafts + 1")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--conv-layout", choices=("DS", "SD"), default="DS")
    parser.add_argument(
        "--bf16-conv-weight",
        action="store_true",
        help="BF16 conv1d weight (not the model's): KdaWide then cannot match",
    )
    args = parser.parse_args()

    from vllm.models.kimi_k3.nvidia.recoverssm import wide_verify
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(100):
        raise SystemExit("KdaWide needs an SM10x GPU")
    torch.manual_seed(0)
    dev = torch.device("cuda")
    print(
        f"heads {args.heads}, q {args.window}, {args.layers} layers, conv "
        f"{args.conv_layout}, conv1d weight "
        f"{'BF16' if args.bf16_conv_weight else 'FP32'}; us per layer"
    )
    header = (
        f"{'batch':>5} {'step':>8} {'records':>7} | {'graph tri':>9} {'kw':>6} "
        f"{'speedup':>7} | {'eager tri':>9} {'kw':>6} | {'host tri':>8} {'kw':>6} "
        f"| {'same':>6} {'max|d|/rms':>10}"
    )
    print(header)
    print("-" * len(header))
    breakdown = []
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
                    args,
                )
                for _ in range(args.layers)
            ]
            for norm in (True, False):
                wide_verify.WideRecoverSSMVerify._failed = None
                same, worst = _difference(layers[0], meta, norm, live_tokens)
                times = {}
                for path in ("triton", "kdawide"):
                    calls = [
                        functools.partial(getattr(layer, path), meta, norm)
                        for layer in layers
                    ]
                    times[path] = (_graph_us(calls), _eager_us(calls), _host_us(calls))
                (tri, tri_eager, tri_host) = times["triton"]
                (wide, wide_eager, wide_host) = times["kdawide"]
                print(
                    f"{batch:>5} {'+norm' if norm else 'pre-norm':>8} "
                    f"{'slot' if record_rows else 'row':>7} | {tri:>9.1f} "
                    f"{wide:>6.1f} {tri / wide:>6.2f}x | {tri_eager:>9.1f} "
                    f"{wide_eager:>6.1f} | {tri_host:>8.1f} {wide_host:>6.1f} | "
                    f"{same:>6.4f} {worst:>10.2e}"
                )
            if not record_rows:
                stages = [
                    _host_us([functools.partial(getattr(layer, stage), meta)])
                    for stage in ("conv1d", "verify", "norm")
                    for layer in layers[:1]
                ]
                breakdown.append((batch, *stages))
            del layers
            torch.cuda.empty_cache()
    print("\nTriton path host us per layer: conv1d update / verify / o_norm")
    for batch, conv, verify, norm in breakdown:
        print(f"{batch:>5}: {conv:.1f} / {verify:.1f} / {norm:.1f}")


if __name__ == "__main__":
    main()
