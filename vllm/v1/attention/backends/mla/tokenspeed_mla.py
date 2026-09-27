# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenSpeed CuTe DSL MLA decode backend (Blackwell, FP8 KV cache only)."""

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Workspace upper bound for tokenspeed_mla_decode (per-device, lazy):
#   num_sms * num_heads * MAX_Q_LEN * (kv_lora_rank + 1) * sizeof(float32)
# Matches the kernel's `get_workspace_size` formula. MAX_Q_LEN=8 covers up to
# EAGLE3 / MTP-2 spec decoding query lengths; larger q_len fails the kernel's
# own buffer check.
_TOKENSPEED_MAX_Q_LEN = 8
import os as _k3hf_os
_K3_HEAD_FOLD = _k3hf_os.environ.get("VLLM_K3_TS_HEAD_FOLD", "0") == "1"
_K3_HF_MAP = {int(_a): int(_b) for _a, _b in (x.split(":") for x in _k3hf_os.environ.get("VLLM_K3_TS_HEAD_FOLD_MAP", "5:4,4:3").split(",") if x)}
_K3_HF_FASTCOPY = _k3hf_os.environ.get("VLLM_K3_HF_FASTCOPY", "0") == "1"
_K3_HF_VALIDATE = _k3hf_os.environ.get("VLLM_K3_HF_VALIDATE", "0") == "1"
_K3_HF_CTRL = _k3hf_os.environ.get("VLLM_K3_HF_VALIDATE_CTRL", "0") == "1"
_K3_HF_ERR: dict = {}


def _k3_hf_record(o_f, lse_f, o_p, lse_p, B, qn):
    buf = next(iter(_K3_HF_ERR.values()), None)
    if buf is None:
        return
    of = o_f.float(); op = o_p.float()
    ff = torch.isfinite(of); fp = torch.isfinite(op)
    buf[10:11].add_((~fp).sum().float().view(1))
    buf[11:12].add_((ff != fp).sum().float().view(1))
    m = ff & fp
    d = torch.where(m, (of - op).abs(), torch.zeros_like(of))
    a = torch.where(m, op.abs(), torch.zeros_like(op))
    torch.maximum(buf[0:1], (d.amax() / a.amax().clamp_min(1e-20)).view(1), out=buf[0:1])
    rel_mean = d.sum() / a.sum().clamp_min(1e-20)
    torch.maximum(buf[1:2], rel_mean.view(1), out=buf[1:2])
    if lse_f is not None and lse_p is not None:
        dl = (lse_f.float() - lse_p.float()).abs()
        dl = torch.where(torch.isfinite(dl), dl, torch.zeros_like(dl))
        torch.maximum(buf[2:3], dl.amax().view(1), out=buf[2:3])
    pos = d.view(B, qn, -1).sum(dim=(0, 2)) / a.view(B, qn, -1).sum(dim=(0, 2)).clamp_min(1e-20)
    torch.maximum(buf[3:3 + qn], pos, out=buf[3:3 + qn])
    buf[8:9].add_(1.0)
    buf[9:10].add_(rel_mean.view(1))


import os as _k3os
_K3_SMALLQ = int(_k3os.environ.get("VLLM_K3_SMALLQ_MQA", "0") or 0)
_K3_SMALLQ_MIN_CTX = 1024
_K3_SMALLQ_SPEC = _k3os.environ.get("VLLM_K3_SMALLQ_SPEC", "0") == "1"
# Front-pad lone small prefills to the next compiled power of two (one kernel
# call instead of up to log2(q) pieces). Fake rows sit before the real ones, so
# their causal bounds are smaller and only read already-written KV.
_K3_SMALLQ_PAD = _k3os.environ.get("VLLM_K3_SMALLQ_PAD", "0") == "1"


def _k3_pieces(q: int) -> list[tuple[int, int]]:
    """Exact power-of-two pieces (offset, length) covering q rows in order."""
    out, a = [], 0
    while a < q:
        n = 1 << ((q - a).bit_length() - 1)
        out.append((a, n)); a += n
    return out

_g_workspace: dict[torch.device, torch.Tensor] = {}


def _get_workspace(
    device: torch.device, num_heads: int, kv_lora_rank: int
) -> torch.Tensor:
    from tokenspeed_mla import get_num_sm

    needed = (
        get_num_sm(device) * num_heads * _TOKENSPEED_MAX_Q_LEN * (kv_lora_rank + 1) * 4
    )
    existing = _g_workspace.get(device)
    if existing is None or existing.numel() < needed:
        _g_workspace[device] = torch.empty(needed, dtype=torch.int8, device=device)
    return _g_workspace[device]


class TokenspeedMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    # The kernel accepts an explicit causal mask, so a non-causal DSpark
    # block can remain fused instead of being flattened to single tokens.
    supports_non_causal_multi_token_decode: ClassVar[bool] = True
    supports_non_causal_multi_token_dcp: ClassVar[bool] = True

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            MLACommonMetadata,
            supports_dcp_with_varlen=True,
        )
        self._k3_smallq = 0
        self._k3_hfv_n = 0
        if _k3hf_os.environ.get("VLLM_K3_SYNC_DEBUG", "0") == "1":
            import warnings as _k3w
            _k3w.filterwarnings("default", message=".*synchroniz.*")
            torch.cuda.set_sync_debug_mode("warn")
        if _K3_HF_VALIDATE:
            _K3_HF_ERR.setdefault(str(device), torch.zeros(12, device=device, dtype=torch.float32))
        import re as _re
        _ids = [int(m.group(1)) for n in layer_names for m in [_re.search(r"layers\.(\d+)\.", n)] if m]
        _is_target = bool(_ids) and max(_ids) < vllm_config.model_config.hf_text_config.num_hidden_layers
        if _K3_SMALLQ > 1 and _is_target and (vllm_config.speculative_config is None or _K3_SMALLQ_SPEC):
            self._k3_smallq = _K3_SMALLQ
            self._k3_base_threshold = self.reorder_batch_threshold
            self.query_len_support = QueryLenSupport.VARLEN
            self.reorder_batch_threshold = self._k3_smallq
            from vllm.distributed.parallel_state import get_dcp_group
            self._k3_dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
            self._k3_heads = vllm_config.model_config.hf_text_config.num_attention_heads
            self._k3_warm_pieces(device)


    def _k3_warm_pieces(self, device) -> None:
        """Compile the piece kernels (q_len = 2^k <= T/2 .. and 1) up front."""
        import os as _os
        if _os.environ.get("VLLM_K3_TS_VAR_SPLIT", "0") == "1":
            from vllm._k3_ts_mla_decode import tokenspeed_mla_decode
        else:
            from tokenspeed_mla import tokenspeed_mla_decode
        H, D, L = self._k3_heads, 576, 512
        page = self.page_size
        cache = torch.zeros(8, page, D, device=device, dtype=torch.float8_e4m3fn)
        ws = _get_workspace(device, H, L)
        W = max(1, self.dcp_world_size)
        n = 1
        while n <= self._k3_smallq:
            q = torch.zeros(1, n, H, D, device=device, dtype=torch.float8_e4m3fn)
            bt = torch.arange(8, device=device, dtype=torch.int32).view(1, 8)
            glob = torch.tensor([2 * page * W], device=device, dtype=torch.int32)
            loc = torch.tensor([2 * page], device=device, dtype=torch.int32)
            tokenspeed_mla_decode(query=q, kv_cache=cache, workspace_buffer=ws, kv_lora_rank=L,
                                  qk_rope_head_dim=64, block_tables=bt, seq_lens=loc, max_seq_len=4 * page,
                                  softmax_scale=0.1, output_scale=1.0, enable_pdl=False, return_lse=True,
                                  causal_mask=True, causal_seqs=glob if W > 1 else None,
                                  cp_world=W, cp_rank=self._k3_dcp_rank)
            n *= 2
        torch.cuda.synchronize(device)

    def _k3_local_len(self, glob: torch.Tensor) -> torch.Tensor:
        """DCP-local key count for a global causal bound."""
        if self.dcp_world_size <= 1:
            return glob.to(torch.int32)
        from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
        return get_dcp_local_seq_lens(glob, self.dcp_world_size, self._k3_dcp_rank,
                                      self.cp_kv_cache_interleave_size).to(torch.int32)

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        if _K3_HF_VALIDATE and _K3_HF_ERR:
            self._k3_hfv_n += 1
            if self._k3_hfv_n % 400 == 0:
                _v = next(iter(_K3_HF_ERR.values())).tolist()
                logger.warning("K3_HF_VALIDATE n=%d rel_max=%.3e rel_mean_max=%.3e rel_mean_avg=%.3e lse_max=%.3e "
                               "pos_rel_mean_max=%s", int(_v[8]), _v[0], _v[1], _v[9] / max(_v[8], 1.0), _v[2],
                               ",".join("%.2e" % x for x in _v[3:8]) + " nonfinite_prod=%d finite_mismatch=%d" % (int(_v[10]), int(_v[11])))
        if self._k3_smallq:
            m = common_attn_metadata
            qsl = m.query_start_loc_cpu
            ql = (qsl[1:] - qsl[:-1]).tolist()
            ctx = m.seq_lens_cpu_upper_bound
            thr = self._k3_smallq
            base = self._k3_base_threshold
            if ctx is not None:
                ctxl = ctx.tolist()
                for i, qq in enumerate(ql):
                    if base < qq <= thr and ctxl[i] - qq < _K3_SMALLQ_MIN_CTX:
                        thr = min(thr, qq - 1)
            else:
                thr = base
            self.reorder_batch_threshold = max(base, thr)
            self._k3_step_q = ql
        return super().build(common_prefix_len, common_attn_metadata, fast_build=fast_build)

    def _build_decode(self, block_table_tensor, seq_lens_device, max_seq_len, query_start_loc_cpu,
                      query_start_loc_device, num_decode_tokens, max_query_len, dcp_tot_seq_lens_device):
        md = super()._build_decode(block_table_tensor, seq_lens_device, max_seq_len, query_start_loc_cpu,
                                   query_start_loc_device, num_decode_tokens, max_query_len,
                                   dcp_tot_seq_lens_device)
        if not self._k3_smallq:
            return md
        qsl = query_start_loc_cpu.tolist()
        nd = len(qsl) - 1
        ql = [qsl[i + 1] - qsl[i] for i in range(nd)]
        if nd == 0 or (all(x == ql[0] for x in ql) and ql[0] in (1, self._k3_base_threshold)):  # k3_smqfix
            return md  # uniform decode / verify: untouched path
        glob_all = dcp_tot_seq_lens_device if dcp_tot_seq_lens_device is not None else seq_lens_device
        uniform_q = {1, self._k3_base_threshold}
        calls = []  # (tok_off, B, q_len, block_table, local_lens, global_bounds)
        i = 0
        while i < nd:
            j = i
            while j + 1 < nd and ql[j + 1] == ql[i]:
                j += 1
            q = ql[i]
            if q in uniform_q:
                calls.append((qsl[i], j - i + 1, q, q, block_table_tensor[i:j + 1], seq_lens_device[i:j + 1],
                              glob_all[i:j + 1]))
            elif _K3_SMALLQ_PAD:
                P = 1 << (q - 1).bit_length()
                for r in range(i, j + 1):
                    calls.append((qsl[r], 1, q, P, block_table_tensor[r:r + 1], seq_lens_device[r:r + 1],
                                  glob_all[r:r + 1]))
            else:
                for r in range(i, j + 1):
                    for a, n in _k3_pieces(q):
                        glob = glob_all[r:r + 1] - (q - a - n)
                        calls.append((qsl[r] + a, 1, n, n, block_table_tensor[r:r + 1], self._k3_local_len(glob),
                                      glob.to(torch.int32)))
            i = j + 1
        md._k3_pieces = calls
        return md

class TokenspeedMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [32, 64]

    @staticmethod
    def get_name() -> str:
        return "TOKENSPEED_MLA"

    @staticmethod
    def get_impl_cls() -> type["TokenspeedMLAImpl"]:
        return TokenspeedMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TokenspeedMLAMetadataBuilder"]:
        return TokenspeedMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # Surface a clear install hint up front rather than letting a raw
        # ModuleNotFoundError fire deep inside `forward_mqa` at first request.
        try:
            import tokenspeed_mla  # noqa: F401
        except ImportError:
            return (
                "tokenspeed_mla package is not installed. "
                "Install it with: `uv pip install tokenspeed-mla`"
            )

        # tokenspeed_mla CuTe DSL kernel is shape-specialized for DeepSeek R1
        # MLA dimensions (qk_nope=128, qk_rope=64, v=128). Reject anything else.
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 0)
            qk_rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", 0)
            v_head_dim = getattr(hf_text_config, "v_head_dim", 0)
            if qk_nope_head_dim != 128 or qk_rope_head_dim != 64 or v_head_dim != 128:
                return (
                    "tokenspeed_mla requires DeepSeek R1 MLA dimensions "
                    "(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128), "
                    f"got ({qk_nope_head_dim}, {qk_rope_head_dim}, {v_head_dim})"
                )
        return None


class TokenspeedMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True
    # tokenspeed_mla_decode returns LSE in log2 units; its own DCP test merges
    # partial outputs with exp2(lse).
    lse_base_on_e: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TokenspeedMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TokenspeedMLAImpl"
            )

        if not is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TokenspeedMLAImpl requires an FP8 KV cache "
                "(--kv-cache-dtype fp8 or fp8_e4m3); "
                f"got kv_cache_dtype={self.kv_cache_dtype!r}."
            )

        # Allocate (or fetch the cached) workspace lazily on first forward —
        # __init__ runs before the device is necessarily set on the worker;
        # we know it for sure at forward time when we see the input tensor.
        self._workspace_buffer: torch.Tensor | None = None
        self.softmax_scale: float | None = None
        self.output_scale: float | None = None

        # Pre-JIT BF16 and FP8 prefill kernels here too — decode impl always
        # runs when tokenspeed is selected, prefill backend may not (user can
        # pair with flash_attn / trtllm). Idempotent.
        from tokenspeed_mla import warmup_compile_prefill

        for q_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            warmup_compile_prefill(
                q_dtype=q_dtype,
                d_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                d_v=self.v_head_dim,
                enable_pdl=False,
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        import os as _os
        if _os.environ.get('VLLM_K3_TS_VAR_SPLIT', '0') == '1':
            from vllm._k3_ts_mla_decode import tokenspeed_mla_decode
        else:
            from tokenspeed_mla import tokenspeed_mla_decode

        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)

        # supports_quant_query_input=True (set in MLACommonImpl) tells the
        # pipeline to concat+FP8-quantize Q upstream via _decode_concat_quant_fp8_op.
        # The kernel is shape-specialized for FP8 Q + FP8 KV, so anything else
        # here means the upstream quant didn't run and the kernel will produce
        # garbage.
        assert q.dtype == torch.float8_e4m3fn, (
            f"TokenspeedMLAImpl expected FP8 query (supports_quant_query_input=True), "
            f"got {q.dtype}. Pipeline isinstance(q, tuple)={isinstance(q, tuple)}, "
            f"q_scale={layer._q_scale_float}, k_scale={layer._k_scale_float}."
        )

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        block_tables = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        causal_seqs = attn_metadata.decode.dcp_tot_seq_lens

        # tokenspeed_mla_decode expects query shape
        # (num_decodes, q_len_per_request, num_heads, head_dim).
        if getattr(attn_metadata.decode, "_k3_pieces", None):
            pass  # small-q piece path keeps the flat [tokens, H, D] query
        elif num_decode_tokens % num_decodes != 0:
            logger.warning_once(
                """TokenspeedMLAImpl got a query of uneven length.
                This usually indicates an issue in batch reordering
                or incorrect setup in dummy_run."""
            )
            q = q.unsqueeze(1)
        else:
            q = q.view(num_decodes, -1, q.shape[-2], q.shape[-1])

        if self.softmax_scale is None:
            # FP8 KV cache is mandatory for this backend, so q_scale/k_scale
            # always apply. softmax_scale is bmm1; output_scale is bmm2 — both
            # required to recover the correct attention output from the FP8
            # KV cache (V is stored as V_real/k_scale).
            self.softmax_scale = (
                self.scale * layer._q_scale_float * layer._k_scale_float
            )
            self.output_scale = layer._k_scale_float

        if self._workspace_buffer is None:
            # Parallelism can change the runtime query head count.
            self._workspace_buffer = _get_workspace(
                q.device, q.shape[-2], self.kv_lora_rank
            )

        # vLLM kv_c_and_k_pe_cache is already (num_blocks, block_size, head_size).
        # tokenspeed_mla_decode wants 3D — pass as-is (no unsqueeze, unlike trtllm).
        return_lse = self.need_to_return_lse_for_decode
        k3_pieces = getattr(attn_metadata.decode, "_k3_pieces", None)
        if k3_pieces:  # non-uniform decode set (small-q prefills present)
            return self._k3_forward_mqa_pieces(q, kv_c_and_k_pe_cache, attn_metadata, tokenspeed_mla_decode,
                                               return_lse, k3_pieces)
        _hf_G = _K3_HF_MAP.get(q.shape[1], 0) if (_K3_HEAD_FOLD and q.dim() == 4 and attn_metadata.causal) else 0
        # Validated batch range only (B=36..64 sweep vs production path: <=6e-4 abs; small B
        # with split-KV folded batches was inexact, see mla_fold_bench2).
        if _hf_G and 36 <= q.shape[0] <= 64 and q.shape[2] % _hf_G == 0 and (q.shape[2] // _hf_G) * q.shape[1] <= 128:
            from tokenspeed_mla import tokenspeed_mla_decode as _ts_stock
            _B, _q, _H, _D = q.shape; _hg = _H // _hf_G
            if _K3_HF_FASTCOPY:
                from vllm._k3_fold_copy import fold_heads as _k3_fold_heads
                _qg = _k3_fold_heads(q, _hf_G)
            else:
                _qg = q.view(_B, _q, _hf_G, _hg, _D).permute(0, 2, 1, 3, 4).reshape(_B * _hf_G, _q, _hg, _D)
            _cs = causal_seqs.repeat_interleave(_hf_G) if (self.dcp_world_size > 1 and causal_seqs is not None) else None
            _ko = _ts_stock(
                query=(q if _K3_HF_CTRL else _qg),
                kv_cache=kv_c_and_k_pe_cache,
                workspace_buffer=self._workspace_buffer,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                block_tables=(block_tables if _K3_HF_CTRL else block_tables.repeat_interleave(_hf_G, 0)),
                seq_lens=(seq_lens if _K3_HF_CTRL else seq_lens.repeat_interleave(_hf_G)),
                max_seq_len=attn_metadata.max_seq_len,
                softmax_scale=self.softmax_scale,
                output_scale=self.output_scale,
                enable_pdl=False,
                return_lse=return_lse,
                causal_mask=attn_metadata.causal,
                causal_seqs=((causal_seqs if self.dcp_world_size > 1 else None) if _K3_HF_CTRL else _cs),
                cp_world=self.dcp_world_size,
                cp_rank=self.dcp_rank,
            )
            _L = self.kv_lora_rank
            if return_lse:
                _o, _lse = _ko
                _lse = _lse.reshape(_B * _q, _H) if _K3_HF_CTRL else _lse.view(_B, _hf_G, _q, _hg).permute(0, 2, 1, 3).reshape(_B * _q, _H)
            else:
                _o, _lse = _ko, None
            if _K3_HF_CTRL:
                _o = _o.reshape(_B * _q, _H, _L)
            elif _K3_HF_FASTCOPY:
                from vllm._k3_fold_copy import unfold_heads as _k3_unfold_heads
                _o = _k3_unfold_heads(_o, _B, _hf_G)
            else:
                _o = _o.view(_B, _hf_G, _q, _hg, _L).permute(0, 2, 1, 3, 4).reshape(_B * _q, _H, _L)
            if _K3_HF_VALIDATE:
                _o = _o.clone()
                _lse = _lse.clone() if _lse is not None else None
                _pk = tokenspeed_mla_decode(
                    query=q,
                    kv_cache=kv_c_and_k_pe_cache,
                    workspace_buffer=self._workspace_buffer,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    max_seq_len=attn_metadata.max_seq_len,
                    softmax_scale=self.softmax_scale,
                    output_scale=self.output_scale,
                    enable_pdl=False,
                    return_lse=return_lse,
                    causal_mask=attn_metadata.causal,
                    causal_seqs=causal_seqs if self.dcp_world_size > 1 else None,
                    cp_world=self.dcp_world_size,
                    cp_rank=self.dcp_rank,
                    **({'k3_memo': attn_metadata.decode.__dict__.setdefault('_k3_vs_memo', {})}
                       if _os.environ.get('VLLM_K3_TS_VS_MEMO', '0') == '1'
                       and _os.environ.get('VLLM_K3_TS_VAR_SPLIT', '0') == '1' else {}),
                )
                _po, _plse = _pk if return_lse else (_pk, None)
                _po = _po.reshape(-1, _po.shape[-2], _po.shape[-1])
                _plse = _plse.reshape(-1, _plse.shape[-1]) if _plse is not None else None
                _k3_hf_record(_o, _lse, _po, _plse, _B, _q)
            return _o, _lse
        kernel_out = tokenspeed_mla_decode(
            query=q,
            kv_cache=kv_c_and_k_pe_cache,
            workspace_buffer=self._workspace_buffer,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            softmax_scale=self.softmax_scale,
            output_scale=self.output_scale,
            enable_pdl=False,
            return_lse=return_lse,
            causal_mask=attn_metadata.causal,
            causal_seqs=causal_seqs if self.dcp_world_size > 1 else None,
            cp_world=self.dcp_world_size,
            cp_rank=self.dcp_rank,
            **({'k3_memo': attn_metadata.decode.__dict__.setdefault('_k3_vs_memo', {})}
               if _os.environ.get('VLLM_K3_TS_VS_MEMO', '0') == '1'
               and _os.environ.get('VLLM_K3_TS_VAR_SPLIT', '0') == '1' else {}),
        )
        if return_lse:
            o, lse = kernel_out
            lse = lse.view(-1, lse.shape[-1])
        else:
            o, lse = kernel_out, None

        o = o.view(-1, o.shape[-2], o.shape[-1])
        return o, lse


    def _k3_forward_mqa_pieces(self, q, kv_cache, attn_metadata, decode_fn, return_lse, calls):
        """Uniform-q runs in one call each; lone small prefills as exact pieces."""
        T, H, D = q.shape
        L = self.kv_lora_rank
        o = torch.empty(T, H, L, dtype=torch.bfloat16, device=q.device)
        lse = torch.empty(T, H, dtype=torch.float32, device=q.device) if return_lse else None
        common = dict(kv_cache=kv_cache, workspace_buffer=self._workspace_buffer, kv_lora_rank=self.kv_lora_rank,
                      qk_rope_head_dim=self.qk_rope_head_dim, max_seq_len=attn_metadata.max_seq_len,
                      softmax_scale=self.softmax_scale, output_scale=self.output_scale, enable_pdl=False,
                      return_lse=return_lse, causal_mask=attn_metadata.causal,
                      cp_world=self.dcp_world_size, cp_rank=self.dcp_rank)
        for off, B, n, P, bt, loc, glob in calls:
            cs = glob if self.dcp_world_size > 1 else None
            if P == n:
                r = decode_fn(query=q[off:off + B * n].view(B, n, H, D), block_tables=bt, seq_lens=loc,
                              causal_seqs=cs, out=o[off:off + B * n].view(B, n, H, L), **common)
                if return_lse:
                    lse[off:off + B * n] = r[1].view(B * n, H)
            else:
                qp = q.new_zeros(1, P, H, D)
                qp[0, P - n:].copy_(q[off:off + n])
                op = torch.empty(1, P, H, L, dtype=torch.bfloat16, device=q.device)
                r = decode_fn(query=qp, block_tables=bt, seq_lens=loc, causal_seqs=cs, out=op, **common)
                o[off:off + n].copy_(op[0, P - n:])
                if return_lse:
                    lse[off:off + n] = r[1].view(P, H)[P - n:]
        return o, lse
