# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental BF16 input projections over sequence-parallel tokens."""

import os

import torch
import torch.distributed._symmetric_memory as symm_mem

import vllm.envs as envs
from vllm.config import CompilationMode, VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia.low_latency_gemm import KimiK3LowLatencyLinearMethod
from vllm.platforms import current_platform

logger = init_logger(__name__)


class KimiK3SPInputProjection:
    """Reuse async TP only for measured SM100/TP8 KDA projection shapes."""

    token_shapes = (4096, 7680, 8192)
    weight_shape = (6288, 7168)
    multicast_shapes: tuple[int, ...] = ()

    def __init__(self) -> None:
        self.group_name = get_tp_group().device_group.group_name
        # Multicast needs all input rows; the pipeline needs one input shard.
        # Pin the maximum before capture; never grow it in forward.
        multicast_rows = max(
            (*self.multicast_shapes, *(m for m in self.token_shapes if m <= 2048)),
            default=0,
        )
        self.workspace_bytes = (
            max(max(self.token_shapes) // 8, multicast_rows) * 7168 * 2
        )
        self.workspace = symm_mem.get_symm_mem_workspace(
            self.group_name, self.workspace_bytes
        )

    def try_apply(
        self,
        x: torch.Tensor,
        linear: torch.nn.Module,
        num_tokens: int,
    ) -> torch.Tensor | None:
        weight = getattr(linear, "weight", None)
        if (
            type(getattr(linear, "quant_method", None))
            not in (UnquantizedLinearMethod, KimiK3LowLatencyLinearMethod)
            or getattr(linear, "bias", None) is not None
            or weight is None
            or weight.shape != self.weight_shape
            or weight.dtype != torch.bfloat16
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or weight.device != x.device
            or not weight.is_contiguous()
            or not x.is_contiguous()
            or x.ndim != 2
            or x.shape[1] != 7168
        ):
            return None
        padded_tokens = x.shape[0] * 8
        if (
            padded_tokens not in self.token_shapes
            or not 0 <= padded_tokens - num_tokens < 8
        ):
            return None
        if padded_tokens in self.multicast_shapes:
            projected = symm_mem._multimem_all_gather_matmul(
                x, [weight.t()], self.group_name
            )[0]
        else:
            projected = torch.ops.symm_mem.fused_all_gather_matmul(
                x, [weight.t()], 0, self.group_name, return_A=False
            )[1][0]
        return projected[:num_tokens]


class KimiK3ExtendedSPInputProjection(KimiK3SPInputProjection):
    token_shapes = tuple(int(_x) for _x in (__import__('os').environ.get('VLLM_K3_KDA_SP_SHAPES') or '1536,3072,4096,6144,7680,7744,7808,8192').split(','))


class KimiK3SharedMLPInputProjection(KimiK3SPInputProjection):
    token_shapes = tuple(int(_x) for _x in (__import__('os').environ.get('VLLM_K3_SMLP_SP_SHAPES') or '4096,6144,7680,7744,7808,8192').split(','))
    weight_shape = (1536, 7168)
    multicast_shapes = (4096,)


def maybe_init_kda_sp_input_projection(
    vllm_config: VllmConfig,
    use_sequence_parallel: bool,
) -> KimiK3SPInputProjection | None:
    projection_type = (
        KimiK3ExtendedSPInputProjection
        if vllm_config.additional_config.get("kda_sp_extended_shapes", False)
        else KimiK3SPInputProjection
    )
    return _maybe_init_sp_input_projection(
        vllm_config, use_sequence_parallel, "kda_sp_input_projection", projection_type
    )


def maybe_init_shared_mlp_sp_input_projection(
    vllm_config: VllmConfig,
    use_sequence_parallel: bool,
) -> KimiK3SPInputProjection | None:
    return _maybe_init_sp_input_projection(
        vllm_config,
        use_sequence_parallel,
        "shared_mlp_sp_input_projection",
        KimiK3SharedMLPInputProjection,
    )


def _maybe_init_sp_input_projection(
    vllm_config: VllmConfig,
    use_sequence_parallel: bool,
    flag: str,
    projection_type: type[KimiK3SPInputProjection],
) -> KimiK3SPInputProjection | None:
    if not vllm_config.additional_config.get(flag, False):
        return None
    parallel_config = vllm_config.parallel_config
    if (
        not use_sequence_parallel
        or parallel_config.tensor_parallel_size != 8
        or parallel_config.use_ubatching
        or vllm_config.model_config.dtype != torch.bfloat16
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability((10, 0))
        or envs.VLLM_BATCH_INVARIANT
        or "TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP" in os.environ
        or vllm_config.compilation_config.pass_config.fuse_gemm_comms
        # K3 defaults to breakable CUDA graphs. Dynamic Dynamo fallback through
        # the existing custom AG is not supported by this experimental path.
        or vllm_config.compilation_config.mode != CompilationMode.NONE
    ):
        logger.warning_once(
            "K3 SP input projection requires SM100, BF16, TP8 model-level SP, "
            "no ubatching/batch invariance, eager or breakable CUDA graphs, "
            "and no other async TP dispatch."
        )
        return None
    tp = get_tp_group()
    communicator = tp.device_communicator
    ca = communicator.ca_comm if communicator is not None else None
    if ca is None or ca.disabled or not ca.mnnvl_multicast_ptr:
        logger.warning_once("KDA SP input projection requires a TP NVLink domain.")
        return None
    if projection_type.multicast_shapes and not hasattr(
        symm_mem, "_multimem_all_gather_matmul"
    ):
        return None
    projection = projection_type()
    logger.info_once(
        "Experimental %s enabled for %s; group workspace requirement %d bytes.",
        flag,
        projection.token_shapes,
        projection.workspace_bytes,
    )
    return projection
