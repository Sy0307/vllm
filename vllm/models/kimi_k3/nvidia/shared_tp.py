"""Isolated TP4 shared-expert experiment; attention and routed EP remain TP8."""

import os

import torch
import torch.distributed._symmetric_memory as symm_mem

import vllm.envs as envs
from vllm.distributed.parallel_state import init_k3_shared_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia.low_latency_gemm import KimiK3LowLatencyLinearMethod

logger = init_logger(__name__)
_resources = None


class SharedTPInputProjection:
    def __init__(self, group, max_m):
        self.group = group
        self.group_name = group.device_group.group_name
        global_shapes = os.environ.get(
            "VLLM_K3_SMLP_SP_MC_SHAPES", "1280,2560,4096,9216"
        )
        self.shapes = {((int(m) + 7) // 8) * 4 for m in global_shapes.split(",") if m}
        symm_mem.enable_symm_mem_for_group(self.group_name)
        self.workspace = symm_mem.get_symm_mem_workspace(
            self.group_name, max_m * 7168 * 2
        )

    def try_apply(self, x, linear, num_tokens):
        weight = linear.weight
        if (
            num_tokens not in self.shapes
            or num_tokens != x.shape[0] * self.group.world_size
            or type(linear.quant_method)
            not in (UnquantizedLinearMethod, KimiK3LowLatencyLinearMethod)
            or weight.shape != (3072, 7168)
            or weight.dtype != torch.bfloat16
            or x.dtype != torch.bfloat16
            or linear.bias is not None
            or not weight.is_contiguous()
            or not x.is_contiguous()
        ):
            return None
        return symm_mem._multimem_all_gather_matmul(x, [weight.t()], self.group_name)[0]


def maybe_init_shared_tp(config, use_sp, run_gemm_rs):
    global _resources
    size = int(os.environ.get("VLLM_K3_SHARED_TP_SIZE", "8"))
    if size == 8:
        return
    assert size == 4
    pc = config.parallel_config
    assert use_sp and envs.VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT
    assert pc.tensor_parallel_size == 8 and pc.data_parallel_size == 1
    assert not pc.use_ubatching and not envs.VLLM_BATCH_INVARIANT
    assert config.model_config.dtype == torch.bfloat16
    if _resources is not None:
        return
    group = init_k3_shared_tp_group(size)
    comm = group.device_communicator
    assert comm is not None and comm.ca_comm is not None
    assert not comm.ca_comm.disabled and comm.ca_comm.mnnvl_multicast_ptr
    max_m = ((config.scheduler_config.max_num_batched_tokens + 7) // 8) * size
    projection = None
    if config.additional_config.get("shared_mlp_sp_input_projection", False):
        projection = SharedTPInputProjection(group, max_m)
    gemm = None
    if run_gemm_rs:
        from vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar import GemmRsAr

        gemm = GemmRsAr(max_M=max(128, max_m), N=7168, tp_group=group)
    _resources = (group, projection, gemm)
    logger.info(
        "K3 shared TP4 initialized: ranks=%s, max group tokens=%d", group.ranks, max_m
    )


def get_shared_tp_resources():
    return _resources


def shared_collective(group, name, x):
    comm = group.device_communicator
    collective = getattr(comm, f"custom_{name}", None)
    output = None if collective is None else collective(x)
    if output is not None:
        return output
    return getattr(group, name)(x, dim=0)
