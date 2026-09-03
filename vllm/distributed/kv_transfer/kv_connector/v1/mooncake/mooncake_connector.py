# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TransferTopology,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.dspark_context_transport import (
    DSPARK_CONTEXT_REGION_KIND,
    dspark_context_kv_transport_enabled,
)
from vllm.distributed.kv_transfer.kv_connector.v1.kda_recoverssm_transport import (
    KDA_BASE_RECURRENT_REGION,
    KDA_TARGET_CONV_REGION,
    KDATargetStateTransport,
    kda_target_state_transport_enabled,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.utils.torch_utils import is_non_overlapping_and_dense
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import select_common_block_size

logger = init_logger(__name__)

try:
    from mooncake.engine import TransferEngine
except ImportError:
    logger.warning(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
        "to run VLLM with MooncakeTransferEngine."
    )
    TransferEngine = None

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)

# Mooncake can reject one very large batch even when every registered source and
# destination region is valid.  Keep Direct P->D submissions bounded in both
# dimensions: descriptor count limits control engine bookkeeping while the byte
# limit also handles one coalesced descriptor spanning many contiguous blocks.
DEFAULT_MAX_TRANSFER_BATCH_DESCRIPTORS = 128
DEFAULT_MAX_TRANSFER_BATCH_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class TransferRegion:
    layer_name: str
    layer_index: int
    base_addr: int
    block_len: int
    kv_block_len: int
    group_index: int = 0
    kind: str = "page"


def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    """Return the TP ratio used by heterogeneous TP transfer planning.

    Positive values mean one local rank maps into a larger remote KV region.
    Negative values mean one local rank must gather from multiple remote KV
    regions.
    """
    if local_tp_size >= remote_tp_size:
        assert local_tp_size % remote_tp_size == 0, (
            f"Local tensor parallel size {local_tp_size} is not divisible "
            f"by remote tensor parallel size {remote_tp_size}."
        )
        return local_tp_size // remote_tp_size

    assert remote_tp_size % local_tp_size == 0, (
        f"Remote tensor parallel size {remote_tp_size} is not divisible "
        f"by local tensor parallel size {local_tp_size}."
    )
    return -(remote_tp_size // local_tp_size)


def _expand_transfer_regions(
    base_addrs: list[int],
    block_lens: list[int],
    kv_block_lens: list[int],
    layer_names: list[str],
    layer_indices: list[int],
    group_indices: list[int] | None = None,
    region_kinds: list[str] | None = None,
) -> list[TransferRegion]:
    """Expand registered KV tensors into the regions transferred by Mooncake."""
    assert (
        len(base_addrs)
        == len(block_lens)
        == len(kv_block_lens)
        == len(layer_names)
        == len(layer_indices)
    ), (
        "Mooncake transfer regions require matching metadata lengths, got "
        f"base_addrs={len(base_addrs)}, block_lens={len(block_lens)}, "
        f"kv_block_lens={len(kv_block_lens)}, "
        f"layer_names={len(layer_names)}, "
        f"layer_indices={len(layer_indices)}."
    )
    if group_indices is None:
        group_indices = [0] * len(layer_names)
    assert len(group_indices) == len(layer_names), (
        "Mooncake transfer regions require matching group metadata lengths, "
        f"got group_indices={len(group_indices)}, layer_names={len(layer_names)}."
    )
    if region_kinds is None:
        region_kinds = ["page"] * len(layer_names)
    assert len(region_kinds) == len(layer_names), (
        "Mooncake transfer regions require matching kind metadata lengths, "
        f"got region_kinds={len(region_kinds)}, layer_names={len(layer_names)}."
    )
    regions: list[TransferRegion] = []
    for (
        base_addr,
        block_len,
        kv_block_len,
        layer_name,
        layer_index,
        group_index,
        region_kind,
    ) in zip(
        base_addrs,
        block_lens,
        kv_block_lens,
        layer_names,
        layer_indices,
        group_indices,
        region_kinds,
    ):
        regions.append(
            TransferRegion(
                layer_name=layer_name,
                layer_index=layer_index,
                base_addr=base_addr,
                block_len=block_len,
                kv_block_len=kv_block_len,
                group_index=group_index,
                kind=region_kind,
            )
        )
    return regions


def _compute_sender_transfer_plan(
    local_tp_rank: int,
    local_tp_size: int,
    remote_tp_rank: int,
    remote_tp_size: int,
    local_kv_block_len: int,
    remote_kv_block_len: int,
    producer_cache_replicated: bool,
) -> tuple[bool, int, int, int]:
    """Plan one producer-rank to one consumer-rank copy for heterogeneous TP."""
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)

    if tp_ratio == 1:
        return True, 0, 0, local_kv_block_len

    if tp_ratio > 0:
        if producer_cache_replicated:
            return local_tp_rank % tp_ratio == 0, 0, 0, local_kv_block_len
        return (
            True,
            0,
            (local_tp_rank % tp_ratio) * local_kv_block_len,
            local_kv_block_len,
        )

    if producer_cache_replicated:
        return True, 0, 0, local_kv_block_len

    ratio_abs = -tp_ratio
    return (
        True,
        (remote_tp_rank % ratio_abs) * remote_kv_block_len,
        0,
        remote_kv_block_len,
    )


def _can_coalesce_block_transfers(
    local_region_block_len: int,
    remote_region_block_len: int,
    src_region_offset: int,
    dst_region_offset: int,
    transfer_len: int,
) -> bool:
    """Whether a contiguous block group can be emitted as one larger copy."""
    return (
        src_region_offset == 0
        and dst_region_offset == 0
        and transfer_len == local_region_block_len
        and transfer_len == remote_region_block_len
    )


def _chunk_transfer_descriptors(
    src_ptrs: list[int],
    dst_ptrs: list[int],
    lengths: list[int],
    *,
    max_descriptors: int,
    max_bytes: int,
) -> list[tuple[list[int], list[int], list[int]]]:
    """Split a transfer plan into bounded, address-contiguous batches.

    An individual coalesced descriptor may itself exceed ``max_bytes``.  In
    that case it is split into adjacent source/destination slices so every
    engine call still respects both limits.
    """
    if not (len(src_ptrs) == len(dst_ptrs) == len(lengths)):
        raise ValueError("Mooncake transfer descriptor arrays must have equal length")
    if max_descriptors <= 0 or max_bytes <= 0:
        raise ValueError("Mooncake transfer batch limits must be positive")

    batches: list[tuple[list[int], list[int], list[int]]] = []
    batch_src: list[int] = []
    batch_dst: list[int] = []
    batch_lengths: list[int] = []
    batch_bytes = 0

    def flush() -> None:
        nonlocal batch_src, batch_dst, batch_lengths, batch_bytes
        if batch_src:
            batches.append((batch_src, batch_dst, batch_lengths))
            batch_src = []
            batch_dst = []
            batch_lengths = []
            batch_bytes = 0

    for src_ptr, dst_ptr, length in zip(src_ptrs, dst_ptrs, lengths):
        if length <= 0:
            raise ValueError("Mooncake transfer descriptor lengths must be positive")
        offset = 0
        while offset < length:
            if len(batch_src) == max_descriptors or batch_bytes == max_bytes:
                flush()
            part_len = min(length - offset, max_bytes - batch_bytes)
            batch_src.append(src_ptr + offset)
            batch_dst.append(dst_ptr + offset)
            batch_lengths.append(part_len)
            batch_bytes += part_len
            offset += part_len

    flush()
    return batches


def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    """Validate transfer-region metadata for a fixed producer/consumer pair.

    This checks registered KV regions, not per-request block counts. A region
    corresponds to one registered KV tensor, or one K/V half after expansion
    for layouts that store K and V together.
    """
    if len(local_regions) != len(remote_regions):
        return (
            "Mooncake asymmetric TP requires matching KV region counts between "
            "producer and consumer."
        )

    if producer_cache_replicated:
        return None

    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if tp_ratio == 1:
            if local_region.kv_block_len != remote_region.kv_block_len:
                return (
                    "Mooncake KV region length mismatch for homogeneous TP at "
                    f"region {idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}."
                )
        elif tp_ratio > 0:
            if remote_region.kv_block_len != local_region.kv_block_len * tp_ratio:
                return (
                    "Mooncake destination KV region length does not match the "
                    "producer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
        else:
            ratio_abs = -tp_ratio
            if local_region.kv_block_len != remote_region.kv_block_len * ratio_abs:
                return (
                    "Mooncake source KV region length does not match the "
                    "consumer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )

    return None


def _align_transfer_regions(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    *,
    require_exact: bool = False,
) -> tuple[list[TransferRegion], list[TransferRegion], str | None]:
    """Align KV transfer regions by layer, semantic kind, and occurrence.

    PP shards own different layer subsets. Positional matching is therefore
    wrong once producer and consumer have different PP layouts. Multiple
    registered transfer buffers for the same layer can carry distinct semantic
    fields; repeated fields of one kind are matched by occurrence order.
    """

    def keyed_regions(
        regions: list[TransferRegion],
    ) -> list[tuple[tuple[str, str, int], TransferRegion]]:
        counts: dict[tuple[str, str], int] = defaultdict(int)
        keyed: list[tuple[tuple[str, str, int], TransferRegion]] = []
        for region in regions:
            region_key = (region.layer_name, region.kind)
            occurrence = counts[region_key]
            counts[region_key] += 1
            keyed.append(((*region_key, occurrence), region))
        return keyed

    local_keyed = keyed_regions(local_regions)
    remote_keyed = keyed_regions(remote_regions)
    remote_by_key = dict(remote_keyed)
    aligned_local: list[TransferRegion] = []
    aligned_remote: list[TransferRegion] = []
    for key, local_region in local_keyed:
        remote_region = remote_by_key.get(key)
        if remote_region is None:
            return (
                [],
                [],
                (
                    "Mooncake producer registered layer has no matching "
                    f"consumer region: {key[0]} kind {key[1]} "
                    f"occurrence {key[2]}."
                ),
            )
        if local_region.layer_index != remote_region.layer_index:
            return (
                [],
                [],
                (
                    "Mooncake registered layer index mismatch for "
                    f"{local_region.layer_name}: producer="
                    f"{local_region.layer_index}, consumer="
                    f"{remote_region.layer_index}."
                ),
            )
        aligned_local.append(local_region)
        aligned_remote.append(remote_region)

    if require_exact:
        local_keys = {key for key, _ in local_keyed}
        extra_remote_keys = sorted(set(remote_by_key) - local_keys)
        if extra_remote_keys:
            layer_name, kind, occurrence = extra_remote_keys[0]
            return (
                [],
                [],
                (
                    "Mooncake consumer registered an unmatched region while "
                    "DSpark context transport requires an exact manifest: "
                    f"{layer_name} kind {kind} occurrence {occurrence}."
                ),
            )

    return aligned_local, aligned_remote, None


def _get_tensor_dense_flag(tensor: torch.Tensor) -> bool | None:
    is_dense = getattr(tensor, "is_non_overlapping_and_dense", None)
    if callable(is_dense):
        return bool(is_dense())
    return None


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[list[int]]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    kv_block_lens: list[int]
    registered_layer_names: list[str] = msgspec.field(default_factory=list)
    registered_layer_indices: list[int] = msgspec.field(default_factory=list)
    registered_group_indices: list[int] = msgspec.field(default_factory=list)
    registered_region_kinds: list[str] = msgspec.field(default_factory=list)
    dspark_context_transport: bool = False


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None
    # Exact consumer-side KDA blocks written by this producer worker.  Direct
    # target-state transport can tail-align one producer live state to the last
    # of multiple consumer slots, so the consumer must not materialize every
    # locally allocated slot (notably its untouched checkpoint slot).
    kda_materialized_block_ids: dict[ReqId, dict[int, list[int]]] | None = None


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[list[int]]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    # Set expire time to avoid infinitely sending requests.
    expire_time: float = float("inf")
    # Designed for one D pairing to multiple P
    pull_tasks_count: int = 0
    # Accumulated across all producer workers paired with this consumer worker.
    kda_materialized_block_ids: dict[int, set[int]] = field(default_factory=dict)


@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[list[int]]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    # Number of paired consumer attempts that reached a terminal response.
    # Both success and failure count: once the engine call has returned, that
    # peer can no longer access the producer buffers and must not pin them.
    completed: int = 0
    sending: int = 0


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # Use (engine_id, dp_rank) to group reqs with same dp.
        # See comments in MooncakeBootstrapServer.
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[list[int]]]] = {}
        self.reqs_not_processed: set[TransferId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        transfer_id = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id = kv_transfer_params["remote_engine_id"]
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_id=transfer_id,
            )
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)


class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    @classmethod
    def supports_kda_recoverssm_transport(cls, extra_config: dict[str, Any]) -> bool:
        return kda_target_state_transport_enabled(extra_config)

    @classmethod
    def supports_dspark_context_transport(cls, extra_config: dict[str, Any]) -> bool:
        return dspark_context_kv_transport_enabled(extra_config)

    @property
    def requires_block_zeroing_before_async_load(self) -> bool:
        dcp_size = self._vllm_config.parallel_config.decode_context_parallel_size or 1
        return (
            self._kv_transfer_config.is_kv_consumer
            and dcp_size > 1
            and self._kv_cache_config.has_mamba_layers
        )

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            assert kv_cache_config is not None, (
                "kv_cache_config is required for SCHEDULER role"
            )
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            # This fallback mostly exists for unit tests that instantiate the
            # connector without a fully populated model config.
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default kv cache layout."
            )
            return None
        if vllm_config.model_config.use_mla:
            return None
        logger.info_once(
            "MooncakeConnector setting KV cache layout to LBHNC for "
            "heterogeneous TP-safe KV transfer."
        )
        return "LBHNC"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def on_new_request(self, request: "Request") -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        pass

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return worker-local transfer stats since the last call.

        Note the P/D asymmetry: because Mooncake is P-push (P calls
        batch_transfer_sync_write), P records successful transfer latency,
        bytes, and descriptor counts, while D only records failures
        (recv/ZMQ errors). Aggregated NIXL-style dashboards will find
        successful-transfer metrics on the P worker, not D.
        """
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return MooncakeKVConnectorStats(data=data or {})


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        assert vllm_config.kv_transfer_config
        self.is_kv_producer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        # GDN is represented as a MambaSpec in vLLM. This Mooncake MambaSpec
        # path is currently tested with GDN; Mamba2 is not validated yet.
        self._has_mamba = kv_cache_config.has_mamba_layers
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._dspark_context_transport_enabled = dspark_context_kv_transport_enabled(
            extra_config
        )
        spec_config = vllm_config.speculative_config
        if self._dspark_context_transport_enabled and (
            spec_config is None
            or not spec_config.use_dspark()
            or not kda_target_state_transport_enabled(extra_config)
        ):
            raise ValueError(
                "Mooncake DSpark context transport requires DSpark and "
                "kda_transport_policy=target_state_v1."
            )

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        self._reqs_need_send: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[TransferId] = set()
        # A direct-truncated producer's final h(N-1) state can live in the
        # partial-tail handoff block rather than the ordinary request block
        # table.  Keep the exact per-group source until request_finished()
        # publishes the Direct send metadata.  Store consumes the same
        # scheduler handoff, so both transports must DMA from this block.
        self._partial_tail_send_overrides: dict[
            ReqId, dict[int, list[tuple[int, int]]]
        ] = {}
        self._mamba_group_indices = {
            group_index
            for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        }

        # Compute sliding window block counts per KV cache group.
        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        # cdiv(n_tokens, block_size) gives blocks/window; add 1 to
        # conservatively account for boundary overlap.
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

    def get_sw_clipped_blocks(
        self,
        block_ids: tuple[list[int], ...] | list[list[int]],
    ) -> list[list[int]]:
        """Clip per-group block IDs to sliding window size."""
        if len(block_ids) == 0 or not self._is_hma_required:
            return list(block_ids)
        return [
            blocks[-self.blocks_per_sw[i] :] if self.blocks_per_sw[i] > 0 else blocks
            for i, blocks in enumerate(block_ids)
        ]

    def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba models since the decoder
        always recomputes the last token and must start from h(N-1)."""
        if self._has_mamba and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_mamba_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-1) instead of h(N). The decoder recomputes the last token to
        derive h(N) correctly.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        params = request.kv_transfer_params
        if (
            params is not None
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > 1
        ):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return

            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def on_new_request(self, request: "Request") -> None:
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_decode") and self._has_mamba:
            self._truncate_mamba_request_for_prefill(request)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if not params:
            return 0, False

        if params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert not self.is_kv_producer
            token_ids = request.prompt_token_ids or []
            count = self._get_remote_prefill_token_count(len(token_ids)) - (
                num_computed_tokens
            )
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "req_id=%s num_external_tokens=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            spec_config = self.vllm_config.speculative_config
            if (
                num_external_tokens > 0
                and spec_config is not None
                and spec_config.use_dspark()
                and not self._dspark_context_transport_enabled
            ):
                request.disable_speculative_decoding = True
                logger.warning_once(
                    "Disabling DSpark for requests with a Mooncake-loaded "
                    "prefix because Direct transport does not bootstrap the "
                    "DSpark context cache. Target-model decoding remains enabled."
                )
            if all(
                p in params
                for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")
            ):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                unhashed_block_ids = (
                    blocks.get_unhashed_block_ids_all_groups()
                    if num_external_tokens > 0
                    else ()
                )
                local_block_ids = self.get_sw_clipped_blocks(unhashed_block_ids)
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            assert not self.is_kv_consumer
            if not params.get("transfer_id"):
                logger.warning("Missing transfer_id in kv_transfer_params from router!")
            else:
                # Add an empty list to worker to create event.
                self._reqs_need_send[request.request_id] = (request, [])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        for req_id in scheduler_output.preempted_req_ids or ():
            self._partial_tail_send_overrides.pop(req_id, None)

        step_partial_tails = getattr(scheduler_output, "partial_tail_offloads", None)
        if step_partial_tails and self.is_kv_producer:
            for req_id, entries in step_partial_tails.items():
                overrides = self._partial_tail_send_overrides.setdefault(req_id, {})
                for group_index, block_id, boundary_tokens in entries:
                    if group_index in self._mamba_group_indices:
                        candidate = (block_id, boundary_tokens)
                        group_candidates = overrides.setdefault(group_index, [])
                        if candidate not in group_candidates:
                            group_candidates.append(candidate)

        # Loop through scheduled reqs and convert to PullReqMeta.
        if not self.is_kv_producer:
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if not self.is_kv_consumer:
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                assert req.kv_transfer_params is not None
                if block_ids:
                    block_ids = self._apply_partial_tail_send_overrides(req, block_ids)
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()
            meta.reqs_not_processed = self._reqs_not_processed
            self._reqs_not_processed = set()

        return meta

    def _apply_partial_tail_send_overrides(
        self,
        request: "Request",
        block_ids: list[list[int]],
    ) -> list[list[int]]:
        params = request.kv_transfer_params
        if not (params and params.get("_p_side_truncated")):
            self._partial_tail_send_overrides.pop(request.request_id, None)
            return block_ids

        overrides = self._partial_tail_send_overrides.pop(request.request_id, {})
        result = [list(group) for group in block_ids]
        diagnostics: dict[int, tuple[int | None, int, bool]] = {}
        failures: list[str] = []
        for group_index in sorted(self._mamba_group_indices):
            if group_index >= len(result):
                failures.append(f"missing group {group_index}/{len(result)}")
                continue
            candidates = {
                block_id
                for block_id, boundary_tokens in overrides.get(group_index, ())
                if boundary_tokens == request.num_prompt_tokens
            }
            if len(candidates) != 1:
                available = sorted(overrides.get(group_index, ()))
                failures.append(
                    f"group {group_index} expected one boundary "
                    f"{request.num_prompt_tokens} source, got {available}"
                )
                continue
            block_id = candidates.pop()
            if block_id == NULL_BLOCK_ID or block_id < 0:
                failures.append(f"group {group_index} invalid block {block_id}")
                continue
            before = [
                value
                for value in result[group_index]
                if value != NULL_BLOCK_ID and value >= 0
            ]
            before_last = before[-1] if before else None
            result[group_index] = [block_id]
            diagnostics[group_index] = (
                before_last,
                block_id,
                before_last == block_id,
            )

        if failures:
            # Queue an intentionally empty source table. The producer worker's
            # all-groups validation converts it to err_reqs before constructing
            # any DMA descriptor, so D invalidates every destination block and
            # safely recomputes instead of consuming an inferred KDA slot.
            logger.error(
                "KDA Direct final-state descriptor failed closed: req=%s, "
                "boundary=%d, failures=%s",
                request.request_id,
                request.num_prompt_tokens,
                failures,
            )
            return [[] for _ in result]

        logger.info(
            "KDA Direct final-state descriptor: req=%s, boundary=%d, "
            "before_last_override_same=%s",
            request.request_id,
            request.num_prompt_tokens,
            diagnostics,
        )
        return result

    def request_finished(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, req_id=%s, request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params or not params.get("transfer_id"):
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert not self.is_kv_producer
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None

        assert not self.is_kv_consumer

        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(params["transfer_id"])
            self._partial_tail_send_overrides.pop(request.request_id, None)
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = any(len(group) > 0 for group in block_ids)

        if delay_free_blocks:
            clipped_blocks = self.get_sw_clipped_blocks(block_ids)
            self._reqs_need_send[request.request_id] = (
                request,
                clipped_blocks,
            )

        return delay_free_blocks, None


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        if TransferEngine is None:
            logger.error("Mooncake is not available")
            raise RuntimeError("Mooncake is not available")
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config
        # Capture device BEFORE TransferEngine init — MNNVL's NVLink allocator
        # may change the current CUDA device during engine.initialize().
        self.device_id = torch.accelerator.current_device_index()
        current_platform.set_device(self.device_id)

        self.engine = TransferEngine()
        self.hostname = get_ip()

        assert (kv_transfer_config := vllm_config.kv_transfer_config)
        self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
        self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )
        self.max_transfer_batch_descriptors = int(
            kv_transfer_config.kv_connector_extra_config.get(
                "max_transfer_batch_descriptors",
                DEFAULT_MAX_TRANSFER_BATCH_DESCRIPTORS,
            )
        )
        self.max_transfer_batch_bytes = int(
            kv_transfer_config.kv_connector_extra_config.get(
                "max_transfer_batch_bytes", DEFAULT_MAX_TRANSFER_BATCH_BYTES
            )
        )
        logger.info(
            "Mooncake Direct producer-ready timeout is %d seconds.",
            envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
        )
        if (
            self.max_transfer_batch_descriptors <= 0
            or self.max_transfer_batch_bytes <= 0
        ):
            raise ValueError(
                "Mooncake Direct transfer batch limits must be positive, got "
                f"descriptors={self.max_transfer_batch_descriptors}, "
                f"bytes={self.max_transfer_batch_bytes}."
            )
        # Create more tasks than workers to keep the thread pool saturated.
        # Tasks can await async events, so a surplus (2x is a robust heuristic)
        # prevents workers from idling.
        self.num_sender_tasks = self.num_sender_workers * 2
        protocol = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
            "mooncake_protocol", "rdma"
        )
        device_name = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
            "device_name", ""
        )
        logger.info(
            "The Mooncake Transfer Engine is using %s as its protocol.", protocol
        )
        ret_value = self.engine.initialize(
            self.hostname, "P2PHANDSHAKE", protocol, device_name
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
        self._pending_bootstrap_queries: dict[str, asyncio.Event] = {}
        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.block_len_per_layer: list[int] = []
        self.kv_block_len_per_layer: list[int] = []
        self.registered_layer_names: list[str] = []
        self.registered_layer_indices: list[int] = []
        self.registered_group_indices: list[int] = []
        self.registered_region_kinds: list[str] = []
        self.seen_base_addresses: list[int] = []

        assert (parallel_config := vllm_config.parallel_config)
        dp_rank = parallel_config.data_parallel_index
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_rank = get_pp_group().rank_in_group

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}

        # For kv_both, we will act both prefiller and decoder.
        if not self.is_kv_consumer:
            # Background threads for sending kvcaches to D.
            # Each pool thread must be bound to the correct CUDA device
            # because CUDA device selection is thread-local.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="vllm-mooncake-sender",
                initializer=self._bind_sender_thread_device,
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches",
                self.num_sender_workers,
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

            # Start bootstrap server on global rank 0.
            if should_launch_bootstrap_server(vllm_config):
                _, port = get_mooncake_bootstrap_addr(vllm_config)
                self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", port)
                self.bootstrap_server.start()

        if not self.is_kv_producer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()
        # Direct receive failures must be surfaced through the same
        # invalid-block recovery contract used by asynchronous Store loads.
        # Both sets are owned by receiver_loop and drained on model steps.
        self.invalid_recving_block_ids: set[int] = set()

        self.xfer_stats = MooncakeKVConnectorStats()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache_config = kv_cache_config
        self._kda_transport_enabled = kda_target_state_transport_enabled(
            kv_transfer_config.kv_connector_extra_config
        )
        self._dspark_context_transport_enabled = dspark_context_kv_transport_enabled(
            kv_transfer_config.kv_connector_extra_config
        )
        spec_config = vllm_config.speculative_config
        if self._dspark_context_transport_enabled and (
            spec_config is None
            or not spec_config.use_dspark()
            or not self._kda_transport_enabled
        ):
            raise ValueError(
                "Mooncake DSpark context transport requires DSpark and "
                "kda_transport_policy=target_state_v1."
            )
        self._kda_transport: KDATargetStateTransport | None = None
        get_target_layers = getattr(
            self.vllm_config.model_config,
            "get_total_num_hidden_layers",
            None,
        )
        self.target_num_layers = (
            get_target_layers() if callable(get_target_layers) else None
        )
        self.use_mla = self.model_config.use_mla
        self._physical_blocks_per_logical_kv_block = 1
        self._sync_block_size_with_kernel()

        self.attn_backends = get_current_attn_backends(vllm_config)
        logger.debug(
            "Detected attention backends %s",
            [backend.get_name() for backend in self.attn_backends],
        )

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
        self._layer_specs: dict[str, KVCacheSpec] = {}
        for group in kv_cache_config.kv_cache_groups:
            group_spec = group.kv_cache_spec
            specs_by_layer = getattr(group_spec, "kv_cache_specs", {})
            for layer_name in group.layer_names:
                self._layer_specs[layer_name] = specs_by_layer.get(
                    layer_name, group_spec
                )
        self._layer_group_indices: dict[str, int] = {
            layer: group_index
            for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer in group.layer_names
        }
        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            is_mamba=kv_cache_config.has_mamba_layers,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
        )

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    def _sync_block_size_with_kernel(self) -> None:
        # When speculative decoding (e.g. Eagle) is enabled, the main model
        # and draft model may use different attention backends with different
        # physical block sizes. Pick the common (smallest) block size so that
        # KV-cache registration and transfer work correctly for both models.
        backends = get_current_attn_backends(self.vllm_config)
        kernel_block_size = select_common_block_size(self.block_size, backends)
        if self.block_size != kernel_block_size:
            logger.info_once(
                "User-specified logical block size (%s) does not match"
                " physical kernel block size (%s). Using the latter.",
                self.block_size,
                kernel_block_size,
            )
            assert self.block_size > kernel_block_size
            self._physical_blocks_per_logical_kv_block = (
                self.block_size // kernel_block_size
            )
            self.block_size = kernel_block_size

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads, including after partial initialization."""
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True

        async_zmq_ctx = getattr(self, "async_zmq_ctx", None)
        if async_zmq_ctx is not None:
            async_zmq_ctx.term()
        sender_executor = getattr(self, "_sender_executor", None)
        if sender_executor is not None:
            sender_executor.shutdown(wait=False)
        sender_loop = getattr(self, "sender_loop", None)
        if sender_loop is not None and sender_loop.is_running():
            sender_loop.call_soon_threadsafe(sender_loop.stop)
            sender_listener = getattr(self, "_sender_listener_t", None)
            if sender_listener is not None:
                sender_listener.join()
        bootstrap_server = getattr(self, "bootstrap_server", None)
        if bootstrap_server is not None:
            bootstrap_server.shutdown()
        receiver_loop = getattr(self, "receiver_loop", None)
        if receiver_loop is not None and receiver_loop.is_running():
            receiver_loop.call_soon_threadsafe(receiver_loop.stop)
            receiver_thread = getattr(self, "_mooncake_receiver_t", None)
            if receiver_thread is not None:
                receiver_thread.join()

    async def register_worker_with_bootstrap(self):
        host, port = get_mooncake_bootstrap_addr(self.vllm_config)
        url = make_zmq_path("http", host, port) + "/register"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            addr=worker_addr,
        )
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                )
                logger.error(
                    "Error registering %s with bootstrap server: %s", payload, err_msg
                )
                raise e

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )

        await self.register_worker_with_bootstrap()

        # Create async worker tasks that process items from the queue
        sender_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_sender_tasks)
        ]

        ready_event.set()

        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                await self.sender_worker_queue.put((identity, metadata_bytes))
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            # Clean up worker tasks
            for task in sender_tasks:
                task.cancel()
            await asyncio.gather(*sender_tasks, return_exceptions=True)
            sock.close()

    async def _sender_worker(self, sock: zmq.asyncio.Socket):
        while True:
            try:
                identity, metadata_bytes = await self.sender_worker_queue.get()
                try:
                    metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                    await self.send_kv_to_decode(identity, sock, metadata)
                except Exception as e:
                    logger.error("Error processing Mooncake xfer request: %s", e)
                    error_response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR, err_msg=str(e)
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(error_response))
                    )
                finally:
                    self.sender_worker_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in _sender_worker: %s", e)

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(meta.remote_tp_size)
        if meta.remote_tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            msg = (
                "This D tp_rank "
                f"{meta.remote_tp_rank} is not paired with P tp_rank "
                f"{self.tp_rank}; expected one of {remote_tp_ranks}."
            )
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        local_regions = self._get_transfer_regions(
            self.kv_caches_base_addr,
            self.block_len_per_layer,
            self.kv_block_len_per_layer,
            self.registered_layer_names,
            self.registered_layer_indices,
            self.registered_group_indices,
            self.registered_region_kinds,
        )
        remote_regions = self._get_transfer_regions(
            meta.kv_caches_base_addr,
            meta.block_lens,
            meta.kv_block_lens,
            meta.registered_layer_names,
            meta.registered_layer_indices,
            meta.registered_group_indices,
            meta.registered_region_kinds,
        )
        local_regions, remote_regions, align_err = _align_transfer_regions(
            local_regions,
            remote_regions,
            require_exact=self._dspark_context_transport_enabled,
        )
        if (
            align_err is None
            and meta.dspark_context_transport != self._dspark_context_transport_enabled
        ):
            align_err = (
                "Mooncake DSpark context transport capability mismatch: "
                f"producer={self._dspark_context_transport_enabled}, "
                f"consumer={meta.dspark_context_transport}."
            )
        if align_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=align_err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        validation_err = _validate_asymmetric_region_lengths(
            local_regions=local_regions,
            remote_regions=remote_regions,
            local_tp_size=self.tp_size,
            remote_tp_size=meta.remote_tp_size,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )
        if validation_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=validation_err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
                self.reqs_need_send[transfer_id] = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
            send_meta = self.reqs_need_send[transfer_id]
            pending_reqs[d_req_id] = send_meta

        async def wait_and_ret(
            d_req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Timeout, abort all pending requests.
                for task in wait_tasks:
                    task.cancel()
                await asyncio.gather(*wait_tasks, return_exceptions=True)
                # A consumer that has timed out and received a terminal error
                # can no longer access this producer's blocks.  Account for it
                # even when P has not finished its forward yet.  The request is
                # deliberately kept pinned until record_send_reqs() publishes
                # the real blocks; that late publication can then release them
                # immediately instead of waiting for a second expiry window.
                for send_meta in pending_reqs.values():
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta, remote_tp_ranks)
                    send_meta.completed += 1
                    self._try_finish_send_meta(send_meta)
                logger.warning(
                    "Timeout waiting for P side ready: %s", list(pending_reqs)
                )
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    err_reqs=list(pending_reqs),
                    err_msg="Timeout waiting for P side ready.",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                break

            wait_tasks = list(pending)
            response_status = (
                MooncakeXferResponseStatus.CONTINUE
                if wait_tasks
                else MooncakeXferResponseStatus.FINISH
            )
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done:
                d_req_id, send_meta = task.result()
                del pending_reqs[d_req_id]
                # Do we still in reqs_need_send (not expired)?
                if send_meta.transfer_id in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta, remote_tp_ranks)
                    ready_reqs.append((d_req_id, send_meta))
                else:
                    # Otherwise (expired, very unlikely), just forget it.
                    logger.warning(
                        "Request %s expired before sending on P side.", d_req_id
                    )

            try:
                (
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                    err_reqs,
                    err_msg,
                    kda_materialized_block_ids,
                ) = await self._build_transfer_params(
                    ready_reqs,
                    meta,
                    local_regions,
                    remote_regions,
                )
                err_req_set = set(err_reqs)
                ok_ready_reqs = [
                    (d_req_id, send_meta)
                    for d_req_id, send_meta in ready_reqs
                    if d_req_id not in err_req_set
                ]

                if src_ptrs:
                    remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
                    ret_value = await self.sender_loop.run_in_executor(
                        self._sender_executor,
                        self._send_blocks,
                        remote_session,
                        src_ptrs,
                        dst_ptrs,
                        lengths,
                    )

                    if ret_value != 0:
                        transfer_err_msg = (
                            f"Mooncake transfer engine returned {ret_value}"
                        )
                        err_msg = (
                            transfer_err_msg
                            if err_msg is None
                            else f"{err_msg}; {transfer_err_msg}"
                        )
                        err_reqs = list(err_reqs)
                        for d_req_id, _ in ok_ready_reqs:
                            err_reqs.append(d_req_id)
                            err_req_set.add(d_req_id)
                        ok_ready_reqs = []

                response = MooncakeXferResponse(
                    status=response_status,
                    ok_reqs=[d_req_id for d_req_id, _ in ok_ready_reqs] or None,
                    err_reqs=err_reqs or None,
                    err_msg=err_msg,
                    kda_materialized_block_ids=(
                        {
                            d_req_id: kda_materialized_block_ids[d_req_id]
                            for d_req_id, _ in ok_ready_reqs
                            if d_req_id in kda_materialized_block_ids
                        }
                        or None
                    ),
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
            finally:
                # A terminal error response is as final as a successful one for
                # producer-buffer lifetime.  The synchronous engine call has
                # returned, so this peer cannot touch the blocks again.  Count
                # every ready attempt and release only after all paired peers
                # have stopped, including if planning or response encoding raises.
                for _, send_meta in ready_reqs:
                    send_meta.sending -= 1
                    send_meta.completed += 1
                    self._try_finish_send_meta(send_meta)

    def _try_finish_send_meta(self, send_meta: SendBlockMeta) -> bool:
        """Release a producer request after every consumer is terminal.

        ``completed`` includes successful transfers, transfer failures, and
        consumers that gave up while waiting for P.  A pre-ready timeout must
        not free a running producer block, so readiness is an explicit part of
        the lifetime gate.
        """
        if (
            not send_meta.ready.is_set()
            or not send_meta.need_send
            or send_meta.completed < send_meta.need_send
            or send_meta.sending != 0
            or self.reqs_need_send.get(send_meta.transfer_id) is not send_meta
        ):
            return False
        assert send_meta.p_req_id
        del self.reqs_need_send[send_meta.transfer_id]
        self.finished_sending_reqs.add(send_meta.p_req_id)
        return True

    def resolve_need_send(
        self,
        send_meta: SendBlockMeta,
        remote_tp_ranks: list[int],
    ):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        logger.debug(
            "Mooncake request %s will be served by %d consumer TP workers: TP ranks=%s",
            send_meta.transfer_id,
            send_meta.need_send,
            remote_tp_ranks,
        )

    def _logical_to_kernel_block_ids(
        self, block_ids: list[list[int]]
    ) -> list[list[int]]:
        # For example, if a 544-token logical block is served by 32-token
        # FA kernel blocks, FA block id k expands to [17k, ..., 17k + 16],
        # while the matching Mamba/GDN state block remains k. Only attention
        # groups need logical block ids expanded to kernel block ids; Mamba/GDN
        # state block ids stay in the logical/page-id space.
        group_specs = self.kv_cache_config.kv_cache_groups
        return [
            self._logical_group_to_kernel_block_ids(
                group,
                is_mamba=isinstance(group_specs[i].kv_cache_spec, MambaSpec),
            )
            for i, group in enumerate(block_ids)
        ]

    def _logical_group_to_kernel_block_ids(
        self,
        block_ids: list[int],
        *,
        is_mamba: bool,
    ) -> list[int]:
        if not block_ids or self._physical_blocks_per_logical_kv_block == 1 or is_mamba:
            return block_ids
        block_arange = np.arange(self._physical_blocks_per_logical_kv_block).reshape(
            1, -1
        )
        return BlockTable.map_to_kernel_blocks(
            np.array(block_ids),
            self._physical_blocks_per_logical_kv_block,
            block_arange,
        ).tolist()

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
        local_regions: list[TransferRegion],
        remote_regions: list[TransferRegion],
    ) -> tuple[
        list[int],
        list[int],
        list[int],
        list[ReqId],
        str | None,
        dict[ReqId, dict[int, list[int]]],
    ]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        err_reqs: list[ReqId] = []
        err_msg: str | None = None
        kda_materialized_blocks: dict[ReqId, dict[int, set[int]]] = {}
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids_per_group = agent_meta.req_blocks[d_req_id]

            if not remote_block_ids_per_group or all(
                len(g) == 0 for g in remote_block_ids_per_group
            ):
                continue

            kda_transport = getattr(self, "_kda_transport", None)
            if kda_transport is not None:
                kda_transport.stage_groups(send_meta.local_block_ids)
                kda_transport.debug_checksums(
                    "P_DIRECT",
                    d_req_id,
                    send_meta.local_block_ids,
                )

            # Region descriptors are semantic and endpoint-neutral, while HMA
            # group indices are endpoint-local. Resolve each aligned region to
            # its own producer and consumer block table instead of requiring
            # identical group counts or numbering across P and D.
            prepared_groups: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
            has_block_error = False
            block_error_msg = "P num blocks less than D"
            group_specs = self.kv_cache_config.kv_cache_groups
            for local_region, remote_region in zip(local_regions, remote_regions):
                local_group_index = local_region.group_index
                remote_group_index = remote_region.group_index
                group_pair = (local_group_index, remote_group_index)
                if group_pair in prepared_groups:
                    continue
                if (
                    local_group_index < 0
                    or remote_group_index < 0
                    or local_group_index >= len(send_meta.local_block_ids)
                    or local_group_index >= len(group_specs)
                    or remote_group_index >= len(remote_block_ids_per_group)
                ):
                    logger.error(
                        "req %s: transfer region references a missing KV group "
                        "(producer=%d/%d, consumer=%d/%d)",
                        d_req_id,
                        local_group_index,
                        len(send_meta.local_block_ids),
                        remote_group_index,
                        len(remote_block_ids_per_group),
                    )
                    has_block_error = True
                    block_error_msg = "KV group mapping missing"
                    break

                local_group = send_meta.local_block_ids[local_group_index]
                remote_group = remote_block_ids_per_group[remote_group_index]
                is_mamba_group = isinstance(
                    group_specs[local_group_index].kv_cache_spec,
                    MambaSpec,
                )
                if is_mamba_group:
                    # Mamba/GDN prefix caching can use null blocks only as
                    # align-mode placeholders. They do not carry transferable
                    # state, so skip them on both producer and consumer sides.
                    local_group = [
                        block_id
                        for block_id in local_group
                        if block_id != NULL_BLOCK_ID
                    ]
                    remote_group = [
                        block_id
                        for block_id in remote_group
                        if block_id != NULL_BLOCK_ID
                    ]

                n_local = len(local_group)
                n_remote = len(remote_group)
                if n_local < n_remote:
                    if is_mamba_group and kda_transport is not None and n_local > 0:
                        # In align mode the consumer can own additional durable
                        # checkpoint slots (for example the canonical partial-tail
                        # checkpoint loaded by MooncakeStore) before its live state
                        # slot.  These are endpoint-local retention slots, not extra
                        # target states that Direct must obtain from the producer.
                        # The producer's live h(N-1) state therefore maps to the
                        # consumer's last live slot.  This exception is deliberately
                        # restricted to the layout-neutral target-state transport;
                        # ordinary whole-page Mamba transport keeps the strict check.
                        logger.info(
                            "req %s: KDA target-state Direct transport tail-aligns "
                            "producer blocks(%d) to consumer blocks(%d) for KV "
                            "groups producer=%d consumer=%d",
                            d_req_id,
                            n_local,
                            n_remote,
                            local_group_index,
                            remote_group_index,
                        )
                        remote_group = remote_group[-n_local:]
                        n_remote = n_local
                    else:
                        logger.error(
                            "req %s: local blocks(%d) < remote blocks(%d) "
                            "for aligned KV groups producer=%d consumer=%d "
                            "(is_mamba_group=%s)",
                            d_req_id,
                            n_local,
                            n_remote,
                            local_group_index,
                            remote_group_index,
                            is_mamba_group,
                        )
                        has_block_error = True
                        break
                elif n_local > n_remote:
                    # Partial prefix cache hit: just read uncomputed blocks.
                    local_group = local_group[-n_remote:] if n_remote > 0 else []
                prepared_groups[group_pair] = (
                    self._logical_group_to_kernel_block_ids(
                        local_group,
                        is_mamba=is_mamba_group,
                    ),
                    self._logical_group_to_kernel_block_ids(
                        remote_group,
                        is_mamba=is_mamba_group,
                    ),
                )

            if has_block_error:
                err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = block_error_msg
                continue

            if not any(local for local, _ in prepared_groups.values()):
                continue

            for local_region, remote_region in zip(local_regions, remote_regions):
                local_block_ids, remote_block_ids = prepared_groups[
                    (local_region.group_index, remote_region.group_index)
                ]
                if not local_block_ids:
                    continue

                # Group by indices within this region's KV-cache group only.
                group_local_block_ids, group_remote_block_ids = (
                    group_concurrent_contiguous(local_block_ids, remote_block_ids)
                )
                (
                    should_transfer,
                    src_region_offset,
                    dst_region_offset,
                    transfer_len,
                ) = self._get_sender_transfer_plan(
                    local_kv_block_len=local_region.kv_block_len,
                    remote_kv_block_len=remote_region.kv_block_len,
                    remote_tp_rank=agent_meta.remote_tp_rank,
                    remote_tp_size=agent_meta.remote_tp_size,
                )
                if not should_transfer:
                    # Replicated KV cache: only one producer rank in the TP group
                    # needs to send the actual bytes for this paired decoder rank.
                    # TODO: Account for replicated producer KV in
                    # get_target_remote_ranks() so we can avoid sending
                    # unnecessary ZMQ requests and remove this branch.
                    continue

                if local_region.kind in (
                    KDA_TARGET_CONV_REGION,
                    KDA_BASE_RECURRENT_REGION,
                ):
                    req_groups = kda_materialized_blocks.setdefault(d_req_id, {})
                    req_groups.setdefault(remote_region.group_index, set()).update(
                        remote_block_ids
                    )

                assert src_region_offset + transfer_len <= local_region.kv_block_len, (
                    "Computed source transfer region exceeds local KV block size."
                )
                assert dst_region_offset + transfer_len <= remote_region.kv_block_len, (
                    "Destination transfer region exceeds remote KV block size."
                )
                # `local_block_ids` / `remote_block_ids` are physical kernel-block
                # IDs at this point.  The per-block transfer plan must therefore
                # fit inside one physical stride.  A later coalescing step may
                # intentionally combine several adjacent physical pages into one
                # larger Mooncake descriptor.
                if (
                    transfer_len > local_region.block_len
                    or transfer_len > remote_region.block_len
                ):
                    raise RuntimeError(
                        "Mooncake transfer length exceeds physical block stride: "
                        f"region={local_region.layer_name}/{local_region.kind}, "
                        f"logical_block_tokens="
                        f"{self.vllm_config.cache_config.block_size}, "
                        f"kernel_block_tokens={self.block_size}, "
                        f"expansion_ratio="
                        f"{self._physical_blocks_per_logical_kv_block}, "
                        f"transfer_len={transfer_len}, "
                        f"local_physical_stride={local_region.block_len}, "
                        f"remote_physical_stride={remote_region.block_len}."
                    )
                # Collapse one contiguous block group into a single larger
                # transfer descriptor when the per-block copy is identical.
                can_coalesce = _can_coalesce_block_transfers(
                    local_region_block_len=local_region.block_len,
                    remote_region_block_len=remote_region.block_len,
                    src_region_offset=src_region_offset,
                    dst_region_offset=dst_region_offset,
                    transfer_len=transfer_len,
                )

                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    if can_coalesce:
                        descriptor_len = transfer_len * len(group_local_block_id)
                        src_ptrs.append(
                            local_region.base_addr
                            + group_local_block_id[0] * local_region.block_len
                            + src_region_offset
                        )
                        dst_ptrs.append(
                            remote_region.base_addr
                            + group_remote_block_id[0] * remote_region.block_len
                            + dst_region_offset
                        )
                        lengths.append(descriptor_len)
                        logger.debug(
                            "Mooncake transfer descriptor: region=%s/%s "
                            "logical_block_tokens=%d kernel_block_tokens=%d "
                            "expansion_ratio=%d physical_stride=%d "
                            "per_physical_transfer_len=%d physical_pages=%d "
                            "descriptor_len=%d",
                            local_region.layer_name,
                            local_region.kind,
                            self.vllm_config.cache_config.block_size,
                            self.block_size,
                            self._physical_blocks_per_logical_kv_block,
                            local_region.block_len,
                            transfer_len,
                            len(group_local_block_id),
                            descriptor_len,
                        )
                    else:
                        for local_block_id, remote_block_id in zip(
                            group_local_block_id, group_remote_block_id
                        ):
                            src_ptrs.append(
                                local_region.base_addr
                                + local_block_id * local_region.block_len
                                + src_region_offset
                            )
                            dst_ptrs.append(
                                remote_region.base_addr
                                + remote_block_id * remote_region.block_len
                                + dst_region_offset
                            )
                            lengths.append(transfer_len)

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                d_req_id,
                sum(len(local) for local, _ in prepared_groups.values()),
                remote_session,
            )

        materialized = {
            req_id: {
                group_index: sorted(block_ids)
                for group_index, block_ids in groups.items()
            }
            for req_id, groups in kda_materialized_blocks.items()
        }
        return src_ptrs, dst_ptrs, lengths, err_reqs, err_msg, materialized

    def _bind_sender_thread_device(self) -> None:
        """ThreadPoolExecutor initializer — binds each pool thread to the
        correct CUDA device.  CUDA device selection is thread-local, so
        without this, NVLink transfers fail for TP ranks > 0."""
        current_platform.set_device(self.device_id)

    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        start_time = time.perf_counter()
        batches = _chunk_transfer_descriptors(
            src_ptrs,
            dst_ptrs,
            lengths,
            max_descriptors=self.max_transfer_batch_descriptors,
            max_bytes=self.max_transfer_batch_bytes,
        )
        ret_value = 0
        for batch_index, (batch_src, batch_dst, batch_lengths) in enumerate(batches):
            try:
                ret_value = self.engine.batch_transfer_sync_write(
                    remote_session, batch_src, batch_dst, batch_lengths
                )
            except Exception:
                ret_value = -1
                logger.exception(
                    "Sending batch %d/%d to %s raised an exception "
                    "(%d descriptors, %d bytes)",
                    batch_index + 1,
                    len(batches),
                    remote_session,
                    len(batch_src),
                    sum(batch_lengths),
                )
            if ret_value != 0:
                break

        duration = time.perf_counter() - start_time
        if ret_value == 0:
            self.xfer_stats.record_transfer(
                duration_s=duration,
                total_bytes=sum(lengths),
                num_descs=len(src_ptrs),
            )
            logger.debug(
                "Sending to %s done in %d bounded batch(es), took %s",
                remote_session,
                len(batches),
                duration,
            )
        else:
            self.xfer_stats.record_failed_transfer()
            logger.warning(
                "Sending to %s failed in batch %d/%d (ret=%s) after %s "
                "(%d total descriptors, %d total bytes)",
                remote_session,
                batch_index + 1,
                len(batches),
                ret_value,
                duration,
                len(src_ptrs),
                sum(lengths),
            )
        return ret_value

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs: list[int] = []
        kv_data_lens: list[int] = []
        region_base_addresses: list[int] = []
        seen_storage_ptrs: set[int] = set()
        self.block_len_per_layer = []
        self.kv_block_len_per_layer = []
        self.registered_layer_names = []
        self.registered_layer_indices = []
        self.registered_group_indices = []
        self.registered_region_kinds = []

        if self._kda_transport_enabled:
            self._kda_transport = KDATargetStateTransport.create(
                kv_caches,
                self.kv_cache_config,
                target_num_layers=self.target_num_layers,
            )

        draft_context_regions = 0
        for layer_name, cache in kv_caches.items():
            layer_index = extract_layer_index(layer_name)
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                logger.debug(
                    "Skipping layer %s because no KV cache spec is present.",
                    layer_name,
                )
                continue
            is_draft_context = bool(
                self._dspark_context_transport_enabled
                and isinstance(layer_spec, AttentionSpec)
                and self.target_num_layers is not None
                and layer_index >= self.target_num_layers
            )
            # One raw page tensor per layer; for Mamba that page holds all the
            # recurrent states, unpacked only when binding the cache for execution.
            self._log_debug_cache_registration(layer_name, cache)
            kda_regions = (
                self._kda_transport.regions_for_layer(layer_name)
                if self._kda_transport is not None
                else ()
            )
            registration_tensors: list[torch.Tensor] = [cache]
            if kda_regions:
                for region in kda_regions:
                    region_base_addresses.append(region.tensor.data_ptr())
                    self.block_len_per_layer.append(region.block_stride_bytes)
                    self.kv_block_len_per_layer.append(region.content_len_bytes)
                    self.registered_layer_names.append(layer_name)
                    self.registered_layer_indices.append(layer_index)
                    self.registered_group_indices.append(
                        self._layer_group_indices[layer_name]
                    )
                    self.registered_region_kinds.append(region.kind)
                    registration_tensors.append(region.tensor)
            else:
                block_is_contiguous = is_non_overlapping_and_dense(cache[0])
                if not block_is_contiguous:
                    region_caches = [cache[:, head] for head in range(cache.shape[1])]
                    assert all(
                        is_non_overlapping_and_dense(region[0])
                        for region in region_caches
                    )
                else:
                    region_caches = [cache]

                for region_cache in region_caches:
                    base_addr = region_cache.data_ptr()
                    block_len = region_cache.stride(0) * region_cache.element_size()
                    region_base_addresses.append(base_addr)

                    if isinstance(layer_spec, AttentionSpec) and block_is_contiguous:
                        assert (
                            layer_spec.page_size_bytes
                            % self._physical_blocks_per_logical_kv_block
                            == 0
                        )
                        kv_block_len = (
                            layer_spec.page_size_bytes
                            // self._physical_blocks_per_logical_kv_block
                        )
                    else:
                        kv_block_len = block_len
                    if kv_block_len > block_len:
                        raise RuntimeError(
                            "Mooncake transfer length exceeds physical block stride "
                            f"for {layer_name}: kv_block_len={kv_block_len}, "
                            f"block_len={block_len}."
                        )
                    logger.info_once(
                        "Mooncake physical transfer layout: "
                        "logical_block_tokens=%d kernel_block_tokens=%d "
                        "expansion_ratio=%d layer=%s physical_block_stride=%d "
                        "descriptor_len=%d",
                        self.vllm_config.cache_config.block_size,
                        self.block_size,
                        self._physical_blocks_per_logical_kv_block,
                        layer_name,
                        block_len,
                        kv_block_len,
                    )
                    self.block_len_per_layer.append(block_len)
                    self.kv_block_len_per_layer.append(kv_block_len)
                    self.registered_layer_names.append(layer_name)
                    self.registered_layer_indices.append(layer_index)
                    self.registered_group_indices.append(
                        self._layer_group_indices[layer_name]
                    )
                    self.registered_region_kinds.append(
                        DSPARK_CONTEXT_REGION_KIND if is_draft_context else "page"
                    )
                    draft_context_regions += int(is_draft_context)

            for tensor in registration_tensors:
                storage = tensor.untyped_storage()
                storage_addr = storage.data_ptr()
                if storage_addr not in seen_storage_ptrs:
                    seen_storage_ptrs.add(storage_addr)
                    kv_data_ptrs.append(storage_addr)
                    kv_data_lens.append(storage.nbytes())

        self.kv_caches_base_addr = region_base_addresses
        self.seen_base_addresses = kv_data_ptrs

        if not kv_data_ptrs:
            raise RuntimeError("No KV cache tensors were registered with Mooncake.")
        if self._dspark_context_transport_enabled:
            logger.info(
                "Mooncake Direct PP rank %d registered %d DSpark context KV "
                "region(s) with policy=%s",
                self.pp_rank,
                draft_context_regions,
                DSPARK_CONTEXT_REGION_KIND,
            )

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        self.device_kv_caches = kv_caches
        logger.debug(
            "registered block_lens=%s kv_block_lens=%s",
            self.block_len_per_layer,
            self.kv_block_len_per_layer,
        )

        # No need to launch server for D node.
        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
        )
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        finished_recving_reqs = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        return finished_recving_reqs

    async def fetch_invalid_recving_block_ids(self) -> set[int]:
        invalid_block_ids = self.invalid_recving_block_ids
        self.invalid_recving_block_ids = set()
        return invalid_block_ids

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()

        expired_transfer_id = []
        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                self.xfer_stats.record_kv_expired_req()
                finished_sending_reqs.add(send_meta.p_req_id)
                expired_transfer_id.append(transfer_id)

        for transfer_id in expired_transfer_id:
            del self.reqs_need_send[transfer_id]

        return finished_sending_reqs

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        recv_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if not self.is_kv_consumer:
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )

        finished_recving_reqs = recv_fut.result() if recv_fut else set()
        finished_sending_reqs = send_fut.result() if send_fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        return finished_sending_reqs or None, finished_recving_reqs or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.is_kv_producer:
            return set()
        return asyncio.run_coroutine_threadsafe(
            self.fetch_invalid_recving_block_ids(), self.receiver_loop
        ).result()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return transfer stats collected since the last call, or None
        if nothing has been recorded in this interval."""
        if self.xfer_stats.is_empty():
            return None
        return self.xfer_stats.clone_and_reset()

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        req_ids = set(pull_metas)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_lens=self.block_len_per_layer,
            kv_block_lens=self.kv_block_len_per_layer,
            registered_layer_names=self.registered_layer_names,
            registered_layer_indices=self.registered_layer_indices,
            registered_group_indices=self.registered_group_indices,
            registered_region_kinds=self.registered_region_kinds,
            dspark_context_transport=self._dspark_context_transport_enabled,
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug(
            "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
        )

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(encoded_data)
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        logger.error(
                            "Error happens during transferring kvcache for %s: %s",
                            req_ids,
                            response.err_msg,
                        )
                        for pull_meta in pull_metas.values():
                            self._record_failed_pull_task(pull_meta)
                        return
                    self.process_pulling_result(response, pull_metas)
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            for pull_meta in pull_metas.values():
                self._record_failed_pull_task(pull_meta)
            return

    @staticmethod
    def _pull_block_ids(pull_meta: PullReqMeta) -> set[int]:
        return {
            block_id
            for group_block_ids in pull_meta.local_block_ids
            for block_id in group_block_ids
            if block_id != NULL_BLOCK_ID and block_id >= 0
        }

    def _finalize_failed_pull(self, pull_meta: PullReqMeta) -> None:
        if pull_meta.d_req_id in self.finished_recving_reqs:
            return
        invalid_block_ids = self._pull_block_ids(pull_meta)
        self.invalid_recving_block_ids.update(invalid_block_ids)
        self.finished_recving_reqs.add(pull_meta.d_req_id)
        self.xfer_stats.record_failed_recv()
        logger.error(
            "Direct KV receive failed closed for req %s; invalidating %d blocks "
            "for local recomputation",
            pull_meta.d_req_id,
            len(invalid_block_ids),
        )

    def _record_failed_pull_task(self, pull_meta: PullReqMeta) -> None:
        # A D worker can pull from several P workers. Do not release/recompute
        # its destination blocks until every peer task has stopped writing.
        if pull_meta.pull_tasks_count > 0:
            pull_meta.pull_tasks_count -= 1
        if pull_meta.pull_tasks_count == 0:
            self._finalize_failed_pull(pull_meta)

    def process_pulling_result(
        self,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        ok_reqs: list[ReqId] = response.ok_reqs or []

        for req_id in ok_reqs:
            pull_meta = pull_metas[req_id]
            response_blocks = (response.kda_materialized_block_ids or {}).get(req_id)
            if response_blocks:
                for group_index, block_ids in response_blocks.items():
                    pull_meta.kda_materialized_block_ids.setdefault(
                        group_index, set()
                    ).update(block_ids)
            # No race because we are in async loop.
            pull_meta.pull_tasks_count -= 1
            if pull_meta.pull_tasks_count == 0:
                kda_transport = getattr(self, "_kda_transport", None)
                if kda_transport is not None:
                    allocated_by_group = {
                        group_index: {
                            block_id
                            for block_id in pull_meta.local_block_ids[group_index]
                            if block_id != NULL_BLOCK_ID and block_id >= 0
                        }
                        for group_index in kda_transport.group_indices
                        if group_index < len(pull_meta.local_block_ids)
                    }
                    expected_groups = {
                        group_index
                        for group_index, block_ids in allocated_by_group.items()
                        if block_ids
                    }
                    materialized_groups = set(pull_meta.kda_materialized_block_ids)
                    missing_groups = expected_groups - materialized_groups
                    invalid_blocks = {
                        group_index: block_ids
                        - allocated_by_group.get(group_index, set())
                        for group_index, block_ids in (
                            pull_meta.kda_materialized_block_ids.items()
                        )
                        if block_ids - allocated_by_group.get(group_index, set())
                    }
                    if missing_groups or invalid_blocks:
                        logger.error(
                            "req %s: incomplete KDA target-state Direct response: "
                            "missing_groups=%s invalid_blocks=%s",
                            req_id,
                            sorted(missing_groups),
                            invalid_blocks,
                        )
                        self._finalize_failed_pull(pull_meta)
                        continue
                    kda_transport.materialize_group_blocks(
                        {
                            group_index: sorted(block_ids)
                            for group_index, block_ids in (
                                pull_meta.kda_materialized_block_ids.items()
                            )
                        }
                    )
                    kda_transport.debug_checksums(
                        "D_DIRECT",
                        req_id,
                        pull_meta.local_block_ids,
                    )
                self.finished_recving_reqs.add(pull_meta.d_req_id)

        if ok_reqs:
            logger.debug("pulling kv_caches for %s finished", ok_reqs)

        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )
            for req_id in response.err_reqs:
                pull_meta = pull_metas.get(req_id)
                if pull_meta is not None:
                    self._record_failed_pull_task(pull_meta)

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data: dict = response.json()
                for _, dp_entry in data.items():
                    remote_engine_id = dp_entry["engine_id"]
                    self._remote_agents[remote_engine_id] = {
                        int(tp_rank): {
                            int(pp_rank): worker_addr
                            for pp_rank, worker_addr in tp_entry.items()
                        }
                        for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                    }
                    self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
        except Exception as e:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                e,
            )

        # Always notify others regardless of connection success or failure.
        self._pending_bootstrap_queries[remote_bootstrap_addr].set()
        del self._pending_bootstrap_queries[remote_bootstrap_addr]

    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(
            self._tp_size[remote_engine_id]
        )
        worker_addrs: list[str] = []
        selected_remote_pp: dict[int, list[int]] = {}
        for remote_tp_rank in remote_tp_ranks:
            pp_to_addr = self._remote_agents[remote_engine_id][remote_tp_rank]
            if self.pp_size == len(pp_to_addr) and self.pp_rank in pp_to_addr:
                pp_ranks = [self.pp_rank]
            else:
                pp_ranks = sorted(pp_to_addr)
            selected_remote_pp[remote_tp_rank] = pp_ranks
            worker_addrs.extend(pp_to_addr[pp_rank] for pp_rank in pp_ranks)

        count = len(worker_addrs)
        logger.debug(
            "Receiving Mooncake KV for engine %s from producer TP ranks %s "
            "and PP ranks %s",
            remote_engine_id,
            remote_tp_ranks,
            selected_remote_pp,
        )
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
        for worker_addr in worker_addrs:
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )

    async def handle_new_engine_id(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
        if remote_bootstrap_addr not in self._pending_bootstrap_queries:
            self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            for pull_meta in pull_metas.values():
                # No peer tasks were launched, so no writes can still be in
                # flight and the allocated destination blocks are immediately
                # safe to invalidate and recompute.
                pull_meta.pull_tasks_count = 0
                self._finalize_failed_pull(pull_meta)
            return

        self.receive_kv(remote_engine_id, pull_metas)

    async def _start_load_kv(
        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
    ):
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
            else:
                self.receive_kv(remote_engine_id, pull_metas)

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if block_ids:
                # Already gone through request_finished()
                send_meta = self.reqs_need_send.get(transfer_id)
                if send_meta is None:
                    # Normally update_state_after_alloc() or the D pull creates
                    # this placeholder first.  Preserve correctness if worker
                    # metadata arrives without it: publish the ready handoff so
                    # a later D pull can still consume the pinned blocks.
                    send_meta = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
                    self.reqs_need_send[transfer_id] = send_meta
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                send_meta.ready.set()
                if self._try_finish_send_meta(send_meta):
                    logger.info(
                        "Released late Mooncake Direct producer handoff %s "
                        "immediately after all %d consumer attempts had "
                        "already reached a terminal response.",
                        transfer_id,
                        send_meta.completed,
                    )
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
                # when D is sending MooncakeXferMetadata.
                if transfer_id not in self.reqs_need_send:
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
        for transfer_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(transfer_id, None)
            if send_meta:
                assert not send_meta.ready.is_set()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if not self.is_kv_producer and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if not self.is_kv_consumer and (
            metadata.reqs_to_send or metadata.reqs_not_processed
        ):
            asyncio.run_coroutine_threadsafe(
                self.record_send_reqs(metadata), self.sender_loop
            )

    def _producer_cache_is_replicated(self) -> bool:
        return self.transfer_topo.local_replicates_kv_cache

    def _get_transfer_regions(
        self,
        base_addrs: list[int],
        block_lens: list[int],
        kv_block_lens: list[int],
        layer_names: list[str],
        layer_indices: list[int],
        group_indices: list[int] | None = None,
        region_kinds: list[str] | None = None,
    ) -> list[TransferRegion]:
        if not group_indices:
            group_indices = [
                self._layer_group_indices.get(layer_name, 0)
                for layer_name in layer_names
            ]
        if not region_kinds:
            region_kinds = None
        return _expand_transfer_regions(
            base_addrs=base_addrs,
            block_lens=block_lens,
            kv_block_lens=kv_block_lens,
            layer_names=layer_names,
            layer_indices=layer_indices,
            group_indices=group_indices,
            region_kinds=region_kinds,
        )

    def _get_sender_transfer_plan(
        self,
        local_kv_block_len: int,
        remote_kv_block_len: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[bool, int, int, int]:
        return _compute_sender_transfer_plan(
            local_tp_rank=self.tp_rank,
            local_tp_size=self.tp_size,
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=remote_tp_size,
            local_kv_block_len=local_kv_block_len,
            remote_kv_block_len=remote_kv_block_len,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )

    def _log_debug_cache_registration(
        self, layer_name: str, cache: torch.Tensor
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Mooncake register view layer=%s shape=%s stride=%s "
            "storage_offset=%d contiguous=%s dense=%s data_ptr=%d",
            layer_name,
            tuple(cache.shape),
            tuple(cache.stride()),
            cache.storage_offset(),
            cache.is_contiguous(),
            _get_tensor_dense_flag(cache),
            cache.data_ptr(),
        )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_index
        * vllm_config.parallel_config.tensor_parallel_size
    )


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool:
    assert (parallel_config := vllm_config.parallel_config)
    # Only the TP=0, PP=0 worker of the designated engine should launch it.
    if get_tensor_model_parallel_rank() != 0:
        return False
    if get_pp_group().rank_in_group != 0:
        return False

    # In hybrid or external LB mode,
    # each instance should have its own bootstrap server.
    if parallel_config.local_engines_only:
        return parallel_config.data_parallel_rank_local == 0

    # In internal LB mode,
    # only the first data-parallel engine should launch the bootstrap server.
    return parallel_config.data_parallel_index == 0


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers.
    Decoders should get addr from kv_transfer_params.
    """
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        # In hybrid or external LB mode, connect to local server.
        host = "127.0.0.1"
    elif parallel_config.nnodes_within_dp > 1:
        # Internal LB multi-node TP/PP uses the model-parallel master as the
        # single bootstrap endpoint for all ranks in the engine.
        host = parallel_config.master_addr
    else:
        host = parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)
