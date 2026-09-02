# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# The transfer-thread scaffolding (KVTransferThread, KVCacheStoreSendingThread,
# KVCacheStoreRecvingThread) is adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Worker-side logic for MooncakeStoreConnector.

Includes the store worker, transfer threads, lookup server,
and MooncakeDistributedStore integration.
"""

import dataclasses
import hashlib
import json
import os
import queue
import socket
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import regex as re
import torch
import zmq

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.kda_recoverssm_transport import (
    KDA_TARGET_STATE_TRANSPORT,
    KDATargetStateTransport,
    kda_target_state_transport_enabled,
)
from vllm.distributed.kv_transfer.kv_connector.v1.dspark_context_transport import (
    DSPARK_CONTEXT_KV_TRANSPORT,
    DSPARK_CONTEXT_REGION_KIND,
    dspark_context_kv_transport_enabled,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
    store_effective_kv_cache_groups,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    BlobBlockHashes,
    ChunkedTokenDatabase,
    KeyMetadata,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    PoolKey,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import (  # noqa: E501
    LOOKUP_MSG,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
)
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_socket
from vllm.utils.torch_utils import is_non_overlapping_and_dense
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    maybe_convert_block_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)

from .metrics import MooncakeStoreConnectorStats

logger = init_logger(__name__)

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_TENANT_ID = "default"

MOONCAKE_NO_AVAILABLE_HANDLE = -200
SEMANTIC_COMMIT_REGION_ID = "__target_state_commit_v1__"
_T = TypeVar("_T")


def _make_semantic_region_id(
    layer_name: str,
    kind: str,
    occurrence: int,
) -> str:
    """Return a compact, partition-independent identity for one cache field."""
    descriptor = f"{layer_name}\0{kind}\0{occurrence}".encode()
    return hashlib.blake2b(descriptor, digest_size=16).hexdigest()


def _rotate_list(values: list[_T], offset: int) -> list[_T]:
    return values[offset:] + values[:offset]


def _replicate_config_supports_group_ids(
    replicate_config_cls: type[Any],
    replicate_config: Any,
) -> bool:
    if hasattr(replicate_config_cls, "group_ids"):
        return True
    return hasattr(replicate_config, "group_ids")


def _make_mooncake_group_id(metadata: KeyMetadata, chunk_hash: str) -> str:
    # Mooncake group ids describe the lifecycle unit. For vLLM, that unit is
    # a prefix chunk, so shard dimensions stay only in the object key.
    prefix = f"{metadata.cache_prefix}@" if metadata.cache_prefix else ""
    return f"vllm-mooncake-store:{prefix}{metadata.model_name}@{chunk_hash}"


# Mirrors FileStorageConfig::local_buffer_size in Mooncake C++.
DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES = 1280 * 1024 * 1024

# Mirrors DirectIO alignment in Mooncake's AllocateBatch.
_DIRECT_IO_ALIGNMENT = 4096
_DIRECT_IO_PADDING_BYTES = 2 * _DIRECT_IO_ALIGNMENT


MooncakeMode = Literal["embedded", "standalone-store"]


@dataclass
class MooncakeStoreConfig:
    """Configuration for MooncakeDistributedStore.

    ``mode`` selects the topology: ``embedded`` (each rank contributes
    ``global_segment_size`` in-process) or ``standalone-store`` (rank
    contributes 0; an external ``mooncake_client`` process owns the pool
    and the SSD tier).
    """

    metadata_server: str
    master_server_address: str
    protocol: str
    device_name: str
    mode: MooncakeMode = "embedded"
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE
    enable_offload: bool = False
    tenant_id: str = DEFAULT_TENANT_ID

    def __post_init__(self) -> None:
        if self.mode not in ("embedded", "standalone-store"):
            raise ValueError(f"unknown Mooncake mode: {self.mode!r}")
        if self.local_buffer_size <= 0:
            raise ValueError("local_buffer_size must be > 0")
        if self.mode == "embedded" and self.global_segment_size == 0:
            raise ValueError("embedded mode requires global_segment_size > 0")
        if self.mode == "standalone-store" and self.global_segment_size != 0:
            raise ValueError("standalone-store mode requires global_segment_size == 0")

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server", ""),
            master_server_address=config.get("master_server_address", ""),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
            mode=config.get("mode", "embedded"),
            global_segment_size=_parse_size(
                config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_size(
                config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)
            ),
            enable_offload=bool(config.get("enable_offload", False)),
            tenant_id=_normalize_tenant_id(config.get("tenant_id", DEFAULT_TENANT_ID)),
        )

    @staticmethod
    def load_from_config() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_path)


def _normalize_tenant_id(value: Any) -> str:
    if value is None:
        return DEFAULT_TENANT_ID
    if not isinstance(value, str):
        raise TypeError(
            f"tenant_id must be a string or null, got {type(value).__name__}: {value!r}"
        )
    tenant_id = value.strip()
    return tenant_id if tenant_id else DEFAULT_TENANT_ID


def _parse_size(value: Any) -> int:
    """Parse storage size strings with units: GB, MB, KB, B."""
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for size: {type(value)}") from e

    cleaned = value.strip().lower()
    if not cleaned:
        raise ValueError("Size cannot be empty.")

    unit_multipliers = {
        "gb": 1024**3,
        "mb": 1024**2,
        "kb": 1024,
        "b": 1,
    }
    match = re.match(r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$", cleaned)
    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"
    multiplier = unit_multipliers[unit]

    try:
        numeric_value = float(number_str)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{value}'") from exc
    return int(numeric_value * multiplier)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _estimate_disk_offload_staging_bytes(size_list: list[int]) -> int:
    data_size = sum(size_list)
    return _align_up(data_size, _DIRECT_IO_ALIGNMENT) + _DIRECT_IO_PADDING_BYTES


def _sum_batch_bytes(sizes: list[list[int]]) -> int:
    return sum(sum(size) for size in sizes)


def _get_usable_disk_offload_buffer_budget_bytes(raw_budget_bytes: int) -> int:
    return max(1, int(raw_budget_bytes * envs.VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO))


def _split_disk_offload_load_batches(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    usable_budget_bytes: int,
    raw_budget_bytes: int,
) -> tuple[list[tuple[list[str], list[list[int]], list[list[int]]]], str | None]:
    """Split a GET into sub-batches that fit the owner's staging buffer.

    ``addrs[i]`` / ``sizes[i]`` are scatter-gather lists (K/V or multi-layer
    segments) for key ``i``. ``usable_budget_bytes`` caps a multi-key batch;
    ``raw_budget_bytes`` is the hard per-key cap.

    Returns ``(batches, oversize_key)``. Aborts with ``([], key)`` if any
    single key exceeds ``raw_budget_bytes``; otherwise ``oversize_key`` is
    ``None``.
    """
    batches: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
    batch_keys: list[str] = []
    batch_addrs: list[list[int]] = []
    batch_sizes: list[list[int]] = []
    batch_bytes = 0

    for key, addr, size in zip(keys, addrs, sizes, strict=True):
        key_bytes = _estimate_disk_offload_staging_bytes(size)
        if key_bytes > raw_budget_bytes:
            return [], key
        if key_bytes > usable_budget_bytes:
            if batch_keys:
                batches.append((batch_keys, batch_addrs, batch_sizes))
                batch_keys, batch_addrs, batch_sizes = [], [], []
                batch_bytes = 0
            batches.append(([key], [addr], [size]))
            continue
        if batch_keys and batch_bytes + key_bytes > usable_budget_bytes:
            batches.append((batch_keys, batch_addrs, batch_sizes))
            batch_keys, batch_addrs, batch_sizes = [], [], []
            batch_bytes = 0
        batch_keys.append(key)
        batch_addrs.append(addr)
        batch_sizes.append(size)
        batch_bytes += key_bytes

    if batch_keys:
        batches.append((batch_keys, batch_addrs, batch_sizes))
    return batches, None


def _call_replica_predicate(replica_desc: Any, method_name: str) -> bool:
    method = getattr(replica_desc, method_name, None)
    if method is None:
        return False
    try:
        return bool(method())
    except Exception:
        return False


def _classify_replica_tier(replica_descs: Any) -> str:
    if not replica_descs:
        return "unknown"
    try:
        replica_desc = replica_descs[0]
    except (IndexError, KeyError, TypeError):
        return "unknown"

    if _call_replica_predicate(replica_desc, "is_memory_replica"):
        return "memory"
    if _call_replica_predicate(
        replica_desc, "is_disk_replica"
    ) or _call_replica_predicate(replica_desc, "is_local_disk_replica"):
        return "disk"
    return "unknown"


def _get_replica_tiers_by_key(store: Any, keys: list[str]) -> dict[str, str]:
    tiers_by_key = {key: "unknown" for key in keys}
    try:
        replica_descs_by_key = store.batch_get_replica_desc(keys)
    except Exception as e:
        logger.warning(
            "Failed to get Mooncake replica descriptors for tier logging "
            "(batch_keys=%d, error=%s); marking tiers unknown",
            len(keys),
            e,
        )
        return tiers_by_key

    for key in keys:
        if hasattr(replica_descs_by_key, "get"):
            replica_descs = replica_descs_by_key.get(key)
        else:
            try:
                replica_descs = replica_descs_by_key[key]
            except (KeyError, TypeError):
                replica_descs = None
        tiers_by_key[key] = _classify_replica_tier(replica_descs)
    return tiers_by_key


def _log_mooncake_load_tier_summary(
    req_id: str,
    batch_keys: list[str],
    load_results: list[int],
    tiers_by_key: dict[str, str],
) -> None:
    tier_counts = {"memory": 0, "disk": 0, "unknown": 0}
    bytes_by_tier = {"memory": 0, "disk": 0, "unknown": 0}
    success_keys = 0
    failed_keys = 0

    for index, key in enumerate(batch_keys):
        tier = tiers_by_key.get(key, "unknown")
        if tier not in tier_counts:
            tier = "unknown"
        tier_counts[tier] += 1

        value = load_results[index] if index < len(load_results) else -1
        if value >= 0:
            success_keys += 1
            bytes_by_tier[tier] += int(value)
        else:
            failed_keys += 1

    logger.info(
        "Mooncake load tier summary: req_id=%s batch_keys=%d "
        "memory_keys=%d disk_keys=%d unknown_keys=%d "
        "success_keys=%d failed_keys=%d bytes_by_tier=%s",
        req_id,
        len(batch_keys),
        tier_counts["memory"],
        tier_counts["disk"],
        tier_counts["unknown"],
        success_keys,
        failed_keys,
        bytes_by_tier,
    )


# ============================================================
# Transfer Threads
# ============================================================


class KVTransferThread(threading.Thread):
    """Base class for async KV cache transfer threads."""

    def __init__(
        self,
        store: Any,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        name: str,
        record_operation: Callable[..., None] | None = None,
        request_queue: queue.Queue[Any] | None = None,
    ):
        super().__init__(daemon=True, name=name)
        self.store = store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.token_databases = token_databases
        self._record_operation_cb = record_operation
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = request_queue or queue.Queue()
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []

    def add_request(self, request: ReqMeta) -> None:
        self.request_queue.put(request)

    def get_and_clear_finished_requests(self) -> set[str]:
        with self.done_task_lock:
            finished = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished

    def set_finished_request(self, req_id: str):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self):
        self.ready_event.set()
        while True:
            request_data = None
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request!")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception:
                req_id = getattr(request_data, "req_id", "<unknown>")
                logger.exception("Error in %s (req=%s)", self.name, req_id)

    def _handle_request(self, req_meta: Any):
        pass

    def _record_operation(
        self,
        operation: str,
        start_time: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        if self._record_operation_cb is None:
            return
        self._record_operation_cb(
            operation=operation,
            duration_seconds=time.perf_counter() - start_time,
            num_keys=num_keys,
            num_bytes=num_bytes,
            status=status,
            num_failed_keys=num_failed_keys,
        )

    def update_kv_event(self, events: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(events)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events


class KVCacheStoreSendingThread(KVTransferThread):
    """Background thread for storing KV cache blocks to the store."""

    def __init__(
        self,
        store: Any,
        coord: MooncakeStoreCoordinator,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        group_put_steps: Sequence[int],
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
        replicate_config: Any = None,
        enable_group_semantics: bool = False,
        supports_group_ids: bool = False,
        record_operation: Callable[..., None] | None = None,
        kda_transport: KDATargetStateTransport | None = None,
        semantic_region_dbs_by_group: Sequence[Sequence[ChunkedTokenDatabase]]
        | None = None,
        semantic_commit_metadata: KeyMetadata | None = None,
    ):
        super().__init__(
            store,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreSendingThread",
            record_operation=record_operation,
        )
        # Only ranks with identical group bytes may stripe PUTs (e.g., MLA).
        self.group_put_steps = group_put_steps
        self.coord = coord
        self.kv_role = kv_role
        # req_id -> ids of its store jobs that are still queued or running.
        # Keying by store_job_id, which never repeats for the engine's lifetime,
        # rather than counting jobs per request id makes the ledger immune to id
        # reuse across preemption: a job left over from a retired generation is
        # missing from the set its resumed generation builds, so it can no longer
        # retire that generation, rewind its resume offset, or mark it skipped.
        self.stored_requests: dict[str, set[int]] = {}
        # store_job_id -> times this rank finished with it. This is a
        # level-triggered ledger: entries remain until the scheduler ACKs them,
        # so a completion cannot be lost with a PP ModelRunnerOutput.
        self._completed_saves: dict[int, int] = {}
        self.enable_kv_event = enable_kv_event
        # Caller always passes a non-None ReplicateConfig — see
        # MooncakeStoreWorker.__init__ where store_replicate_config is built.
        self.replicate_config = replicate_config
        self.enable_group_semantics = enable_group_semantics
        self.supports_group_ids = supports_group_ids
        self.kda_transport = kda_transport
        self.semantic_region_dbs_by_group = semantic_region_dbs_by_group
        self.semantic_commit_metadata = semantic_commit_metadata or dataclasses.replace(
            token_databases[0].metadata,
            group_id=-1,
            region_id=SEMANTIC_COMMIT_REGION_ID,
        )
        self._semantic_put_summary_logged = False
        self._semantic_commit_summary_logged = False
        # Boundaries awaiting a complete, worker-local semantic data manifest.
        # A boundary may first arrive through only one HMA group; retain it so
        # a later store batch can publish the canonical commit once every
        # endpoint-neutral region object for this PP/TP shard is visible.
        self._semantic_commit_candidates: dict[bytes, BlockHash] = {}

        # Pause store requests when CPU/disk offloading is under pressure.
        self._store_pressure_active = False
        self._skip_store_requests: set[str] = set()

        # Per-request high-water mark of tokens actually persisted; the next
        # batch resumes here, so pressure-skipped or failed ranges are retried.
        self._saved_offset: dict[str, int] = {}
        # Retained only after a failed store so retry events can recover the
        # token suffix without full snapshots on the normal path.
        self._retry_token_ids: dict[str, tuple[int, list[int]]] = {}

    def add_request(self, request: ReqMeta) -> None:
        # Register before enqueueing so a job is never picked up unledgered.
        assert request.store_job_id is not None
        with self.done_task_lock:
            self.stored_requests.setdefault(request.req_id, set()).add(
                request.store_job_id
            )
        super().add_request(request)

    def is_live_store_job(self, req_meta: ReqMeta) -> bool:
        with self.done_task_lock:
            return req_meta.store_job_id in self.stored_requests.get(
                req_meta.req_id, ()
            )

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
            self._skip_store_requests.discard(req_id)
            self._saved_offset.pop(req_id, None)
            self._retry_token_ids.pop(req_id, None)

    def finish_store_job(self, req_meta: ReqMeta) -> None:
        """Retire a job from the ledger and report its blocks as no longer read.

        Every path out of a job must reach this, skips and failures included: a
        job that never reports leaves its blocks referenced for the rest of the
        run. The discard is a no-op for a job whose generation already retired.
        """
        store_job_id = req_meta.store_job_id
        assert store_job_id is not None, (
            "a queued store job always carries a store_job_id"
        )
        with self.done_task_lock:
            live = self.stored_requests.get(req_meta.req_id)
            if live is not None:
                live.discard(store_job_id)
            self._completed_saves[store_job_id] = (
                self._completed_saves.get(store_job_id, 0) + 1
            )

    def take_completed_saves(self) -> dict[int, int]:
        with self.done_task_lock:
            return dict(self._completed_saves)

    def acknowledge_completed_saves(self, store_job_ids: set[int]) -> None:
        if not store_job_ids:
            return
        with self.done_task_lock:
            for store_job_id in store_job_ids:
                self._completed_saves.pop(store_job_id, None)

    def _record_saved(self, req_meta: ReqMeta, token_len: int) -> None:
        # Guard on job liveness so neither a concurrent finish/preempt pop nor a
        # stale job's offset is written back over the live generation's.
        with self.done_task_lock:
            if req_meta.store_job_id in self.stored_requests.get(req_meta.req_id, ()):
                self._saved_offset[req_meta.req_id] = token_len

    def _get_retry_token_ids(self, req_meta: ReqMeta) -> tuple[int, list[int]] | None:
        """Return retry state only if this store job is still live."""
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(
                req_meta.req_id, ()
            ):
                return None
            return self._retry_token_ids.get(req_meta.req_id)

    def _update_retry_token_ids(
        self,
        req_meta: ReqMeta,
        save_completed: bool,
        token_ids_start: int,
        event_token_ids: list[int] | None,
    ) -> None:
        """Update retry state without letting a stale job touch a reused ID."""
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(
                req_meta.req_id, ()
            ):
                return
            if save_completed:
                self._retry_token_ids.pop(req_meta.req_id, None)
            elif event_token_ids is not None:
                self._retry_token_ids[req_meta.req_id] = (
                    token_ids_start,
                    event_token_ids,
                )

    def _should_skip_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            return self._store_pressure_active and req_id in self._skip_store_requests

    def _mark_request_skipped_for_pressure(self, req_meta: ReqMeta) -> bool:
        req_id = req_meta.req_id
        with self.done_task_lock:
            already_skipped = req_id in self._skip_store_requests
            self._store_pressure_active = True
            # The pressure itself is global, but only a live job may sentence its
            # own request to being skipped.
            if req_meta.store_job_id in self.stored_requests.get(req_id, ()):
                self._skip_store_requests.add(req_id)
        return already_skipped

    def _clear_store_pressure(self) -> bool:
        with self.done_task_lock:
            if not self._store_pressure_active and not self._skip_store_requests:
                return False
            self._store_pressure_active = False
            self._skip_store_requests.clear()
        return True

    def _put_semantic_entries(
        self,
        marker_keys: list[str],
        chunk_hashes: list[BlockHash],
        group_indices: list[int],
        block_ids: list[int],
        current_event: torch.cuda.Event | None,
        *,
        debug_request_id: str | None = None,
        debug_boundaries: Sequence[int] | None = None,
    ) -> list[bool]:
        """Persist semantic region objects, then publish one-byte markers.

        A marker is the lookup commit record for one ``(group, chunk)``.  It is
        published only after every target-state region owned by this worker is
        present. Region keys omit PP rank, so a consumer can materialize a
        layer even when P and D assign that layer to different PP stages.
        """
        region_groups = self.semantic_region_dbs_by_group
        assert region_groups is not None
        assert len(marker_keys) == len(chunk_hashes)
        assert len(marker_keys) == len(group_indices) == len(block_ids)
        if debug_boundaries is not None:
            assert len(marker_keys) == len(debug_boundaries)

        debug_entries = (
            self.kda_transport is not None
            and debug_request_id is not None
            and debug_boundaries is not None
            and os.getenv("VLLM_KDA_TRANSPORT_ENTRY_CHECKSUM") == "1"
        )

        def log_entry_checksums(location: str, entry_indices: set[int]) -> None:
            if not debug_entries or not entry_indices:
                return
            assert self.kda_transport is not None
            assert debug_request_id is not None
            assert debug_boundaries is not None
            for entry_index in sorted(entry_indices):
                self.kda_transport.debug_entry_checksums(
                    location,
                    debug_request_id,
                    debug_boundaries[entry_index],
                    group_indices[entry_index],
                    block_ids[entry_index],
                    bytes(chunk_hashes[entry_index]),
                )

        def stage_entry_indices(entry_indices: set[int]) -> None:
            if self.kda_transport is None or not entry_indices:
                return
            stage_blocks: dict[int, list[int]] = {}
            for entry_index in sorted(entry_indices):
                blocks = stage_blocks.setdefault(group_indices[entry_index], [])
                block_id = block_ids[entry_index]
                if block_id not in blocks:
                    blocks.append(block_id)
            self.kda_transport.stage_group_blocks(stage_blocks)

        entry_ok = [True] * len(marker_keys)
        region_counts = [0] * len(marker_keys)
        data_keys: list[str] = []
        data_addrs: list[list[int]] = []
        data_sizes: list[list[int]] = []
        data_entry_indices: list[int] = []
        data_group_ids: list[str] | None = (
            [] if self.enable_group_semantics and self.supports_group_ids else None
        )
        for entry_index, (chunk_hash, group_index, block_id) in enumerate(
            zip(chunk_hashes, group_indices, block_ids, strict=True)
        ):
            for region_db in region_groups[group_index]:
                region_counts[entry_index] += 1
                addr, size = region_db.prepare_value_for_block(block_id)
                key = region_db.key_for(chunk_hash)
                data_keys.append(key)
                data_addrs.append(addr)
                data_sizes.append(size)
                data_entry_indices.append(entry_index)
                if data_group_ids is not None:
                    data_group_ids.append(
                        _make_mooncake_group_id(
                            region_db.metadata,
                            chunk_hash.hex(),
                        )
                    )

        missing_data_indices: list[int] = []
        if data_keys:
            exists_start = time.perf_counter()
            try:
                data_exists = self.store.batch_is_exist(data_keys)
            except Exception:
                self._record_operation(
                    "save_region_exists",
                    exists_start,
                    len(data_keys),
                    status="error",
                    num_failed_keys=len(data_keys),
                )
                return [False] * len(marker_keys)
            if len(data_exists) != len(data_keys):
                self._record_operation(
                    "save_region_exists",
                    exists_start,
                    len(data_keys),
                    status="error",
                    num_failed_keys=len(data_keys),
                )
                logger.warning(
                    "Semantic Mooncake region existence check returned %d "
                    "results for %d keys (first_key=%s, last_key=%s)",
                    len(data_exists),
                    len(data_keys),
                    data_keys[0],
                    data_keys[-1],
                )
                return [False] * len(marker_keys)
            self._record_operation(
                "save_region_exists",
                exists_start,
                len(data_keys),
            )
            missing_data_indices = [
                i for i, exists in enumerate(data_exists) if exists != 1
            ]

        missing_counts = [0] * len(marker_keys)
        for data_index in missing_data_indices:
            missing_counts[data_entry_indices[data_index]] += 1
        if debug_entries:
            assert debug_request_id is not None
            assert debug_boundaries is not None
            for entry_index, (chunk_hash, group_index, block_id) in enumerate(
                zip(chunk_hashes, group_indices, block_ids, strict=True)
            ):
                logger.info(
                    "KDA_STORE_PUT_PLAN req=%s boundary=%d group=%d block=%d "
                    "hash=%s regions=%d missing=%d",
                    debug_request_id,
                    debug_boundaries[entry_index],
                    group_index,
                    block_id,
                    chunk_hash.hex(),
                    region_counts[entry_index],
                    missing_counts[entry_index],
                )

        if missing_data_indices:
            if current_event is not None:
                current_event.synchronize()
            missing_entry_indices = {
                data_entry_indices[i] for i in missing_data_indices
            }
            stage_entry_indices(missing_entry_indices)
            log_entry_checksums("STORE_PUT_SOURCE", missing_entry_indices)

            existing_entry_indices = {
                entry_index
                for entry_index, missing_count in enumerate(missing_counts)
                if missing_count == 0
            }
            if debug_entries and existing_entry_indices:
                stage_entry_indices(existing_entry_indices)
                log_entry_checksums(
                    "STORE_PUT_EXISTING_SOURCE", existing_entry_indices
                )

            missing_keys = [data_keys[i] for i in missing_data_indices]
            missing_addrs = [data_addrs[i] for i in missing_data_indices]
            missing_sizes = [data_sizes[i] for i in missing_data_indices]
            if data_group_ids is not None:
                self.replicate_config.group_ids = [
                    data_group_ids[i] for i in missing_data_indices
                ]
            put_start = time.perf_counter()
            try:
                results = self.store.batch_put_from_multi_buffers(
                    missing_keys,
                    missing_addrs,
                    missing_sizes,
                    self.replicate_config,
                )
                log_entry_checksums("STORE_PUT_AFTER", missing_entry_indices)
            except Exception:
                self._record_operation(
                    "save_region_put",
                    put_start,
                    len(missing_keys),
                    num_bytes=_sum_batch_bytes(missing_sizes),
                    status="error",
                    num_failed_keys=len(missing_keys),
                )
                return [False] * len(marker_keys)
            if len(results) != len(missing_keys):
                self._record_operation(
                    "save_region_put",
                    put_start,
                    len(missing_keys),
                    num_bytes=_sum_batch_bytes(missing_sizes),
                    status="error",
                    num_failed_keys=len(missing_keys),
                )
                logger.warning(
                    "Semantic Mooncake region put returned %d results for %d "
                    "keys (bytes=%d, first_key=%s, last_key=%s)",
                    len(results),
                    len(missing_keys),
                    _sum_batch_bytes(missing_sizes),
                    missing_keys[0],
                    missing_keys[-1],
                )
                return [False] * len(marker_keys)
            failed = [i for i, result in enumerate(results) if result < 0]
            self._record_operation(
                "save_region_put",
                put_start,
                len(missing_keys),
                num_bytes=_sum_batch_bytes(missing_sizes),
                status="partial_failure" if failed else "ok",
                num_failed_keys=len(failed),
            )
            for failed_index in failed:
                data_index = missing_data_indices[failed_index]
                entry_ok[data_entry_indices[data_index]] = False
            if failed:
                logger.warning(
                    "Semantic Mooncake region put failed (first_failures=%s)",
                    [
                        (
                            missing_keys[i],
                            results[i],
                            sum(missing_sizes[i]),
                        )
                        for i in failed[:3]
                    ],
                )
            failed_codes = {results[i] for i in failed}
            if MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes:
                # The caller owns request-level pressure bookkeeping. Returning
                # False prevents its marker and saved-offset commit.
                logger.warning(
                    "Semantic Mooncake region put hit storage pressure (failed=%d/%d)",
                    len(failed),
                    len(missing_keys),
                )
        elif debug_entries and marker_keys:
            if current_event is not None:
                current_event.synchronize()
            existing_entry_indices = set(range(len(marker_keys)))
            stage_entry_indices(existing_entry_indices)
            log_entry_checksums("STORE_PUT_EXISTING_SOURCE", existing_entry_indices)

        # Mooncake put completion is not a transactional visibility guarantee.
        # Publish commit markers only after every region for the entry is
        # observable through the same API used by readers.
        visibility_start = time.perf_counter()
        try:
            visible_data = self.store.batch_is_exist(data_keys)
        except Exception:
            self._record_operation(
                "save_region_commit_verify",
                visibility_start,
                len(data_keys),
                status="error",
                num_failed_keys=len(data_keys),
            )
            return [False] * len(marker_keys)
        if len(visible_data) != len(data_keys):
            self._record_operation(
                "save_region_commit_verify",
                visibility_start,
                len(data_keys),
                status="error",
                num_failed_keys=len(data_keys),
            )
            logger.warning(
                "Semantic Mooncake commit verification returned %d results "
                "for %d region keys",
                len(visible_data),
                len(data_keys),
            )
            return [False] * len(marker_keys)
        invisible = [i for i, exists in enumerate(visible_data) if exists != 1]
        self._record_operation(
            "save_region_commit_verify",
            visibility_start,
            len(data_keys),
            status="partial_failure" if invisible else "ok",
            num_failed_keys=len(invisible),
        )
        for data_index in invisible:
            entry_ok[data_entry_indices[data_index]] = False
        if invisible:
            logger.warning(
                "Semantic Mooncake commit verification found %d/%d invisible "
                "regions (first=%s)",
                len(invisible),
                len(data_keys),
                data_keys[invisible[0]],
            )

        marker_indices = [i for i, ok in enumerate(entry_ok) if ok]
        if not marker_indices:
            return entry_ok

        marker_addrs: list[list[int]] = []
        marker_sizes: list[list[int]] = []
        marker_group_ids: list[str] | None = (
            [] if self.enable_group_semantics and self.supports_group_ids else None
        )
        for i in marker_indices:
            marker_db = self.token_databases[group_indices[i]]
            addr, size = marker_db.prepare_value_for_block(block_ids[i])
            marker_addrs.append(addr)
            marker_sizes.append(size)
            if marker_group_ids is not None:
                marker_group_ids.append(
                    _make_mooncake_group_id(
                        marker_db.metadata,
                        chunk_hashes[i].hex(),
                    )
                )
        if marker_group_ids is not None:
            self.replicate_config.group_ids = marker_group_ids
        committed_keys = [marker_keys[i] for i in marker_indices]
        marker_start = time.perf_counter()
        try:
            marker_results = self.store.batch_put_from_multi_buffers(
                committed_keys,
                marker_addrs,
                marker_sizes,
                self.replicate_config,
            )
        except Exception:
            self._record_operation(
                "save_marker_put",
                marker_start,
                len(committed_keys),
                num_bytes=_sum_batch_bytes(marker_sizes),
                status="error",
                num_failed_keys=len(committed_keys),
            )
            for i in marker_indices:
                entry_ok[i] = False
            return entry_ok
        if len(marker_results) != len(committed_keys):
            self._record_operation(
                "save_marker_put",
                marker_start,
                len(committed_keys),
                num_bytes=_sum_batch_bytes(marker_sizes),
                status="error",
                num_failed_keys=len(committed_keys),
            )
            logger.warning(
                "Semantic Mooncake marker put returned %d results for %d keys "
                "(first_key=%s, last_key=%s)",
                len(marker_results),
                len(committed_keys),
                committed_keys[0],
                committed_keys[-1],
            )
            for i in marker_indices:
                entry_ok[i] = False
            return entry_ok
        failed_markers = [i for i, result in enumerate(marker_results) if result < 0]
        self._record_operation(
            "save_marker_put",
            marker_start,
            len(committed_keys),
            num_bytes=_sum_batch_bytes(marker_sizes),
            status="partial_failure" if failed_markers else "ok",
            num_failed_keys=len(failed_markers),
        )
        for failed_index in failed_markers:
            entry_ok[marker_indices[failed_index]] = False
        self._publish_semantic_commits(
            chunk_hashes,
            group_indices,
            entry_ok,
        )
        if (
            not self._semantic_put_summary_logged
            and missing_data_indices
            and not failed
            and not failed_markers
        ):
            self._semantic_put_summary_logged = True
            logger.info(
                "Semantic Mooncake first commit: entries=%d, region_keys=%d, "
                "region_bytes=%d, returned_region_bytes=%d, "
                "region_results=%d, marker_results=%d, "
                "first_region=%s, last_region=%s, first_marker=%s, "
                "last_marker=%s",
                len(marker_keys),
                len(missing_keys),
                _sum_batch_bytes(missing_sizes),
                sum(result for result in results if result > 0),
                len(results),
                len(marker_results),
                missing_keys[0],
                missing_keys[-1],
                committed_keys[0],
                committed_keys[-1],
            )
        return entry_ok

    def _publish_semantic_commits(
        self,
        chunk_hashes: Sequence[BlockHash],
        group_indices: Sequence[int],
        entry_ok: list[bool],
    ) -> None:
        """Publish endpoint-neutral per-stage commits at complete boundaries.

        Local ``group_id`` values are HMA implementation details and may differ
        between a no-spec Prefill endpoint and a RecoverSSM Decode endpoint.
        A canonical commit is therefore keyed only by the physical model shard,
        producer PP stage, schema id, and boundary hash. It becomes visible only
        after every endpoint-neutral semantic region object owned by this worker
        is observable. This data-manifest check is intentionally independent of
        local HMA group ids and group boundaries; readers require the commit from
        every TP/DCP/PCP/PP worker.
        """
        current_entry_indices_by_hash: dict[bytes, list[int]] = {}
        for index, (chunk_hash, _group_index, ok) in enumerate(
            zip(chunk_hashes, group_indices, entry_ok, strict=True)
        ):
            if not ok:
                continue
            hash_bytes = bytes(chunk_hash)
            self._semantic_commit_candidates.setdefault(hash_bytes, chunk_hash)
            current_entry_indices_by_hash.setdefault(hash_bytes, []).append(index)
        if not current_entry_indices_by_hash:
            return

        # Bound the cross-call ledger before constructing the manifest query.
        # Dropping an old incomplete boundary can only turn a future lookup into
        # a safe miss.
        while len(self._semantic_commit_candidates) > 16384:
            oldest_hash = next(iter(self._semantic_commit_candidates))
            self._semantic_commit_candidates.pop(oldest_hash)

        region_groups = self.semantic_region_dbs_by_group
        assert region_groups is not None
        region_dbs = [region_db for group in region_groups for region_db in group]
        if not region_dbs:
            return

        pending_hashes = list(self._semantic_commit_candidates)
        manifest_keys: list[str] = []
        manifest_ranges: list[tuple[int, int]] = []
        for hash_bytes in pending_hashes:
            chunk_hash = self._semantic_commit_candidates[hash_bytes]
            start = len(manifest_keys)
            # Region ids are normally unique per local layer/field. Dedup makes
            # the correctness check robust to an accidentally repeated descriptor.
            manifest_keys.extend(
                dict.fromkeys(region_db.key_for(chunk_hash) for region_db in region_dbs)
            )
            manifest_ranges.append((start, len(manifest_keys)))

        manifest_start = time.perf_counter()
        try:
            manifest_results = self.store.batch_is_exist(manifest_keys)
        except Exception as error:
            logger.warning(
                "Semantic Mooncake manifest verification failed: "
                "boundaries=%d, keys=%d, first_key=%s, error=%s",
                len(pending_hashes),
                len(manifest_keys),
                manifest_keys[0],
                error,
            )
            manifest_results = []
        if len(manifest_results) != len(manifest_keys):
            self._record_operation(
                "save_semantic_manifest_verify",
                manifest_start,
                len(manifest_keys),
                status="error",
                num_failed_keys=len(manifest_keys),
            )
            logger.warning(
                "Semantic Mooncake manifest verification returned %d results "
                "for %d keys (boundaries=%d, first_key=%s)",
                len(manifest_results),
                len(manifest_keys),
                len(pending_hashes),
                manifest_keys[0],
            )
            for indices in current_entry_indices_by_hash.values():
                for index in indices:
                    entry_ok[index] = False
            return
        incomplete_manifest_keys = sum(result != 1 for result in manifest_results)
        self._record_operation(
            "save_semantic_manifest_verify",
            manifest_start,
            len(manifest_keys),
            status="partial_failure" if incomplete_manifest_keys else "ok",
            num_failed_keys=incomplete_manifest_keys,
        )
        complete_hashes = [
            hash_bytes
            for hash_bytes, (start, end) in zip(
                pending_hashes, manifest_ranges, strict=True
            )
            if all(result == 1 for result in manifest_results[start:end])
        ]
        if not complete_hashes:
            return

        commit_keys = [
            PoolKey(
                self.semantic_commit_metadata,
                self._semantic_commit_candidates[hash_bytes].hex(),
            ).to_string()
            for hash_bytes in complete_hashes
        ]
        marker_db = self.token_databases[0]
        marker_addr, marker_size = marker_db.prepare_value_for_block(0)
        commit_addrs = [marker_addr] * len(commit_keys)
        commit_sizes = [marker_size] * len(commit_keys)
        if self.enable_group_semantics and self.supports_group_ids:
            self.replicate_config.group_ids = [
                _make_mooncake_group_id(
                    self.semantic_commit_metadata,
                    self._semantic_commit_candidates[hash_bytes].hex(),
                )
                for hash_bytes in complete_hashes
            ]
        put_start = time.perf_counter()
        try:
            commit_results = self.store.batch_put_from_multi_buffers(
                commit_keys,
                commit_addrs,
                commit_sizes,
                self.replicate_config,
            )
        except Exception as error:
            logger.warning(
                "Semantic Mooncake canonical commit put failed: "
                "boundaries=%d, first_key=%s, error=%s",
                len(commit_keys),
                commit_keys[0],
                error,
            )
            commit_results = []
        if len(commit_results) != len(commit_keys):
            self._record_operation(
                "save_semantic_commit_put",
                put_start,
                len(commit_keys),
                num_bytes=_sum_batch_bytes(commit_sizes),
                status="error",
                num_failed_keys=len(commit_keys),
            )
            logger.warning(
                "Semantic Mooncake canonical commit put returned %d results "
                "for %d keys (first_key=%s)",
                len(commit_results),
                len(commit_keys),
                commit_keys[0],
            )
            for hash_bytes in complete_hashes:
                for index in current_entry_indices_by_hash.get(hash_bytes, ()):
                    entry_ok[index] = False
            return
        failed_commits = [i for i, result in enumerate(commit_results) if result < 0]
        self._record_operation(
            "save_semantic_commit_put",
            put_start,
            len(commit_keys),
            num_bytes=_sum_batch_bytes(commit_sizes),
            status="partial_failure" if failed_commits else "ok",
            num_failed_keys=len(failed_commits),
        )
        for failed_index in failed_commits:
            hash_bytes = complete_hashes[failed_index]
            for index in current_entry_indices_by_hash.get(hash_bytes, ()):
                entry_ok[index] = False
        if failed_commits:
            logger.warning(
                "Semantic Mooncake canonical commit put failed for %d/%d "
                "boundaries (codes=%s, first_key=%s)",
                len(failed_commits),
                len(commit_keys),
                sorted({commit_results[index] for index in failed_commits}),
                commit_keys[failed_commits[0]],
            )
            return

        verify_start = time.perf_counter()
        try:
            visible_commits = self.store.batch_is_exist(commit_keys)
        except Exception as error:
            logger.warning(
                "Semantic Mooncake canonical commit visibility check failed: "
                "boundaries=%d, first_key=%s, error=%s",
                len(commit_keys),
                commit_keys[0],
                error,
            )
            visible_commits = []
        if len(visible_commits) != len(commit_keys):
            invisible_commits = list(range(len(commit_keys)))
        else:
            invisible_commits = [
                i for i, exists in enumerate(visible_commits) if exists != 1
            ]
        self._record_operation(
            "save_semantic_commit_verify",
            verify_start,
            len(commit_keys),
            status="partial_failure" if invisible_commits else "ok",
            num_failed_keys=len(invisible_commits),
        )
        for invisible_index in invisible_commits:
            hash_bytes = complete_hashes[invisible_index]
            for index in current_entry_indices_by_hash.get(hash_bytes, ()):
                entry_ok[index] = False
        if invisible_commits:
            logger.warning(
                "Semantic Mooncake canonical commits not visible for %d/%d "
                "boundaries (result_count=%d, first_key=%s)",
                len(invisible_commits),
                len(commit_keys),
                len(visible_commits),
                commit_keys[invisible_commits[0]],
            )
        for index, hash_bytes in enumerate(complete_hashes):
            if index not in invisible_commits:
                self._semantic_commit_candidates.pop(hash_bytes, None)
        if not invisible_commits and not self._semantic_commit_summary_logged:
            self._semantic_commit_summary_logged = True
            logger.info(
                "Semantic Mooncake canonical commit: boundaries=%d, first=%s, last=%s",
                len(commit_keys),
                commit_keys[0],
                commit_keys[-1],
            )

    def _maybe_offload_partial_tail(self, req_meta: ReqMeta) -> bool:
        """Offload the request's sub-block partial tail (its last prompt hash
        boundary) so a later request can hit the sub-block prefix.

        Covers every block from the normal save's lcm floor to the boundary:
        the normal save floors to ``lcm_block_size``, so a smaller-block
        group's full blocks in that gap are never persisted elsewhere, and
        the consumer's lookup needs every group at every probed boundary.
        Full blocks are keyed by their block-end hash, the partial boundary
        block by the boundary sub-hash; the mamba "align" boundary block is
        the core-provided CoW block. All keys are deduped against the store.

        Returns:
            True when no put is needed or every put succeeds, False otherwise.
        """
        if not self.coord.enable_partial_hash_hits or not req_meta.block_hashes:
            return True
        partial_tail_offloads = req_meta.partial_tail_offloads
        if not partial_tail_offloads:
            return True
        hash_block_size = self.coord.hash_block_size
        handoffs: dict[tuple[int, int], int] = {}
        for group_id, block_id, boundary_tokens in partial_tail_offloads:
            key = (group_id, boundary_tokens)
            previous = handoffs.setdefault(key, block_id)
            if previous != block_id:
                raise ValueError(
                    "Conflicting KDA state hand-offs for request "
                    f"{req_meta.req_id}, group {group_id}, boundary "
                    f"{boundary_tokens}: {previous} != {block_id}"
                )
        boundaries = sorted({boundary for _, boundary in handoffs})
        boundary = boundaries[-1]
        logger.info(
            "Semantic KDA state handoff: req=%s, boundaries=%s, groups=%s",
            req_meta.req_id,
            boundaries,
            sorted({group_id for group_id, _boundary in handoffs}),
        )
        if boundary == 0:
            return True
        if boundary // hash_block_size - 1 >= len(req_meta.block_hashes):
            return True
        keys: list[str] = []
        addrs: list[list[int]] = []
        sizes: list[list[int]] = []
        key_hashes: list[BlockHash] = []
        key_block_ids: list[int] = []
        key_group_indices: list[int] = []
        key_boundaries: list[int] = []
        group_ids: list[str] | None = (
            [] if self.enable_group_semantics and self.supports_group_ids else None
        )
        saved = self._saved_offset.get(req_meta.req_id, 0)
        for g_idx, db in enumerate(self.token_databases):
            group_blocks = req_meta.block_ids[g_idx]
            explicit_group_handoffs = {
                handoff_boundary: block_id
                for (group_id, handoff_boundary), block_id in handoffs.items()
                if group_id == g_idx
            }
            explicit_block_ids = set(explicit_group_handoffs.values())
            # Distribute across ranks by the same rule as normal chunks.
            put_step = self.group_put_steps[g_idx]
            put_step_rank = (self.tp_rank + g_idx) % put_step
            # Save ordinary physical slots by their natural block-end boundary,
            # but never infer a key for an explicitly described state slot. A
            # FlashKDA checkpoint can live near the tail while representing an
            # earlier scheduler/DCP boundary, so physical-index keying would
            # silently mislabel it.
            last_block = cdiv(boundary, db.block_size) - 1
            for block_idx in range(
                min(saved // db.block_size, last_block), last_block + 1
            ):
                if block_idx % put_step != put_step_rank:
                    continue
                valid_end = min((block_idx + 1) * db.block_size, boundary)
                # An explicit state descriptor is authoritative for its semantic
                # boundary even when its CoW/running block is not present in the
                # request's physical block table.  Emitting the inferred table
                # slot as well would put the same canonical key twice in one
                # Mooncake batch with different source addresses. Mooncake keeps
                # one of those values, so the later exact replay can silently
                # restore the inferred checkpoint instead of h(boundary).
                if valid_end in explicit_group_handoffs:
                    continue
                key_hash = req_meta.block_hashes[valid_end // hash_block_size - 1]
                if block_idx >= len(group_blocks):
                    continue
                block_id = group_blocks[block_idx]
                if block_id in explicit_block_ids:
                    continue
                if block_id == NULL_BLOCK_ID:
                    logger.debug(
                        "Skipping unavailable partial-tail source block "
                        "(req=%s, group=%d, block=%d)",
                        req_meta.req_id,
                        g_idx,
                        block_idx,
                    )
                    continue
                addr, size = db.prepare_value_for_block(block_id)
                key = db.key_for(key_hash)
                keys.append(key)
                addrs.append(addr)
                sizes.append(size)
                key_hashes.append(key_hash)
                key_block_ids.append(block_id)
                key_group_indices.append(g_idx)
                key_boundaries.append(valid_end)
                if group_ids is not None:
                    group_ids.append(
                        _make_mooncake_group_id(
                            db.metadata,
                            key.rsplit("@", 1)[-1],
                        )
                    )

            # Explicit KDA state descriptors own their semantic key. This
            # covers both the final running h(N-1) slot (including an exactly
            # block-aligned boundary) and internal Prefill checkpoints whose
            # physical slot index is intentionally unrelated to the key.
            for handoff_boundary, block_id in sorted(
                explicit_group_handoffs.items()
            ):
                if handoff_boundary <= 0:
                    continue
                hash_index = handoff_boundary // hash_block_size - 1
                if hash_index < 0 or hash_index >= len(req_meta.block_hashes):
                    continue
                logical_block_idx = (handoff_boundary - 1) // db.block_size
                if logical_block_idx % put_step != put_step_rank:
                    continue
                key_hash = req_meta.block_hashes[hash_index]
                addr, size = db.prepare_value_for_block(block_id)
                key = db.key_for(key_hash)
                keys.append(key)
                addrs.append(addr)
                sizes.append(size)
                key_hashes.append(key_hash)
                key_block_ids.append(block_id)
                key_group_indices.append(g_idx)
                key_boundaries.append(handoff_boundary)
                if group_ids is not None:
                    group_ids.append(
                        _make_mooncake_group_id(
                            db.metadata,
                            key.rsplit("@", 1)[-1],
                        )
                    )

        if not keys:
            return True
        exists_start = time.perf_counter()
        try:
            exists = self.store.batch_is_exist(keys)
        except Exception as e:
            self._record_operation(
                "save_exists",
                exists_start,
                len(keys),
                status="error",
                num_failed_keys=len(keys),
            )
            logger.error(
                "Failed to check partial-tail keys for request %s: %s",
                req_meta.req_id,
                e,
            )
            return False
        if len(exists) != len(keys):
            self._record_operation(
                "save_exists",
                exists_start,
                len(keys),
                status="error",
                num_failed_keys=len(keys),
            )
            logger.warning(
                "Partial-tail Mooncake existence check returned %d results "
                "for %d keys (req=%s)",
                len(exists),
                len(keys),
                req_meta.req_id,
            )
            return False
        self._record_operation("save_exists", exists_start, len(keys))
        missing = [i for i, e in enumerate(exists) if e != 1]
        if self.semantic_region_dbs_by_group is None:
            if not missing:
                return True
            keys = [keys[i] for i in missing]
            addrs = [addrs[i] for i in missing]
            sizes = [sizes[i] for i in missing]
            key_hashes = [key_hashes[i] for i in missing]
            key_block_ids = [key_block_ids[i] for i in missing]
            key_group_indices = [key_group_indices[i] for i in missing]
            if group_ids is not None:
                group_ids = [group_ids[i] for i in missing]
        if self.semantic_region_dbs_by_group is not None:
            if (
                self.kda_transport is not None
                and os.getenv("VLLM_KDA_TRANSPORT_ENTRY_CHECKSUM") == "1"
            ):
                # The ordinary save batch may contain multiple physical pages.
                # Stage and checksum only the explicitly described KDA state
                # slots so the subsequent load can be paired by semantic key.
                if req_meta.current_event is not None:
                    req_meta.current_event.synchronize()
                exact_blocks: dict[int, list[int]] = {}
                for group_index, block_id in zip(
                    key_group_indices, key_block_ids, strict=True
                ):
                    if handoffs.get((group_index, boundary)) == block_id:
                        exact_blocks.setdefault(group_index, []).append(block_id)
                self.kda_transport.stage_group_blocks(exact_blocks)
                for group_index, block_ids in exact_blocks.items():
                    key_hash = req_meta.block_hashes[
                        boundary // hash_block_size - 1
                    ]
                    for block_id in dict.fromkeys(block_ids):
                        self.kda_transport.debug_entry_checksums(
                            "STORE_SAVE",
                            req_meta.req_id,
                            boundary,
                            group_index,
                            block_id,
                            bytes(key_hash),
                        )
            # A local HMA marker may outlive a failed/invisible canonical
            # commit. Re-run the semantic transaction for every candidate,
            # including marker hits, so retries and process restarts can repair
            # the endpoint-neutral commit instead of making a transient failure
            # a permanent logical miss. Region objects remain deduplicated by
            # _put_semantic_entries; only one-byte markers are republished.
            results = self._put_semantic_entries(
                keys,
                key_hashes,
                key_group_indices,
                key_block_ids,
                req_meta.current_event,
                debug_request_id=req_meta.req_id,
                debug_boundaries=key_boundaries,
            )
            if self.kda_transport is not None:
                checksum_blocks: dict[int, list[int]] = {}
                for group_index, block_id in zip(
                    key_group_indices, key_block_ids, strict=True
                ):
                    checksum_blocks.setdefault(group_index, []).append(block_id)
                self.kda_transport.debug_checksums(
                    "P_STORE",
                    req_meta.req_id,
                    checksum_blocks,
                )
            if all(results):
                logger.info(
                    "Semantic partial-tail commit: req=%s, boundary=%d, "
                    "entries=%d, groups=%s",
                    req_meta.req_id,
                    boundary,
                    len(results),
                    sorted(set(key_group_indices)),
                )
                if self._clear_store_pressure():
                    logger.info(
                        "Mooncake CPU/disk offloading pressure cleared after a "
                        "successful semantic partial-tail batch"
                    )
                return True
            return False
        if req_meta.current_event is not None:
            # Fence the CoW block copy enqueued earlier this step.
            req_meta.current_event.synchronize()
        if self.kda_transport is not None:
            stage_blocks: dict[int, list[int]] = {}
            for group_index, block_id in zip(
                key_group_indices, key_block_ids, strict=True
            ):
                stage_blocks.setdefault(group_index, []).append(block_id)
            self.kda_transport.stage_group_blocks(stage_blocks)
        if group_ids is not None:
            assert len(group_ids) == len(keys)
            self.replicate_config.group_ids = group_ids
        batch_bytes = _sum_batch_bytes(sizes)
        put_start = time.perf_counter()
        try:
            res = self.store.batch_put_from_multi_buffers(
                keys, addrs, sizes, self.replicate_config
            )
        except Exception as e:
            self._record_operation(
                "save_put",
                put_start,
                len(keys),
                num_bytes=batch_bytes,
                status="error",
                num_failed_keys=len(keys),
            )
            logger.error(
                "Failed to put partial-tail keys for request %s: %s",
                req_meta.req_id,
                e,
            )
            return False

        failed = [i for i, value in enumerate(res) if value < 0]
        self._record_operation(
            "save_put",
            put_start,
            len(keys),
            num_bytes=batch_bytes,
            status="partial_failure" if failed else "ok",
            num_failed_keys=len(failed),
        )
        if failed:
            failed_codes = {res[i] for i in failed}
            logger.warning(
                "Partial-tail put failed for request %s: %d/%d keys failed (codes=%s)",
                req_meta.req_id,
                len(failed),
                len(keys),
                failed_codes,
            )
            if MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes:
                self._mark_request_skipped_for_pressure(req_meta)
            return False

        if self._clear_store_pressure():
            logger.info(
                "Mooncake CPU/disk offloading pressure cleared after a "
                "successful partial-tail batch"
            )
        return True

    def _handle_request(self, req_meta: ReqMeta):
        # The single `finally` is the only way out, so the scheduler releases
        # this job's GPU block references however the job ends.
        save_completed = False
        token_len = 0
        req_id = req_meta.req_id
        event_token_ids = req_meta.token_ids
        token_ids_start = req_meta.token_ids_start
        try:
            # Cache hits are always a multiple of ``lcm_block_size`` tokens,
            # which is also ``store_mask``'s precondition.
            lcm_block_size = self.coord.lcm_block_size
            token_len = req_meta.token_len_chunk // lcm_block_size * lcm_block_size
            block_ids_per_group = req_meta.block_ids
            current_event = req_meta.current_event

            if not self.is_live_store_job(req_meta):
                return

            if self.enable_kv_event:
                retry_token_ids = self._get_retry_token_ids(req_meta)
                if retry_token_ids is not None and event_token_ids is not None:
                    retry_start, retry_ids = retry_token_ids
                    if retry_start + len(retry_ids) == token_ids_start:
                        event_token_ids = retry_ids + event_token_ids
                        token_ids_start = retry_start

            if self._should_skip_request(req_id):
                logger.debug(
                    "Skipping Mooncake store for request %s while CPU/disk "
                    "offloading is under pressure",
                    req_id,
                )
                return

            # Offload the sub-block partial tail (independent of the normal
            # block-aligned save, which may be skipped this step).
            if req_meta.partial_tail_offloads is not None and not (
                self._maybe_offload_partial_tail(req_meta)
            ):
                return

            if token_len == 0:
                return

            # Resume from where this rank left off; only the new suffix is saved.
            save_start = self._saved_offset.get(req_id, 0)

            # Within each lcm region only per-spec relevant chunks are loaded
            # (e.g., SWA or linear attn), so mask out irrelevant chunks
            store_masks = self.coord.store_mask(
                token_len,
                save_start,
                num_prompt_tokens=req_meta.num_prompt_tokens,
            )

            starts: list[int] = []
            ends: list[int] = []
            keys: list[str] = []
            chunk_hashes: list[BlockHash] = []
            kv_event_block_hashes: list[BlockHash] = []
            group_indices: list[int] = []
            for g_idx, db in enumerate(self.token_databases):
                # Rotate the stride phase per group to balance load across ranks.
                put_step = self.group_put_steps[g_idx]
                put_step_rank = (self.tp_rank + g_idx) % put_step
                for start, end, block_hash in db.process_tokens(
                    token_len,
                    req_meta.block_hashes,
                    mask_num=save_start,
                    chunk_mask=store_masks[g_idx],
                    put_step=put_step,
                    put_step_rank=put_step_rank,
                ):
                    block_idx = start // db.block_size
                    group_blocks = block_ids_per_group[g_idx]
                    if block_idx >= len(group_blocks) or (
                        group_blocks[block_idx] == NULL_BLOCK_ID
                    ):
                        logger.debug(
                            "Skipping unavailable Mooncake store source block "
                            "(req=%s, group=%d, block=%d)",
                            req_id,
                            g_idx,
                            block_idx,
                        )
                        continue
                    starts.append(start)
                    ends.append(end)
                    keys.append(db.key_for(block_hash))
                    chunk_hashes.append(block_hash)
                    if self.enable_kv_event:
                        kv_event_block_hashes.append(block_hash)
                    group_indices.append(g_idx)

            if not keys:
                self._record_saved(req_meta, token_len)
                save_completed = True
                return

            # Check which blocks already exist (dedup)
            save_exists_start = time.perf_counter()
            try:
                exists_states = self.store.batch_is_exist(keys)
            except Exception:
                self._record_operation(
                    "save_exists",
                    save_exists_start,
                    len(keys),
                    status="error",
                    num_failed_keys=len(keys),
                )
                raise
            if len(exists_states) != len(keys):
                self._record_operation(
                    "save_exists",
                    save_exists_start,
                    len(keys),
                    status="error",
                    num_failed_keys=len(keys),
                )
                logger.warning(
                    "Mooncake store existence check returned %d results for "
                    "%d keys (req=%s, first_key=%s, last_key=%s)",
                    len(exists_states),
                    len(keys),
                    req_id,
                    keys[0],
                    keys[-1],
                )
                return
            self._record_operation(
                "save_exists",
                save_exists_start,
                len(keys),
            )
            missing_indices = [
                i for i, exists in enumerate(exists_states) if exists != 1
            ]

            if self.semantic_region_dbs_by_group is None:
                if not missing_indices:
                    self._record_saved(req_meta, token_len)
                    save_completed = True
                    return

                if len(missing_indices) != len(keys):
                    starts = [starts[i] for i in missing_indices]
                    ends = [ends[i] for i in missing_indices]
                    keys = [keys[i] for i in missing_indices]
                    chunk_hashes = [chunk_hashes[i] for i in missing_indices]
                    if self.enable_kv_event:
                        kv_event_block_hashes = [
                            kv_event_block_hashes[i] for i in missing_indices
                        ]
                    group_indices = [group_indices[i] for i in missing_indices]

            group_ids = (
                [
                    _make_mooncake_group_id(
                        self.token_databases[g_idx].metadata,
                        key.rsplit("@", 1)[-1],
                    )
                    for key, g_idx in zip(keys, group_indices, strict=True)
                ]
                if self.enable_group_semantics and self.supports_group_ids
                else None
            )

            logger.debug(
                "Storing KV cache for %d blocks (groups=%s) for request %s",
                len(keys),
                set(group_indices),
                req_id,
            )

            addrs: list[list[int]] = []
            sizes: list[list[int]] = []
            stored_events: list[BlockStored] = []
            chunks_per_group: list[list[tuple[int, int]]] = [
                [] for _ in self.token_databases
            ]
            for start, end, g_idx in zip(starts, ends, group_indices, strict=True):
                chunks_per_group[g_idx].append((start, end))
            for g_idx, chunks in enumerate(chunks_per_group):
                if not chunks:
                    continue
                db = self.token_databases[g_idx]
                group_addrs, group_sizes, _ = db.prepare_values(
                    chunks, block_ids_per_group[g_idx]
                )
                addrs.extend(group_addrs)
                sizes.extend(group_sizes)

            if self.enable_kv_event:
                new_block_hashes = [
                    maybe_convert_block_hash(bh) for bh in kv_event_block_hashes
                ]
                token_ids_end = token_ids_start + len(event_token_ids or ())

            for idx, (s, e, g_idx) in enumerate(
                zip(starts, ends, group_indices, strict=True)
            ):
                db = self.token_databases[g_idx]
                if self.enable_kv_event:
                    token_ids = (
                        event_token_ids[s - token_ids_start : e - token_ids_start]
                        if event_token_ids is not None
                        and token_ids_start <= s
                        and e <= token_ids_end
                        else []
                    )
                    stored_event = BlockStored(
                        block_hashes=[new_block_hashes[idx]],
                        # Derive the direct predecessor from the unfiltered
                        # request chain. Adjacent PUTs need not be adjacent in
                        # that chain after Store dedup, masks, or TP striding.
                        parent_block_hash=(
                            maybe_convert_block_hash(
                                req_meta.block_hashes[s // db.hash_block_size - 1]
                            )
                            if s > 0
                            else None
                        ),
                        token_ids=token_ids,
                        block_size=db.block_size,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                        group_idx=g_idx,
                    )
                    stored_events.append(stored_event)

            if self.semantic_region_dbs_by_group is not None:
                key_block_ids = [
                    block_ids_per_group[g_idx][
                        start // self.token_databases[g_idx].block_size
                    ]
                    for start, g_idx in zip(starts, group_indices, strict=True)
                ]
                semantic_results = self._put_semantic_entries(
                    keys,
                    chunk_hashes,
                    group_indices,
                    key_block_ids,
                    current_event,
                    debug_request_id=req_id,
                    debug_boundaries=ends,
                )
                failed = [i for i, result in enumerate(semantic_results) if not result]
                if self.enable_kv_event and failed:
                    failed_indices = set(failed)
                    stored_events = [
                        event
                        for i, event in enumerate(stored_events)
                        if i not in failed_indices
                    ]
                if not failed:
                    self._record_saved(req_meta, token_len)
                    save_completed = True
                    if self._clear_store_pressure():
                        logger.info(
                            "Mooncake CPU/disk offloading pressure cleared "
                            "after a successful semantic store batch"
                        )
                else:
                    logger.warning(
                        "Semantic Mooncake store failed to commit %d/%d markers "
                        "for request %s",
                        len(failed),
                        len(keys),
                        req_id,
                    )
                if self.enable_kv_event and stored_events:
                    self.update_kv_event(stored_events)
                return

            if current_event is not None:
                current_event.synchronize()
            if self.kda_transport is not None:
                stage_blocks = {}
                for start, g_idx in zip(starts, group_indices, strict=True):
                    db = self.token_databases[g_idx]
                    block_id = block_ids_per_group[g_idx][start // db.block_size]
                    stage_blocks.setdefault(g_idx, []).append(block_id)
                self.kda_transport.stage_group_blocks(stage_blocks)

            if group_ids is not None:
                assert len(group_ids) == len(keys)
                self.replicate_config.group_ids = group_ids

            batch_bytes = _sum_batch_bytes(sizes)
            put_start = time.perf_counter()
            try:
                res = self.store.batch_put_from_multi_buffers(
                    keys,
                    addrs,
                    sizes,
                    self.replicate_config,
                )
                failed = [i for i, v in enumerate(res) if v < 0]
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed else "ok",
                    num_failed_keys=len(failed),
                )
                if failed:
                    failed_codes = set(res[i] for i in failed)
                    if self.enable_kv_event:
                        failed_indices = set(failed)
                        stored_events = [
                            event
                            for i, event in enumerate(stored_events)
                            if i not in failed_indices
                        ]
                    logger.warning(
                        "batch_put failed: %d/%d keys failed "
                        "(codes=%s, batch_bytes=%d, num_keys=%d), "
                        "first_key=%s",
                        len(failed),
                        len(keys),
                        failed_codes,
                        batch_bytes,
                        len(keys),
                        keys[0] if keys else "N/A",
                    )
                    if (
                        MOONCAKE_NO_AVAILABLE_HANDLE in failed_codes
                        and not self._mark_request_skipped_for_pressure(req_meta)
                    ):
                        logger.warning(
                            "Detected Mooncake CPU/disk offloading pressure "
                            "(NO_AVAILABLE_HANDLE); skipping future store "
                            "batches for request %s until a later store "
                            "batch succeeds",
                            req_id,
                        )
                else:
                    self._record_saved(req_meta, token_len)
                    save_completed = True
                    if self._clear_store_pressure():
                        logger.info(
                            "Mooncake CPU/disk offloading pressure cleared "
                            "after a successful store batch"
                        )
            except Exception as e:
                self._record_operation(
                    "save_put",
                    put_start,
                    len(keys),
                    num_bytes=batch_bytes,
                    status="error",
                    num_failed_keys=len(keys),
                )
                logger.error("Failed to put key %s, error: %s", keys, e)
                stored_events.clear()

            if self.enable_kv_event and stored_events:
                self.update_kv_event(stored_events)
        finally:
            if self.enable_kv_event and token_len:
                self._update_retry_token_ids(
                    req_meta,
                    save_completed,
                    token_ids_start,
                    event_token_ids,
                )
            self.finish_store_job(req_meta)
            self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    """Background thread for loading KV cache blocks from the store."""

    def __init__(
        self,
        store: Any,
        coord: MooncakeStoreCoordinator,
        token_databases: list[ChunkedTokenDatabase],
        block_size: int,
        tp_rank: int,
        ready_event: threading.Event,
        disk_offload_buffer_budget_bytes: int | None = None,
        record_operation: Callable[..., None] | None = None,
        request_queue: queue.Queue[Any] | None = None,
        kda_transport: KDATargetStateTransport | None = None,
        semantic_region_dbs_by_group: Sequence[Sequence[ChunkedTokenDatabase]]
        | None = None,
    ):
        super().__init__(
            store,
            token_databases,
            block_size,
            tp_rank,
            ready_event,
            name="KVCacheStoreRecvingThread",
            record_operation=record_operation,
            request_queue=request_queue,
        )
        # _invalid_block_ids can be access by both the Worker and RecvingThread
        self._invalid_block_ids_lock = threading.Lock()
        self._invalid_block_ids: set[int] = set()
        self.disk_offload_buffer_budget_bytes = disk_offload_buffer_budget_bytes
        self.usable_disk_offload_buffer_budget_bytes = (
            None
            if disk_offload_buffer_budget_bytes is None
            else _get_usable_disk_offload_buffer_budget_bytes(
                disk_offload_buffer_budget_bytes
            )
        )
        self.coord = coord
        self.kda_transport = kda_transport
        self.semantic_region_dbs_by_group = semantic_region_dbs_by_group

    def _add_load_error_block_ids(self, block_ids: list[int]) -> None:
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(block_ids)

    def get_and_clear_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid_block_ids = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid_block_ids

    def _handle_request(self, req_meta: ReqMeta):
        if self.semantic_region_dbs_by_group is not None:
            self._handle_semantic_request(req_meta)
            return
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )

        # Skip chunks the consumer's per-group spec wouldn't populate
        # locally (e.g. SWA pre-window) even if the producer stored them.
        load_mask_per_group = self.coord.load_mask(req_meta.block_hashes, token_len)

        addr_list: list[list[int]] = []
        size_list: list[list[int]] = []
        key_list: list[str] = []
        block_id_list: list[int] = []
        group_index_list: list[int] = []
        for g_idx, db in enumerate(self.token_databases):
            mask = load_mask_per_group[g_idx]
            chunks: list[tuple[int, int]] = []
            for start, end, block_hash in db.process_tokens(
                token_len, req_meta.block_hashes, mask_num
            ):
                chunk_idx = start // db.block_size
                if chunk_idx >= len(mask) or not mask[chunk_idx]:
                    continue
                key_list.append(db.key_for(block_hash))
                chunks.append((start, end))
                group_index_list.append(g_idx)
            g_addrs, g_sizes, g_block_ids = db.prepare_values(
                chunks, req_meta.block_ids[g_idx]
            )
            addr_list.extend(g_addrs)
            size_list.extend(g_sizes)
            block_id_list.extend(g_block_ids)

        # Rotate aligned lists by tp_rank for load balancing.
        rotation = self.tp_rank % len(key_list)
        key_list_c = _rotate_list(key_list, rotation)
        addr_list_c = _rotate_list(addr_list, rotation)
        size_list_c = _rotate_list(size_list, rotation)
        block_id_list_c = _rotate_list(block_id_list, rotation)
        group_index_list_c = _rotate_list(group_index_list, rotation)

        load_batches = [
            (
                key_list_c,
                addr_list_c,
                size_list_c,
                block_id_list_c,
                group_index_list_c,
            )
        ]
        if self.usable_disk_offload_buffer_budget_bytes is not None:
            total_staging_bytes = sum(
                _estimate_disk_offload_staging_bytes(size) for size in size_list_c
            )
            if total_staging_bytes > self.usable_disk_offload_buffer_budget_bytes:
                assert self.disk_offload_buffer_budget_bytes is not None
                split_batches, oversized_key = _split_disk_offload_load_batches(
                    key_list_c,
                    addr_list_c,
                    size_list_c,
                    self.usable_disk_offload_buffer_budget_bytes,
                    self.disk_offload_buffer_budget_bytes,
                )
                if oversized_key is not None:
                    oversized_key_index = key_list_c.index(oversized_key)
                    # Mark every block: we skip the whole request, and the
                    # tp_rank rotation means oversized_key isn't necessarily
                    # the first block in the request's original order.
                    self._add_load_error_block_ids(block_id_list_c)
                    oversized_key_bytes = _estimate_disk_offload_staging_bytes(
                        size_list_c[oversized_key_index]
                    )
                    logger.warning(
                        "Skipping Mooncake load for request %s because key %s "
                        "requires %d staging bytes, exceeding budget %d",
                        req_id,
                        oversized_key,
                        oversized_key_bytes,
                        self.disk_offload_buffer_budget_bytes,
                    )
                    self.set_finished_request(req_id)
                    self.request_queue.task_done()
                    return
                load_batches = []
                block_id_offset = 0
                for batch_keys, batch_addrs, batch_sizes in split_batches:
                    next_block_id_offset = block_id_offset + len(batch_keys)
                    batch_block_ids = block_id_list_c[
                        block_id_offset:next_block_id_offset
                    ]
                    batch_group_indices = group_index_list_c[
                        block_id_offset:next_block_id_offset
                    ]
                    load_batches.append(
                        (
                            batch_keys,
                            batch_addrs,
                            batch_sizes,
                            batch_block_ids,
                            batch_group_indices,
                        )
                    )
                    block_id_offset = next_block_id_offset

        current_batch_keys: list[str] = key_list_c
        current_batch_block_ids: list[int] = block_id_list_c
        batch_bytes = 0
        loaded_blocks_by_group: dict[int, list[int]] = {}
        try:
            for (
                batch_keys,
                batch_addrs,
                batch_sizes,
                batch_block_ids,
                batch_group_indices,
            ) in load_batches:
                current_batch_keys = batch_keys
                current_batch_block_ids = batch_block_ids
                batch_bytes = _sum_batch_bytes(batch_sizes)
                tiers_by_key: dict[str, str] | None = None
                if envs.VLLM_MOONCAKE_STORE_TIER_LOG:
                    tiers_by_key = _get_replica_tiers_by_key(self.store, batch_keys)
                # Reset so the recorded RPC duration excludes tier lookup.
                load_get_start = time.perf_counter()
                res = self.store.batch_get_into_multi_buffers(
                    batch_keys, batch_addrs, batch_sizes
                )
                if tiers_by_key is not None:
                    _log_mooncake_load_tier_summary(
                        req_id, batch_keys, res, tiers_by_key
                    )
                failed = [
                    (key, value, block_id)
                    for key, value, block_id in zip(
                        batch_keys, res, batch_block_ids, strict=True
                    )
                    if value < 0
                ]
                self._record_operation(
                    "load_get",
                    load_get_start,
                    len(batch_keys),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed else "ok",
                    num_failed_keys=len(failed),
                )
                for group_index, block_id, result in zip(
                    batch_group_indices, batch_block_ids, res, strict=True
                ):
                    if result >= 0:
                        loaded_blocks_by_group.setdefault(group_index, []).append(
                            block_id
                        )
                if failed:
                    self._add_load_error_block_ids(
                        [block_id for _, _, block_id in failed]
                    )
                    logger.warning(
                        "Failed to get %d Mooncake keys from sub-batch "
                        "(batch_keys=%d, first_failures=%s)",
                        len(failed),
                        len(batch_keys),
                        [(key, value) for key, value, _ in failed[:3]],
                    )
                    break
        except Exception as e:
            self._add_load_error_block_ids(current_batch_block_ids)
            self._record_operation(
                "load_get",
                load_get_start,
                len(current_batch_keys),
                num_bytes=batch_bytes,
                status="error",
                num_failed_keys=len(current_batch_keys),
            )
            logger.warning(
                "Failed to get Mooncake sub-batch %s, error: %s",
                current_batch_keys[:3],
                e,
            )

        if self.kda_transport is not None and loaded_blocks_by_group:
            self.kda_transport.materialize_group_blocks(loaded_blocks_by_group)

        self.set_finished_request(req_id)
        self.request_queue.task_done()

    def _handle_semantic_request(self, req_meta: ReqMeta) -> None:
        """Load per-layer target-state objects and materialize local KDA pages."""
        assert self.semantic_region_dbs_by_group is not None
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )
        load_mask_per_group = self.coord.load_mask(req_meta.block_hashes, token_len)

        # An entry is one scheduler-visible (group, chunk) restore. Each entry
        # fans out to all semantic regions owned by this PP worker.
        entry_groups: list[int] = []
        entry_blocks: list[int] = []
        entry_expected: list[int] = []
        entry_boundaries: list[int] = []
        entry_hashes: list[BlockHash] = []
        keys: list[str] = []
        addrs: list[list[int]] = []
        sizes: list[list[int]] = []
        key_entry_indices: list[int] = []
        for group_index, marker_db in enumerate(self.token_databases):
            mask = load_mask_per_group[group_index]
            for start, end, block_hash in marker_db.process_tokens(
                token_len,
                req_meta.block_hashes,
                mask_num,
            ):
                chunk_index = start // marker_db.block_size
                if chunk_index >= len(mask) or not mask[chunk_index]:
                    continue
                block_id = req_meta.block_ids[group_index][chunk_index]
                entry_index = len(entry_groups)
                entry_groups.append(group_index)
                entry_blocks.append(block_id)
                entry_boundaries.append(end)
                entry_hashes.append(block_hash)
                region_dbs = self.semantic_region_dbs_by_group[group_index]
                entry_expected.append(len(region_dbs))
                for region_db in region_dbs:
                    region_addrs, region_sizes = region_db.prepare_value_for_block(
                        block_id
                    )
                    keys.append(region_db.key_for(block_hash))
                    addrs.append(region_addrs)
                    sizes.append(region_sizes)
                    key_entry_indices.append(entry_index)

        if not keys:
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return

        rotation = self.tp_rank % len(keys)
        keys = _rotate_list(keys, rotation)
        addrs = _rotate_list(addrs, rotation)
        sizes = _rotate_list(sizes, rotation)
        key_entry_indices = _rotate_list(key_entry_indices, rotation)

        # Markers protect against incomplete writes. They cannot protect
        # against a later partial eviction, so validate the immutable region
        # set immediately before GET. One missing region invalidates the whole
        # logical cache load; mixing restored and recomputed state is unsafe.
        preflight_start = time.perf_counter()
        try:
            visible = self.store.batch_is_exist(keys)
        except Exception as error:
            self._record_operation(
                "load_region_preflight",
                preflight_start,
                len(keys),
                status="error",
                num_failed_keys=len(keys),
            )
            self._add_load_error_block_ids(entry_blocks)
            logger.warning(
                "Semantic Mooncake load preflight failed for request %s: %s",
                req_id,
                error,
            )
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return
        if len(visible) != len(keys):
            self._record_operation(
                "load_region_preflight",
                preflight_start,
                len(keys),
                status="error",
                num_failed_keys=len(keys),
            )
            self._add_load_error_block_ids(entry_blocks)
            logger.warning(
                "Semantic Mooncake load preflight returned %d results for %d "
                "keys (req=%s)",
                len(visible),
                len(keys),
                req_id,
            )
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return
        missing_preflight = [i for i, exists in enumerate(visible) if exists != 1]
        self._record_operation(
            "load_region_preflight",
            preflight_start,
            len(keys),
            status="partial_failure" if missing_preflight else "ok",
            num_failed_keys=len(missing_preflight),
        )
        if missing_preflight:
            self._add_load_error_block_ids(entry_blocks)
            first_index = missing_preflight[0]
            logger.warning(
                "Semantic Mooncake load preflight rejected the logical cache: "
                "%d/%d regions missing (req=%s, first_key=%s, entry=%d)",
                len(missing_preflight),
                len(keys),
                req_id,
                keys[first_index],
                key_entry_indices[first_index],
            )
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return

        batches: list[tuple[list[str], list[list[int]], list[list[int]], list[int]]] = [
            (keys, addrs, sizes, key_entry_indices)
        ]
        if self.usable_disk_offload_buffer_budget_bytes is not None:
            total_staging_bytes = sum(
                _estimate_disk_offload_staging_bytes(size) for size in sizes
            )
            if total_staging_bytes > self.usable_disk_offload_buffer_budget_bytes:
                assert self.disk_offload_buffer_budget_bytes is not None
                split_batches, oversized_key = _split_disk_offload_load_batches(
                    keys,
                    addrs,
                    sizes,
                    self.usable_disk_offload_buffer_budget_bytes,
                    self.disk_offload_buffer_budget_bytes,
                )
                if oversized_key is not None:
                    self._add_load_error_block_ids(entry_blocks)
                    logger.warning(
                        "Skipping semantic Mooncake load for request %s: key %s "
                        "exceeds the disk staging budget",
                        req_id,
                        oversized_key,
                    )
                    self.set_finished_request(req_id)
                    self.request_queue.task_done()
                    return
                batches = []
                offset = 0
                for batch_keys, batch_addrs, batch_sizes in split_batches:
                    end = offset + len(batch_keys)
                    batches.append(
                        (
                            batch_keys,
                            batch_addrs,
                            batch_sizes,
                            key_entry_indices[offset:end],
                        )
                    )
                    offset = end

        success_counts = [0] * len(entry_groups)
        failed_entries: set[int] = set()
        first_failures: list[tuple[str, int | str]] = []
        for batch_keys, batch_addrs, batch_sizes, batch_entry_indices in batches:
            batch_bytes = _sum_batch_bytes(batch_sizes)
            tiers_by_key: dict[str, str] | None = None
            if envs.VLLM_MOONCAKE_STORE_TIER_LOG:
                tiers_by_key = _get_replica_tiers_by_key(self.store, batch_keys)
            load_start = time.perf_counter()
            try:
                results = self.store.batch_get_into_multi_buffers(
                    batch_keys,
                    batch_addrs,
                    batch_sizes,
                )
            except Exception as error:
                failed_entries.update(batch_entry_indices)
                first_failures.extend(
                    (key, f"{type(error).__name__}: {error}")
                    for key in batch_keys[: 3 - len(first_failures)]
                )
                self._record_operation(
                    "load_region_get",
                    load_start,
                    len(batch_keys),
                    num_bytes=batch_bytes,
                    status="error",
                    num_failed_keys=len(batch_keys),
                )
                logger.warning(
                    "Failed to get semantic Mooncake sub-batch %s, error: %s",
                    batch_keys[:3],
                    error,
                )
                continue
            if len(results) != len(batch_keys):
                failed_entries.update(batch_entry_indices)
                if len(first_failures) < 3:
                    first_failures.append(
                        (
                            batch_keys[0],
                            f"result_count={len(results)}/{len(batch_keys)}",
                        )
                    )
                self._record_operation(
                    "load_region_get",
                    load_start,
                    len(batch_keys),
                    num_bytes=batch_bytes,
                    status="error",
                    num_failed_keys=len(batch_keys),
                )
                logger.warning(
                    "Semantic Mooncake get returned %d results for %d keys "
                    "(req=%s, first_key=%s, last_key=%s)",
                    len(results),
                    len(batch_keys),
                    req_id,
                    batch_keys[0],
                    batch_keys[-1],
                )
                continue
            if tiers_by_key is not None:
                _log_mooncake_load_tier_summary(
                    req_id,
                    batch_keys,
                    results,
                    tiers_by_key,
                )
            failed_count = 0
            for result_index, (result, entry_index) in enumerate(
                zip(results, batch_entry_indices, strict=True)
            ):
                if result < 0:
                    failed_entries.add(entry_index)
                    failed_count += 1
                    if len(first_failures) < 3:
                        first_failures.append((batch_keys[result_index], result))
                else:
                    success_counts[entry_index] += 1
            self._record_operation(
                "load_region_get",
                load_start,
                len(batch_keys),
                num_bytes=batch_bytes,
                status="partial_failure" if failed_count else "ok",
                num_failed_keys=failed_count,
            )

        loaded_blocks_by_group: dict[int, list[int]] = {}
        for entry_index, (group_index, block_id, expected) in enumerate(
            zip(entry_groups, entry_blocks, entry_expected, strict=True)
        ):
            if entry_index in failed_entries or success_counts[entry_index] != expected:
                failed_entries.add(entry_index)
                continue
            if expected:
                loaded_blocks_by_group.setdefault(group_index, []).append(block_id)

        if failed_entries:
            # A semantic load is one logical cache transaction. Mooncake's
            # multi-key GET is not atomic: a key can disappear after preflight
            # while sibling regions/chunks still load successfully. Never
            # materialize that partial snapshot or mix it with recomputation.
            # Invalidating every advertised block makes the scheduler replay the
            # complete logical prefix from a known-good state.
            self._add_load_error_block_ids(entry_blocks)
            logger.warning(
                "Semantic Mooncake load failed for %d/%d cache entries; "
                "invalidating all %d logical entries "
                "(req=%s, first_failures=%s)",
                len(failed_entries),
                len(entry_groups),
                len(entry_blocks),
                req_id,
                first_failures,
            )
        elif self.kda_transport is not None and loaded_blocks_by_group:
            self.kda_transport.materialize_group_blocks(loaded_blocks_by_group)
            for entry_index, (group_index, block_id, boundary, block_hash) in enumerate(
                zip(
                    entry_groups,
                    entry_blocks,
                    entry_boundaries,
                    entry_hashes,
                    strict=True,
                )
            ):
                if entry_index in failed_entries or boundary != token_len:
                    continue
                self.kda_transport.debug_entry_checksums(
                    "STORE_LOAD",
                    req_id,
                    boundary,
                    group_index,
                    block_id,
                    bytes(block_hash),
                )
            self.kda_transport.debug_checksums(
                "D_STORE",
                req_id,
                loaded_blocks_by_group,
            )

        self.set_finished_request(req_id)
        self.request_queue.task_done()


# ============================================================
# Store Worker
# ============================================================


class MooncakeStoreWorker:
    """Worker-side component for MooncakeStoreConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        try:
            from mooncake.store import (  # type: ignore
                MooncakeDistributedStore,
                ReplicateConfig,
            )
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/"
                "en/build.md to run vLLM with MooncakeStoreConnector."
            ) from e

        model_config = vllm_config.model_config
        self.model_config = model_config
        get_total_num_hidden_layers = getattr(
            model_config,
            "get_total_num_hidden_layers",
            None,
        )
        self.target_num_layers = (
            get_total_num_hidden_layers()
            if callable(get_total_num_hidden_layers)
            else 2**31 - 1
        )
        parallel_config = vllm_config.parallel_config

        self.dp_rank = parallel_config.data_parallel_index
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0

        assert vllm_config.kv_transfer_config is not None
        kv_role = vllm_config.kv_transfer_config.kv_role
        assert kv_role is not None
        self.kv_role = kv_role
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._kda_transport_enabled = kda_target_state_transport_enabled(extra_config)
        self._dspark_context_transport_enabled = dspark_context_kv_transport_enabled(
            extra_config
        )
        spec_config = vllm_config.speculative_config
        if self._dspark_context_transport_enabled and (
            spec_config is None
            or not spec_config.use_dspark()
            or not self._kda_transport_enabled
        ):
            raise ValueError(
                "MooncakeStore DSpark context transport requires DSpark and "
                "kda_transport_policy=target_state_v1."
            )
        self._kda_transport: KDATargetStateTransport | None = None
        self.can_put = self.kv_role in ("kv_producer", "kv_both") or (
            extra_config.get("save_decode_cache", False)
        )
        self.load_async = extra_config.get("load_async", True)
        # Mirrors MooncakeStoreConnector._capacity_only.
        self._capacity_only = (
            self.kv_role == "kv_consumer"
            and not extra_config.get("enable_lookup", True)
            and not self.can_put
        )
        self.cache_config = vllm_config.cache_config
        self.block_size, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        if self._kda_transport_enabled:
            transport_block_size = extra_config.get("kda_transport_block_size")
            if (
                not isinstance(transport_block_size, int)
                or isinstance(transport_block_size, bool)
                or transport_block_size <= 0
            ):
                raise ValueError(
                    "KDA target-state Store transport requires a positive "
                    "kda_transport_block_size"
                )
            if self.cache_config.block_size != transport_block_size:
                raise ValueError(
                    "KDA target-state Store transport requires the local HMA "
                    f"block size to be {transport_block_size}, got "
                    f"{self.cache_config.block_size}"
                )
            if self.cache_config.mamba_block_size != transport_block_size:
                raise ValueError(
                    "KDA target-state Store transport requires matching Mamba "
                    f"checkpoint boundaries ({transport_block_size}), got "
                    f"{self.cache_config.mamba_block_size}"
                )
        self.num_layers = model_config.get_num_layers(parallel_config)

        self.num_kv_head = model_config.get_total_num_kv_heads()

        # Initialize MooncakeDistributedStore with its own TransferEngine
        store_config = MooncakeStoreConfig.load_from_config()
        self.store = MooncakeDistributedStore()
        local_ip = get_ip()
        local_hostname = rdma_utils.get_requester_local_hostname(local_ip)
        setup_kwargs: dict[str, str] = {}
        if store_config.tenant_id != DEFAULT_TENANT_ID:
            setup_kwargs["tenant_id"] = store_config.tenant_id
        ret = self.store.setup(
            local_hostname,
            store_config.metadata_server,
            store_config.global_segment_size,
            store_config.local_buffer_size,
            store_config.protocol,
            store_config.device_name,
            store_config.master_server_address,
            **setup_kwargs,
        )
        if ret != 0:
            msg = "Initialize MooncakeDistributedStore failed."
            logger.error(msg)
            raise RuntimeError(msg)

        preferred_segment = rdma_utils.get_configured_preferred_segment(extra_config)
        self.preferred_segment = preferred_segment
        self.store_replicate_config = ReplicateConfig()
        self.enable_group_semantics = (
            str(extra_config.get("enable_group_semantics", "False")).strip().lower()
            == "true"
        )
        self._supports_group_ids = _replicate_config_supports_group_ids(
            ReplicateConfig, self.store_replicate_config
        )
        if self.enable_group_semantics and not self._supports_group_ids:
            logger.warning(
                "Mooncake group semantics is enabled, but the installed "
                "Mooncake package does not support ReplicateConfig.group_ids. "
                "Falling back to the existing batch_put_from_multi_buffers path."
            )
        if preferred_segment is not None:
            self.store_replicate_config.preferred_segment = preferred_segment

        logger.info(
            "Mooncake mode=%s (global_segment_size=%d, local_buffer_size=%d, "
            "preferred_segment=%s, enable_offload=%s, tenant_id=%s)",
            store_config.mode,
            store_config.global_segment_size,
            store_config.local_buffer_size,
            preferred_segment or "<none>",
            store_config.enable_offload,
            store_config.tenant_id,
        )
        if store_config.mode == "embedded":
            if store_config.enable_offload and preferred_segment is None:
                logger.warning(
                    "enable_offload is set in embedded mode without "
                    "preferred_segment; SSD tier will only see puts that "
                    "happen to land on the owner segment."
                )
            if preferred_segment is not None:
                logger.warning(
                    "preferred_segment=%s with mode=embedded: rank-"
                    "contributed segments will be idle.",
                    preferred_segment,
                )
        elif (
            store_config.mode == "standalone-store" and not store_config.enable_offload
        ):
            logger.warning(
                "standalone-store mode without enable_offload: large prefills "
                "may exceed the owner DirectIO budget."
            )

        self.disk_offload_buffer_budget_bytes = (
            DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES
            if store_config.enable_offload
            else None
        )

        # Start lookup server on rank 0 for scheduler-side prefix queries
        self.lookup_server: LookupKeyServer | None = None
        if vllm_config.parallel_config.rank == 0:
            self.lookup_server = LookupKeyServer(self, vllm_config)

        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        # Pool of load-receive threads
        self.kv_recv_threads: list[KVCacheStoreRecvingThread] = []
        self.num_recv_threads = max(1, envs.VLLM_MOONCAKE_LOAD_RECV_THREADS)
        self.recv_request_queue: queue.Queue[ReqMeta] = queue.Queue()
        # Connector metadata remains bound while an async load is pending and
        # get_finished() may be polled many times. Submit each request exactly
        # once until the scheduler retires or preempts it.
        self._submitted_load_req_ids: set[str] = set()
        self.finished_store_req: set[str] = set()
        self._kv_connector_stats_lock = threading.Lock()
        self.kv_connector_stats = MooncakeStoreConnectorStats()

        self._kv_cache_config = kv_cache_config
        self.token_dbs: list[ChunkedTokenDatabase] = []

        # a capacity-only instance does not need below utils
        if self._capacity_only:
            logger.info(
                "Mooncake store in capacity-only mode: segment mounted "
                "(global_segment_size=%d), KV transfer disabled.",
                store_config.global_segment_size,
            )
            return

        # Single-group + PCP/DCP > 1: scale the lone group's spec.block_size to
        # self.block_size (= scheduler_block_size) so the coordinator's
        # ``block_size % hash_block_size == 0`` invariant holds.
        groups = store_effective_kv_cache_groups(
            kv_cache_config.kv_cache_groups, self.dcp_size
        )
        if len(groups) == 1 and groups[0].kv_cache_spec.block_size != self.block_size:
            g = groups[0]
            groups = [
                dataclasses.replace(
                    g,
                    kv_cache_spec=dataclasses.replace(
                        g.kv_cache_spec, block_size=self.block_size
                    ),
                )
            ]
        self._kv_cache_groups: list[KVCacheGroupSpec] = groups
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        use_eagle = bool(
            spec_cfg.use_eagle()
            if spec_cfg is not None and callable(getattr(spec_cfg, "use_eagle", None))
            else False
        )
        self.coord = MooncakeStoreCoordinator(
            self._kv_cache_groups,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
            use_eagle=use_eagle,
            retention_interval=kv_cache_config.prefix_cache_retention_interval,
        )
        # One ChunkedTokenDatabase per group; addresses populated in
        # register_kv_caches once the kv-cache layout is known. Each group's
        # key namespace is its TP shard id: ranks holding identical bytes
        # (MLA / shared GQA KV heads) share a namespace, TP-sharded Mamba
        # state gets one namespace per rank.
        cache_prefix = str(extra_config.get("cache_prefix", ""))
        if self._kda_transport_enabled:
            cache_prefix = (
                f"{cache_prefix}:{KDA_TARGET_STATE_TRANSPORT}"
                if cache_prefix
                else KDA_TARGET_STATE_TRANSPORT
            )
        if self._dspark_context_transport_enabled:
            cache_prefix = (
                f"{cache_prefix}:{DSPARK_CONTEXT_KV_TRANSPORT}"
                if cache_prefix
                else DSPARK_CONTEXT_KV_TRANSPORT
            )
        metadata = KeyMetadata(
            model_name=model_config.model.rstrip("/").split("/")[-1],
            tp_rank=self.tp_rank,
            pcp_rank=self.pcp_rank,
            dcp_rank=self.dcp_rank,
            pp_rank=self.pp_rank,
            cache_prefix=cache_prefix,
        )
        self._semantic_commit_metadata = dataclasses.replace(
            metadata,
            group_id=-1,
            region_id=SEMANTIC_COMMIT_REGION_ID,
        )
        self._group_tp_replication_factors: tuple[int, ...] = (
            self._compute_group_tp_replication_factors()
        )
        self.token_dbs = [
            ChunkedTokenDatabase(
                dataclasses.replace(
                    metadata,
                    group_id=g_idx,
                    tp_rank=self.tp_rank // self._group_tp_replication_factors[g_idx],
                ),
                g.kv_cache_spec.block_size,
                hash_block_size=self.hash_block_size,
            )
            for g_idx, g in enumerate(self._kv_cache_groups)
        ]
        self._init_lookup_key_prefixes()

    def _spec_tp_replication_factor(self, spec: KVCacheSpec) -> int:
        if self.dcp_size > 1:
            return 1
        inner_specs = (
            tuple(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )
        # Any rank-specific state makes the whole packed value rank-specific.
        if any(isinstance(inner, MambaSpec) for inner in inner_specs):
            return 1
        # A pure MLA packed value is replicated on every TP rank.
        if all(
            isinstance(inner, (MLAAttentionSpec, SlidingWindowMLASpec))
            for inner in inner_specs
        ):
            return self.tp_size
        return max(1, self.tp_size // self.num_kv_head)

    def _compute_group_tp_replication_factors(self) -> tuple[int, ...]:
        """Return the number of byte-identical TP replicas per cache group.

        DCP and Mamba use 1; MLA uses ``tp_size``; GQA uses
        ``tp_size // num_kv_head``.
        """
        return tuple(
            self._spec_tp_replication_factor(group.kv_cache_spec)
            for group in self._kv_cache_groups
        )

    def _init_lookup_key_prefixes(self) -> None:
        def rank_namespaces(factor: int) -> tuple[tuple[int, int, int, int], ...]:
            if self.dcp_size > 1:
                # DCP is a TP subdivision: dcp_rank == tp_rank % dcp_size.
                return tuple(
                    (tp_rank, pcp_rank, tp_rank % self.dcp_size, pp_rank)
                    for pcp_rank in range(self.pcp_size)
                    for tp_rank in range(self.tp_size)
                    for pp_rank in range(self.pp_size)
                )
            return tuple(
                (shard_rank, pcp_rank, 0, pp_rank)
                for pcp_rank in range(self.pcp_size)
                for shard_rank in range(self.tp_size // factor)
                for pp_rank in range(self.pp_size)
            )

        self._lookup_key_prefixes = tuple(
            tuple(
                PoolKey.build_prefix(
                    db.metadata,
                    tp_rank=tp_rank,
                    pcp_rank=pcp_rank,
                    dcp_rank=dcp_rank,
                    pp_rank=pp_rank,
                )
                for tp_rank, pcp_rank, dcp_rank, pp_rank in rank_namespaces(
                    self._group_tp_replication_factors[g_idx]
                )
            )
            for g_idx, db in enumerate(self.token_dbs)
        )
        commit_metadata = getattr(
            self,
            "_semantic_commit_metadata",
            dataclasses.replace(
                self.token_dbs[0].metadata,
                group_id=-1,
                region_id=SEMANTIC_COMMIT_REGION_ID,
            ),
        )
        self._semantic_commit_key_prefixes = tuple(
            PoolKey.build_prefix(
                commit_metadata,
                tp_rank=tp_rank,
                pcp_rank=pcp_rank,
                dcp_rank=(tp_rank % self.dcp_size if self.dcp_size > 1 else 0),
                pp_rank=pp_rank,
            )
            for pcp_rank in range(self.pcp_size)
            for tp_rank in range(self.tp_size)
            for pp_rank in range(self.pp_size)
        )

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register KV cache tensors and start transfer threads."""
        if self._capacity_only:
            return
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        assert self.cache_config.num_gpu_blocks is not None
        self.num_blocks = self.cache_config.num_gpu_blocks

        seen_storage_ptrs: set[int] = set()
        kda_transport_enabled = getattr(self, "_kda_transport_enabled", False)
        kda_transport = getattr(self, "_kda_transport", None)
        if kda_transport_enabled:
            kda_transport = KDATargetStateTransport.create(
                kv_caches,
                self._kv_cache_config,
                target_num_layers=self.target_num_layers,
            )
            self._kda_transport = kda_transport

        def register_storage(tensor: torch.Tensor) -> None:
            storage = tensor.untyped_storage()
            base_addr = storage.data_ptr()
            if base_addr in seen_storage_ptrs:
                return
            seen_storage_ptrs.add(base_addr)
            region_len = storage.nbytes()
            ret = self.store.register_buffer(base_addr, region_len)
            if ret != 0:
                logger.error(
                    "register_buffer failed for addr %#x len %d: %d",
                    base_addr,
                    region_len,
                    ret,
                )

        regions_by_group: list[list[tuple[int, int, int]]]
        semantic_region_specs: list[list[tuple[str, int, int, int]]] | None = None
        if kda_transport_enabled:
            regions_by_group = [[] for _ in self.token_dbs]
            semantic_region_specs = [[] for _ in self.token_dbs]
            layer_to_group = {
                layer_name: group_index
                for group_index, group in enumerate(self._kv_cache_groups)
                for layer_name in group.layer_names
            }
            skipped_decode_local_layers: list[str] = []
            draft_context_regions = 0
            for layer_name, raw_cache in kv_caches.items():
                group_index = layer_to_group.get(layer_name)
                if group_index is None:
                    continue
                try:
                    layer_index = extract_layer_index(layer_name)
                except (AssertionError, IndexError):
                    layer_index = -1
                is_draft_context = layer_index >= getattr(
                    self, "target_num_layers", 2**31 - 1
                )
                if is_draft_context and not self._dspark_context_transport_enabled:
                    # Kimi-K3 DSpark appends its draft MLA layers after the
                    # target range. They are excluded until the connector
                    # explicitly opts into their independent wire schema.
                    skipped_decode_local_layers.append(layer_name)
                    continue
                kda_regions = (
                    kda_transport.regions_for_layer(layer_name)
                    if kda_transport is not None
                    else ()
                )
                if kda_regions:
                    register_storage(raw_cache)
                    occurrences: dict[str, int] = {}
                    for region in kda_regions:
                        register_storage(region.tensor)
                        occurrence = occurrences.get(region.kind, 0)
                        occurrences[region.kind] = occurrence + 1
                        region_id = _make_semantic_region_id(
                            layer_name,
                            region.kind,
                            occurrence,
                        )
                        regions_by_group[group_index].append(
                            (
                                region.tensor.data_ptr(),
                                region.block_stride_bytes,
                                region.content_len_bytes,
                            )
                        )
                        semantic_region_specs[group_index].append(
                            (
                                region_id,
                                region.tensor.data_ptr(),
                                region.block_stride_bytes,
                                region.content_len_bytes,
                            )
                        )
                    continue

                cache = group_kernel_blocks(raw_cache, self.num_blocks)
                register_storage(cache)
                if not is_non_overlapping_and_dense(cache[0]):
                    for head_idx in range(cache.shape[1]):
                        head_cache = cache[:, head_idx]
                        assert is_non_overlapping_and_dense(head_cache[0])
                        regions_by_group[group_index].append(
                            (
                                head_cache.data_ptr(),
                                head_cache.stride(0) * head_cache.element_size(),
                                head_cache[0].numel() * head_cache.element_size(),
                            )
                        )
                        semantic_region_specs[group_index].append(
                            (
                                _make_semantic_region_id(
                                    layer_name,
                                    (
                                        DSPARK_CONTEXT_REGION_KIND
                                        if is_draft_context
                                        else "page"
                                    ),
                                    head_idx,
                                ),
                                head_cache.data_ptr(),
                                head_cache.stride(0) * head_cache.element_size(),
                                head_cache[0].numel() * head_cache.element_size(),
                            )
                        )
                        draft_context_regions += int(is_draft_context)
                else:
                    regions_by_group[group_index].append(
                        (
                            cache.data_ptr(),
                            cache.stride(0) * cache.element_size(),
                            cache[0].numel() * cache.element_size(),
                        )
                    )
                    semantic_region_specs[group_index].append(
                        (
                            _make_semantic_region_id(
                                layer_name,
                                (
                                    DSPARK_CONTEXT_REGION_KIND
                                    if is_draft_context
                                    else "page"
                                ),
                                0,
                            ),
                            cache.data_ptr(),
                            cache.stride(0) * cache.element_size(),
                            cache[0].numel() * cache.element_size(),
                        )
                    )
                    draft_context_regions += int(is_draft_context)
            if self._dspark_context_transport_enabled:
                logger.info(
                    "Included %d DSpark context KV region(s) for PP rank %d in "
                    "semantic Store transport with policy=%s",
                    draft_context_regions,
                    self.pp_rank,
                    DSPARK_CONTEXT_KV_TRANSPORT,
                )
            if skipped_decode_local_layers:
                logger.info(
                    "Excluded %d Decode-local draft KV layers from semantic "
                    "Store transport (first=%s)",
                    len(skipped_decode_local_layers),
                    skipped_decode_local_layers[0],
                )
        else:
            # Preserve the legacy Store value schema unless the explicit KDA
            # target-state policy is enabled.  In that schema every group key
            # carries the same packed collection of registered cache regions.
            seen_region_ptrs: set[int] = set()
            legacy_regions: list[tuple[int, int, int]] = []
            for raw_cache in kv_caches.values():
                cache = group_kernel_blocks(raw_cache, self.num_blocks)
                register_storage(cache)
                storage = cache.untyped_storage()
                base_addr = storage.data_ptr()
                region_len = storage.nbytes()

                if not is_non_overlapping_and_dense(cache[0]):
                    for head_idx in range(cache.shape[1]):
                        head_cache = cache[:, head_idx]
                        assert is_non_overlapping_and_dense(head_cache[0])
                        region_addr = head_cache.data_ptr()
                        if region_addr in seen_region_ptrs:
                            continue
                        seen_region_ptrs.add(region_addr)
                        block_len = head_cache.stride(0) * head_cache.element_size()
                        legacy_regions.append((region_addr, block_len, block_len))
                elif (
                    cache.stride(0) * cache.element_size() * self.num_blocks
                    == region_len
                ):
                    if base_addr in seen_region_ptrs:
                        continue
                    seen_region_ptrs.add(base_addr)
                    block_len = region_len // self.num_blocks
                    legacy_regions.append((base_addr, block_len, block_len))
                else:
                    region_addr = cache.data_ptr()
                    if region_addr in seen_region_ptrs:
                        continue
                    seen_region_ptrs.add(region_addr)
                    block_len = cache.stride(0) * cache.element_size()
                    legacy_regions.append((region_addr, block_len, block_len))
            regions_by_group = [list(legacy_regions) for _ in self.token_dbs]

        logger.info(
            "Registered KV caches: num_groups=%d, num_segments=%d, num_blocks=%d",
            len(self.token_dbs),
            sum(len(regions) for regions in regions_by_group),
            self.num_blocks,
        )

        semantic_region_dbs_by_group: list[list[ChunkedTokenDatabase]] | None = None
        if semantic_region_specs is not None:
            marker_device = next(iter(kv_caches.values())).device
            self._semantic_store_marker = torch.zeros(
                1,
                dtype=torch.int8,
                device=marker_device,
            )
            register_storage(self._semantic_store_marker)
            marker_addr = self._semantic_store_marker.data_ptr()
            semantic_region_dbs_by_group = []
            for group_index, (marker_db, region_specs) in enumerate(
                zip(self.token_dbs, semantic_region_specs, strict=True)
            ):
                # Scheduler lookup checks these one-byte commit objects. The
                # same registered byte can back every marker key.
                marker_db.set_kv_caches_base_addr([marker_addr])
                marker_db.set_block_len([1])
                marker_db.set_block_stride([0])
                region_dbs: list[ChunkedTokenDatabase] = []
                for region_id, addr, block_stride, content_len in region_specs:
                    region_db = ChunkedTokenDatabase(
                        dataclasses.replace(
                            marker_db.metadata,
                            # The semantic layer id owns the object, not its
                            # endpoint-local HMA group or PP placement. Marker
                            # keys retain both local dimensions for scheduler
                            # lookup; transport data must remain portable when
                            # P and D construct different HMA group numbering.
                            pp_rank=-1,
                            group_id=-1,
                            region_id=region_id,
                        ),
                        marker_db.block_size,
                        hash_block_size=marker_db.hash_block_size,
                    )
                    region_db.set_kv_caches_base_addr([addr])
                    region_db.set_block_len([content_len])
                    region_db.set_block_stride([block_stride])
                    region_dbs.append(region_db)
                semantic_region_dbs_by_group.append(region_dbs)
                logger.info(
                    "Semantic Store group %d: marker_block_tokens=%d, "
                    "regions=%d, content_bytes_per_block=%d",
                    group_index,
                    marker_db.block_size,
                    len(region_dbs),
                    sum(db.block_len[0] for db in region_dbs),
                )
            wire_manifest = sorted(
                (db.metadata.region_id, db.block_len[0])
                for region_dbs in semantic_region_dbs_by_group
                for db in region_dbs
            )
            local_layout_manifest = sorted(
                (
                    group_index,
                    db.metadata.region_id,
                    db.block_len[0],
                    db.block_stride[0],
                )
                for group_index, region_dbs in enumerate(semantic_region_dbs_by_group)
                for db in region_dbs
            )
            wire_payload = json.dumps(wire_manifest, separators=(",", ":")).encode()
            local_layout_payload = json.dumps(
                local_layout_manifest, separators=(",", ":")
            ).encode()
            manifest_metadata = self.token_dbs[0].metadata
            logger.info(
                "Semantic Store manifest: schema=%s, regions=%d, "
                "wire_fingerprint=%s, local_layout_fingerprint=%s, "
                "shard=tp%d/pcp%d/dcp%d/pp%d, first_wire=%s, last_wire=%s",
                KDA_TARGET_STATE_TRANSPORT,
                len(wire_manifest),
                hashlib.blake2b(wire_payload, digest_size=16).hexdigest(),
                hashlib.blake2b(local_layout_payload, digest_size=16).hexdigest(),
                manifest_metadata.tp_rank,
                manifest_metadata.pcp_rank,
                manifest_metadata.dcp_rank,
                manifest_metadata.pp_rank,
                wire_manifest[0] if wire_manifest else None,
                wire_manifest[-1] if wire_manifest else None,
            )
        else:
            for db, regions in zip(self.token_dbs, regions_by_group, strict=True):
                addrs = [region[0] for region in regions]
                block_strides = [region[1] for region in regions]
                block_lens = [region[2] for region in regions]
                db.set_kv_caches_base_addr(addrs)
                db.set_block_len(block_lens)
                db.set_block_stride(block_strides)
        self.semantic_region_dbs_by_group = semantic_region_dbs_by_group

        # Start transfer threads
        if self.can_put:
            ready_event_sending = threading.Event()
            self.kv_send_thread = KVCacheStoreSendingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                self._group_tp_replication_factors,
                self.kv_role,
                ready_event_sending,
                self.enable_kv_events,
                self.store_replicate_config,
                enable_group_semantics=self.enable_group_semantics,
                supports_group_ids=self._supports_group_ids,
                record_operation=self._record_kv_connector_operation,
                kda_transport=kda_transport,
                semantic_region_dbs_by_group=semantic_region_dbs_by_group,
                semantic_commit_metadata=getattr(
                    self,
                    "_semantic_commit_metadata",
                    dataclasses.replace(
                        self.token_dbs[0].metadata,
                        group_id=-1,
                        region_id=SEMANTIC_COMMIT_REGION_ID,
                    ),
                ),
            )
            self.kv_send_thread.start()

        self.kv_recv_threads = []
        ready_events_recving = []
        for i in range(self.num_recv_threads):
            ready_event_recving = threading.Event()
            recv_thread = KVCacheStoreRecvingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                ready_event_recving,
                disk_offload_buffer_budget_bytes=self.disk_offload_buffer_budget_bytes,
                record_operation=self._record_kv_connector_operation,
                request_queue=self.recv_request_queue,
                kda_transport=kda_transport,
                semantic_region_dbs_by_group=semantic_region_dbs_by_group,
            )
            recv_thread.name = f"KVCacheStoreRecvingThread-{i}"
            recv_thread.start()
            self.kv_recv_threads.append(recv_thread)
            ready_events_recving.append(ready_event_recving)
        for ready_event_recving in ready_events_recving:
            ready_event_recving.wait()
        logger.info(
            "Started %d Mooncake KV-load receive thread(s)", self.num_recv_threads
        )

    def start_load_kv(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """No-op: loads are issued in get_finished() for overlap."""
        pass

    def wait_for_save(
        self,
        metadata: MooncakeStoreConnectorMetadata,
    ):
        """No-op: stores are issued in get_finished() for overlap."""
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        """Issue all I/O and get completed send/recv request IDs.

        All load and store I/O requests are issued here (after model
        compute is launched on the compute stream) for better
        compute-I/O overlap.
        """
        if self._capacity_only:
            return set(), set()

        if self.kv_send_thread is not None:
            self.kv_send_thread.acknowledge_completed_saves(
                meta.acknowledged_store_job_ids
            )

        # Issue async loads
        retired_load_req_ids = finished_req_ids | meta.preempted_req_ids
        self._submitted_load_req_ids.difference_update(retired_load_req_ids)
        for request in meta.requests:
            load_spec = request.load_spec
            if (
                load_spec is None
                or not load_spec.can_load
                or request.req_id in retired_load_req_ids
                or request.req_id in self._submitted_load_req_ids
            ):
                continue

            load_spec.token_len = load_spec.kvpool_cached_tokens
            self._submitted_load_req_ids.add(request.req_id)
            self.recv_request_queue.put(request)

        assert self.load_async, "load_async must be True for better performance."
        # Issue stores with CUDA event synchronization.
        if self.can_put:
            current_event = None
            for request in meta.requests:
                if request.can_save:
                    current_event = torch.cuda.Event()
                    current_event.record()
                    break

            for request in meta.requests:
                if not request.can_save:
                    continue
                request.current_event = current_event
                assert self.kv_send_thread is not None
                self.kv_send_thread.add_request(request)
            self._close_ended_store_requests(finished_req_ids, meta)

        # Blocks read by a store job are released by the scheduler when the job
        # reports back (see build_connector_worker_meta), so no request ever waits
        # on a `finished_sending` signal to get its blocks back.
        done_sending: set[str] = set()
        done_recving: set[str] = set()
        if self.load_async:
            for recv_thread in self.kv_recv_threads:
                done_recving |= recv_thread.get_and_clear_finished_requests()

        logger.debug(
            "Completed send: %d, recv: %d, tp_rank: %d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        block_ids: set[int] = set()
        for recv_thread in self.kv_recv_threads:
            block_ids |= recv_thread.get_and_clear_block_ids_with_load_errors()
        return block_ids

    def _record_kv_connector_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        with self._kv_connector_stats_lock:
            self.kv_connector_stats.record_operation(
                operation=operation,
                duration_seconds=duration_seconds,
                num_keys=num_keys,
                num_bytes=num_bytes,
                status=status,
                num_failed_keys=num_failed_keys,
            )

    def get_kv_connector_stats(self) -> MooncakeStoreConnectorStats | None:
        with self._kv_connector_stats_lock:
            if self.kv_connector_stats.is_empty():
                return None
            kv_connector_stats = self.kv_connector_stats
            self.kv_connector_stats = MooncakeStoreConnectorStats()
            return kv_connector_stats

    def _close_ended_store_requests(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> None:
        """Retire the ledger entries of requests that finished or were preempted.

        An entry may only go once its jobs have drained, because they still read
        the resume offset it owns; a request that comes back after preemption
        then saves from the start rather than from where the last attempt got to.
        """
        assert self.kv_send_thread is not None

        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(req_id)

        for req_id in finished_req_ids | self.finished_store_req:
            if self.kv_send_thread.stored_requests.get(req_id):
                # Queued jobs still need the resume offset; retire on a later step.
                self.finished_store_req.add(req_id)
            else:
                self.finished_store_req.discard(req_id)
                self.kv_send_thread.delete_finished_stored_request(req_id)

    def build_connector_worker_meta(self) -> MooncakeStoreWorkerMetadata | None:
        if self.kv_send_thread is None:
            return None
        completed_saves = self.kv_send_thread.take_completed_saves()
        if not completed_saves:
            return None
        worker_rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        return MooncakeStoreWorkerMetadata(
            completed_saves=completed_saves,
            completed_save_ranks={
                store_job_id: (worker_rank,) for store_job_id in completed_saves
            },
        )

    def lookup(self, num_tokens: int, block_hashes: Sequence[BlockHash]) -> int:
        """Check how many prefix tokens exist in the store.

        Checks across all rank-specific key namespaces that may be loaded. A
        hit covering all ``num_tokens`` is re-derived below the request end so
        the last token is recomputed for sampling.
        """
        if self._capacity_only:
            return 0

        token_len = self.coord.align_lookup_length(num_tokens)
        if not block_hashes or token_len <= 0:
            return 0

        semantic_regions = getattr(self, "semantic_region_dbs_by_group", None)
        if semantic_regions is not None:
            # Local marker keys deliberately retain endpoint-local HMA group
            # ids for that endpoint's scheduler.  They cannot be used to
            # discover a no-spec P -> RecoverSSM D hit because the two HMA
            # layouts may assign the same semantic layers to different groups.
            # Canonical commits are endpoint-neutral and are published only
            # after every target-state region for each producer shard is
            # visible.  The receive path still preflights every concrete region
            # key, so later partial eviction fails closed instead of mixing
            # restored and recomputed state.
            return self._lookup_semantic_commit_boundary(
                num_tokens,
                token_len,
                block_hashes,
            )

        # Build per-(group, hash) candidate keys expanded across rank namespaces.
        # candidate_meta stores the (group, hash_bytes) for key slice.
        candidate_keys: list[str] = []
        candidate_meta: list[tuple[int, bytes]] = []
        fine_grained = self.coord.enable_partial_hash_hits
        lookup_masks = None if fine_grained else self.coord.lookup_mask(token_len)
        for g_idx, db in enumerate(self.token_dbs):
            spec_block_size = db.block_size
            key_prefixes = self._lookup_key_prefixes[g_idx]
            if fine_grained:
                max_units = min(len(block_hashes), token_len // self.hash_block_size)
                unit_ids: range | list[int] = range(max_units)
                group_hashes: Sequence[BlockHash] = block_hashes
            else:
                lookup_mask = lookup_masks[g_idx]  # type: ignore[index]
                group_hashes = self.coord.block_hashes_for_spec(
                    block_hashes, self._kv_cache_groups[g_idx].kv_cache_spec
                )
                max_chunks = min(len(group_hashes), cdiv(token_len, spec_block_size))
                mask_limit = (
                    max_chunks
                    if lookup_mask is None
                    else min(max_chunks, len(lookup_mask))
                )
                unit_ids = [
                    chunk_id
                    for chunk_id in range(mask_limit)
                    if lookup_mask is None or lookup_mask[chunk_id]
                ]
            for chunk_id in unit_ids:
                h = group_hashes[chunk_id]
                hash_hex = h.hex()
                for key_prefix in key_prefixes:
                    candidate_keys.append(
                        PoolKey.build_key_string(key_prefix, hash_hex)
                    )
                candidate_meta.append((g_idx, bytes(h)))

        if not candidate_keys:
            return 0

        lookup_start = time.perf_counter()
        try:
            res = self.store.batch_is_exist(candidate_keys)
        except Exception as e:
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(candidate_keys),
                status="error",
                num_failed_keys=len(candidate_keys),
            )
            logger.error("Remote connection failed in lookup: %s", e)
            return 0
        if len(res) != len(candidate_keys):
            self._record_kv_connector_operation(
                "lookup_exists",
                time.perf_counter() - lookup_start,
                len(candidate_keys),
                status="error",
                num_failed_keys=len(candidate_keys),
            )
            logger.error(
                "Mooncake lookup returned %d results for %d keys",
                len(res),
                len(candidate_keys),
            )
            return 0
        self._record_kv_connector_operation(
            "lookup_exists",
            time.perf_counter() - lookup_start,
            len(candidate_keys),
        )

        # A (group, hash) is "present" only when every namespace that will be
        # loaded has it (per-group count: sharded groups need every rank's
        # shard, replicated groups one namespace per unique KV head).
        exists_set = set()
        pos = 0
        for g_idx, hash_bytes in candidate_meta:
            count = len(self._lookup_key_prefixes[g_idx])
            if all(res[pos + j] == 1 for j in range(count)):
                exists_set.add((g_idx, hash_bytes))
            pos += count

        cached_block_pool = ExternalCachedBlockPool(
            self.hash_block_size,
            exists_set,
        )
        _masks, hit_length = self.coord.find_longest_cache_hit(
            block_hashes,
            token_len,
            cached_block_pool,
        )
        if hit_length >= num_tokens:
            usable_length = self.coord.align_lookup_length(num_tokens - 1)
            if usable_length <= 0:
                return 0
            _masks, hit_length = self.coord.find_longest_cache_hit(
                block_hashes,
                usable_length,
                cached_block_pool,
            )

        return hit_length

    def _lookup_semantic_commit_boundary(
        self,
        num_tokens: int,
        token_len: int,
        block_hashes: Sequence[BlockHash],
    ) -> int:
        """Return the longest endpoint-neutral, fully committed boundary."""
        usable_length = token_len
        if usable_length >= num_tokens:
            usable_length = self.coord.align_lookup_length(num_tokens - 1)
        if usable_length <= 0:
            return 0

        commit_prefixes = self._semantic_commit_key_prefixes
        max_units = min(
            len(block_hashes),
            usable_length // self.hash_block_size,
        )
        commit_hashes = [bytes(block_hashes[index]) for index in range(max_units)]
        commit_keys = [
            PoolKey.build_key_string(prefix, hash_bytes.hex())
            for hash_bytes in commit_hashes
            for prefix in commit_prefixes
        ]
        if not commit_keys:
            return 0

        commit_start = time.perf_counter()
        try:
            commit_results = self.store.batch_is_exist(commit_keys)
        except Exception as e:
            self._record_kv_connector_operation(
                "lookup_semantic_commit",
                time.perf_counter() - commit_start,
                len(commit_keys),
                status="error",
                num_failed_keys=len(commit_keys),
            )
            logger.error("Semantic Mooncake commit lookup failed: %s", e)
            return 0
        if len(commit_results) != len(commit_keys):
            self._record_kv_connector_operation(
                "lookup_semantic_commit",
                time.perf_counter() - commit_start,
                len(commit_keys),
                status="error",
                num_failed_keys=len(commit_keys),
            )
            logger.error(
                "Semantic Mooncake commit lookup returned %d results for %d keys",
                len(commit_results),
                len(commit_keys),
            )
            return 0
        self._record_kv_connector_operation(
            "lookup_semantic_commit",
            time.perf_counter() - commit_start,
            len(commit_keys),
        )

        prefix_count = len(commit_prefixes)
        committed_hit = 0
        pos = 0
        for index in range(max_units):
            if all(commit_results[pos + offset] == 1 for offset in range(prefix_count)):
                committed_hit = (index + 1) * self.hash_block_size
            pos += prefix_count
        logger.debug(
            "Semantic Mooncake canonical lookup: committed_hit=%d, "
            "usable_length=%d, token_len=%d",
            committed_hit,
            usable_length,
            token_len,
        )
        return committed_hit

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            return self.kv_send_thread.get_kv_events()
        return []

    def close(self) -> None:
        """Release the MooncakeDistributedStore handle on teardown.

        Closing the store frees its TransferEngine, the registered RDMA
        buffers, and the connection to the master server. Idempotent so it is
        safe to call from both the explicit shutdown path and ``__del__``.
        """
        store = getattr(self, "store", None)
        if store is None:
            return
        self.store = None
        try:
            store.close()
        except Exception as e:
            logger.warning("Error closing MooncakeDistributedStore: %s", e)


# ============================================================
# Lookup Key Server
# ============================================================


class LookupKeyServer:
    """ZMQ server on worker rank 0 for the LookupKey admin channel.

    Handles two request types, tagged at frame 0:
    - ``LOOKUP_MSG``: prefix-cache hit query, returns hit count.
    - ``RESET_MSG``: drains the send thread queue, then runs
      ``store.remove_all(force=True)``. Caller must have paused the
      scheduler first.
    """

    def __init__(
        self,
        store_worker: MooncakeStoreWorker,
        vllm_config: VllmConfig,
    ):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self._ipc_path = socket_path.removeprefix("ipc://")
        if os.path.exists(self._ipc_path):
            os.unlink(self._ipc_path)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.store_worker = store_worker
        self.running = True

        def process_request():
            while self.running:
                all_frames = self.socket.recv_multipart(copy=False)
                msg_type = bytes(all_frames[0])

                if msg_type == LOOKUP_MSG:
                    num_tokens = int.from_bytes(all_frames[1], byteorder="big")
                    hash_len = int.from_bytes(all_frames[2], byteorder="big")
                    blob = all_frames[3].buffer
                    block_hashes = BlobBlockHashes(blob, hash_len)
                    result = self.store_worker.lookup(num_tokens, block_hashes)
                    self.socket.send(result.to_bytes(4, "big"))

                elif msg_type == RESET_MSG:
                    try:
                        # Drain in-flight puts before wiping the master;
                        # otherwise stale puts can repopulate it post-reset.
                        # Safe across HMA: store.remove_all wipes the underlying
                        # flat key space, clearing every (group_id, hash) entry.
                        if self.store_worker.kv_send_thread is not None:
                            self.store_worker.kv_send_thread.request_queue.join()
                        self.store_worker.store.remove_all(force=True)
                        logger.info("Mooncake store reset via remove_all succeeded.")
                        self.socket.send(RESP_OK)
                    except Exception as e:
                        logger.error("Mooncake remove_all failed: %s", e)
                        self.socket.send(RESP_ERR)

                else:
                    logger.warning(
                        "LookupKeyServer received unknown msg_type: %r",
                        msg_type,
                    )
                    self.socket.send(RESP_ERR)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        if os.path.exists(self._ipc_path):
            os.unlink(self._ipc_path)


# ============================================================
# Lookup Key Client
# ============================================================


class LookupKeyClient:
    """ZMQ client for the LookupKey admin channel.

    Routes both prefix-cache lookups and admin commands (currently:
    ``reset``) to ``LookupKeyServer`` on worker rank 0. The first frame
    of every request is a named tag from ``protocol.py``.
    """

    def __init__(self, vllm_config: VllmConfig):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

        # Async lookup support
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="MooncakeLookupClient"
        )
        self.futures: dict[str, Future[int]] = {}

    def _lookup(self, num_tokens: int, block_hashes: list[BlockHash]) -> int:
        hash_len = len(block_hashes[0]) if block_hashes else 0
        all_frames = (
            LOOKUP_MSG,
            num_tokens.to_bytes(4, byteorder="big"),
            hash_len.to_bytes(2, byteorder="big"),
            b"".join(block_hashes),
        )
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        return int.from_bytes(resp, "big")

    def lookup(
        self,
        req_id: str,
        num_tokens: int,
        block_hashes: list[BlockHash],
        non_block: bool = False,
    ) -> int | None:
        """If non_block is True, will return None until the result is ready,
        so the caller retries on a later step."""
        future = self.futures.get(req_id)
        if future is None:
            future = self.executor.submit(self._lookup, num_tokens, list(block_hashes))
            self.futures[req_id] = future
        if non_block and not future.done():
            return None
        try:
            return future.result()
        except Exception as e:
            logger.error("Async Mooncake lookup failed for %s: %s", req_id, e)
            return 0
        finally:
            del self.futures[req_id]

    def discard(self, req_id: str) -> None:
        """Drop any cached/in-flight lookup for ``req_id`` (e.g. on abort)."""
        future = self.futures.pop(req_id, None)
        if future is not None:
            future.cancel()

    def _reset(self) -> bool:
        """Trigger ``store.remove_all(force=True)`` on worker rank 0.

        Ordering assumption: caller MUST ensure no in-flight Mooncake
        lookups or transfers when invoking reset. In RL workflows this
        holds naturally at the step boundary after weight updates and
        rollout drain. Returns True on ACK, False on NACK.
        """
        self.socket.send(RESET_MSG)
        resp = self.socket.recv()
        return bytes(resp) == RESP_OK

    def reset(self) -> bool:
        return self.executor.submit(self._reset).result()

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.socket.close(linger=0)


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    """Construct IPC path for ZMQ lookup socket."""
    assert vllm_config.kv_transfer_config is not None
    dp_rank = vllm_config.parallel_config.data_parallel_index
    base_url = envs.VLLM_RPC_BASE_PATH
    rpc_port = 0
    hostname = socket.gethostname()
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return (
        f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_host_{hostname}_dp_rank{dp_rank}"
    )
