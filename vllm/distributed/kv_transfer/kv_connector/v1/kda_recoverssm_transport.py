# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout-neutral target-state transport for KDA RecoverSSM."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from math import prod

import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec

logger = logging.getLogger(__name__)

KDA_TARGET_STATE_TRANSPORT = "target_state_v1"
KDA_TARGET_CONV_REGION = "kda_target_conv_v1"
KDA_BASE_RECURRENT_REGION = "kda_base_recurrent_v1"


@dataclass(frozen=True)
class KDATransportRegion:
    kind: str
    tensor: torch.Tensor
    block_stride_bytes: int
    content_len_bytes: int


def _split_page_states(
    cache: torch.Tensor,
    spec: MambaSpec,
) -> tuple[torch.Tensor, ...]:
    if cache.ndim != 4 or tuple(cache.shape[1:3]) != (1, 1):
        raise ValueError(
            "KDA target-state transport requires a [blocks, 1, 1, bytes] cache"
        )
    if cache.element_size() != 1:
        raise ValueError("KDA target-state transport requires a byte cache view")

    pages = cache[:, 0, 0]
    if pages.shape[1] < spec.state_content_size_bytes:
        raise ValueError("KDA cache page is shorter than its declared state fields")
    states: list[torch.Tensor] = []
    offset = 0
    for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True):
        state_bytes = torch.empty((), dtype=dtype).element_size()
        num_bytes = prod(shape) * state_bytes
        state = pages[:, offset : offset + num_bytes].view(dtype)
        states.append(state.view(cache.shape[0], *shape))
        offset += num_bytes
    if offset != spec.state_content_size_bytes:
        raise ValueError("KDA state fields do not cover the declared page content")
    return tuple(states)


class KDATargetStateLayerTransport:
    """Expose only the target KDA state shared by P and RecoverSSM D."""

    def __init__(
        self,
        layer_name: str,
        group_index: int,
        cache: torch.Tensor,
        spec: MambaSpec,
        *,
        conv_state_dim_first: bool | None = None,
    ) -> None:
        if spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN:
            raise ValueError("KDA transport requires a GDN_ATTN MambaSpec")
        if len(spec.shapes) not in (2, 4):
            raise ValueError("KDA transport requires base or RecoverSSM state fields")
        if spec.num_speculative_blocks:
            raise ValueError(
                "ordinary speculative KDA pages are not target-state transportable"
            )

        self.layer_name = layer_name
        self.group_index = group_index
        self.cache = cache
        self.spec = spec
        self.states = _split_page_states(cache, spec)
        self.conv_state_dim_first = (
            is_conv_state_dim_first()
            if conv_state_dim_first is None
            else conv_state_dim_first
        )

        conv_state, recurrent_state = self.states[:2]
        local_conv = (
            conv_state if self.conv_state_dim_first else conv_state.transpose(-1, -2)
        )
        if len(self.states) == 4:
            spec_query_len = spec.shapes[2][1]
            self.conv_history_len = local_conv.shape[-1] - spec_query_len + 1
        else:
            self.conv_history_len = local_conv.shape[-1]
        if self.conv_history_len <= 0:
            raise ValueError("RecoverSSM conv state has no target history")

        self.local_conv = local_conv
        self.recurrent_state = recurrent_state
        self.target_conv = torch.empty(
            (
                cache.shape[0],
                local_conv.shape[-2],
                self.conv_history_len,
            ),
            dtype=conv_state.dtype,
            device=conv_state.device,
        )

    @property
    def regions(self) -> tuple[KDATransportRegion, KDATransportRegion]:
        conv_len = self.target_conv[0].numel() * self.target_conv.element_size()
        recurrent_len = (
            self.recurrent_state[0].numel() * self.recurrent_state.element_size()
        )
        return (
            KDATransportRegion(
                kind=KDA_TARGET_CONV_REGION,
                tensor=self.target_conv,
                block_stride_bytes=(
                    self.target_conv.stride(0) * self.target_conv.element_size()
                ),
                content_len_bytes=conv_len,
            ),
            KDATransportRegion(
                kind=KDA_BASE_RECURRENT_REGION,
                tensor=self.recurrent_state,
                block_stride_bytes=(
                    self.recurrent_state.stride(0) * self.recurrent_state.element_size()
                ),
                content_len_bytes=recurrent_len,
            ),
        )

    def stage_blocks(self, block_ids: list[int]) -> None:
        indices = self._indices(block_ids)
        if indices is None:
            return
        source = self.local_conv.index_select(0, indices)[..., : self.conv_history_len]
        self.target_conv.index_copy_(0, indices, source)

    def materialize_blocks(self, block_ids: list[int]) -> None:
        indices = self._indices(block_ids)
        if indices is None:
            return
        conv = torch.zeros(
            (indices.numel(), *self.local_conv.shape[1:]),
            dtype=self.local_conv.dtype,
            device=self.local_conv.device,
        )
        conv[..., : self.conv_history_len].copy_(
            self.target_conv.index_select(0, indices)
        )
        self.local_conv.index_copy_(0, indices, conv)
        for record in self.states[2:]:
            record.index_fill_(0, indices, 0)

    def _indices(self, block_ids: list[int]) -> torch.Tensor | None:
        valid = sorted(
            {
                block_id
                for block_id in block_ids
                if block_id != NULL_BLOCK_ID and block_id >= 0
            }
        )
        if not valid:
            return None
        if valid[-1] >= self.cache.shape[0]:
            raise ValueError(
                f"KDA block id {valid[-1]} exceeds {self.cache.shape[0]} blocks"
            )
        return torch.tensor(valid, dtype=torch.long, device=self.cache.device)


class KDATargetStateTransport:
    def __init__(
        self,
        layers: dict[str, KDATargetStateLayerTransport],
        attention_layers: dict[str, tuple[int, torch.Tensor, int]] | None = None,
        target_num_layers: int | None = None,
    ) -> None:
        self.layers = layers
        # Diagnostic-only raw attention pages: layer ->
        # (group index, cache tensor, physical blocks per scheduler block).
        self._attention_layers = attention_layers or {}
        self._target_num_layers = target_num_layers
        self._layers_by_group: dict[int, list[KDATargetStateLayerTransport]] = {}
        for layer in layers.values():
            self._layers_by_group.setdefault(layer.group_index, []).append(layer)
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_config: KVCacheConfig,
        *,
        conv_state_dim_first: bool | None = None,
        target_num_layers: int | None = None,
    ) -> KDATargetStateTransport:
        layers: dict[str, KDATargetStateLayerTransport] = {}
        attention_layers: dict[str, tuple[int, torch.Tensor, int]] = {}
        for group_index, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            specs_by_layer = getattr(group_spec, "kv_cache_specs", {})
            for layer_name in group.layer_names:
                spec = specs_by_layer.get(layer_name, group_spec)
                cache = kv_caches.get(layer_name)
                if cache is None:
                    continue
                if isinstance(spec, AttentionSpec):
                    if (
                        cache.ndim > 0
                        and kv_cache_config.num_blocks > 0
                        and cache.shape[0] % kv_cache_config.num_blocks == 0
                    ):
                        attention_layers[layer_name] = (
                            group_index,
                            cache,
                            cache.shape[0] // kv_cache_config.num_blocks,
                        )
                    continue
                if not isinstance(spec, MambaSpec):
                    continue
                layers[layer_name] = KDATargetStateLayerTransport(
                    layer_name,
                    group_index,
                    cache,
                    spec,
                    conv_state_dim_first=conv_state_dim_first,
                )
        return cls(layers, attention_layers, target_num_layers)

    def regions_for_layer(self, layer_name: str) -> tuple[KDATransportRegion, ...]:
        layer = self.layers.get(layer_name)
        return () if layer is None else layer.regions

    @property
    def group_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._layers_by_group))

    def stage_groups(self, block_ids_by_group: list[list[int]]) -> None:
        self._apply_groups(block_ids_by_group, materialize=False)

    def materialize_groups(self, block_ids_by_group: list[list[int]]) -> None:
        self._apply_groups(block_ids_by_group, materialize=True)

    def stage_group_blocks(self, block_ids_by_group: dict[int, list[int]]) -> None:
        groups = [[] for _ in range(max(self._layers_by_group, default=-1) + 1)]
        for group_index, block_ids in block_ids_by_group.items():
            if group_index < len(groups):
                groups[group_index] = block_ids
        self.stage_groups(groups)

    def materialize_group_blocks(
        self, block_ids_by_group: dict[int, list[int]]
    ) -> None:
        groups = [[] for _ in range(max(self._layers_by_group, default=-1) + 1)]
        for group_index, block_ids in block_ids_by_group.items():
            if group_index < len(groups):
                groups[group_index] = block_ids
        self.materialize_groups(groups)

    def _apply_groups(
        self,
        block_ids_by_group: list[list[int]],
        *,
        materialize: bool,
    ) -> None:
        if not self.layers:
            return
        with self._lock:
            devices = {layer.cache.device for layer in self.layers.values()}
            if len(devices) != 1:
                raise ValueError("KDA transport layers must share one device")
            device = next(iter(devices))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for group_index, block_ids in enumerate(block_ids_by_group):
                for layer in self._layers_by_group.get(group_index, ()):
                    if materialize:
                        layer.materialize_blocks(block_ids)
                    else:
                        layer.stage_blocks(block_ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def debug_checksums(
        self,
        location: str,
        request_id: str,
        block_ids_by_group: list[list[int]] | dict[int, list[int]],
    ) -> None:
        """Log normalized KDA and partial attention-page byte digests.

        This synchronizes and copies device data to the host, so it is enabled
        only for the two-request correctness canary.
        """
        if os.getenv("VLLM_KDA_TRANSPORT_CHECKSUM") != "1":
            return
        groups = (
            block_ids_by_group
            if isinstance(block_ids_by_group, dict)
            else dict(enumerate(block_ids_by_group))
        )

        def valid_ids(group_index: int) -> list[int]:
            return sorted(
                {
                    block_id
                    for block_id in groups.get(group_index, ())
                    if block_id != NULL_BLOCK_ID and block_id >= 0
                }
            )

        kda_digest = hashlib.blake2b(digest_size=16)
        target_attention_digest = hashlib.blake2b(digest_size=16)
        draft_attention_digest = hashlib.blake2b(digest_size=16)
        kda_bytes = 0
        target_attention_bytes = 0
        draft_attention_bytes = 0
        target_attention_layers: dict[str, str] = {}
        draft_attention_layers: dict[str, str] = {}
        selected_groups: dict[int, tuple[int, int]] = {}
        with self._lock:
            devices = {
                layer.cache.device for layer in self.layers.values()
            } | {cache.device for _, cache, _ in self._attention_layers.values()}
            for device in devices:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            for layer_name in sorted(self.layers):
                layer = self.layers[layer_name]
                block_ids = valid_ids(layer.group_index)
                indices = layer._indices(block_ids)
                if indices is None:
                    continue
                selected_groups[layer.group_index] = (
                    len(block_ids),
                    block_ids[-1],
                )
                for region in layer.regions:
                    selected = region.tensor.index_select(0, indices).contiguous()
                    payload = selected.view(torch.uint8).cpu().numpy().tobytes()
                    kda_digest.update(layer_name.encode())
                    kda_digest.update(region.kind.encode())
                    kda_digest.update(payload)
                    kda_bytes += len(payload)

            for layer_name in sorted(self._attention_layers):
                group_index, cache, physical_per_logical = self._attention_layers[
                    layer_name
                ]
                block_ids = valid_ids(group_index)
                if not block_ids:
                    continue
                logical_block_id = block_ids[-1]
                physical_start = logical_block_id * physical_per_logical
                physical_end = physical_start + physical_per_logical
                if physical_end > cache.shape[0]:
                    raise ValueError(
                        f"Attention block {logical_block_id} maps to "
                        f"[{physical_start}, {physical_end}) beyond "
                        f"{cache.shape[0]} blocks for {layer_name}"
                    )
                selected_groups[group_index] = (
                    len(block_ids),
                    logical_block_id,
                )
                selected = cache[physical_start:physical_end].contiguous()
                payload = selected.view(torch.uint8).cpu().numpy().tobytes()
                layer_digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
                is_draft = (
                    self._target_num_layers is not None
                    and extract_layer_index(layer_name) >= self._target_num_layers
                )
                if is_draft:
                    draft_attention_digest.update(layer_name.encode())
                    draft_attention_digest.update(payload)
                    draft_attention_bytes += len(payload)
                    draft_attention_layers[layer_name] = layer_digest
                else:
                    target_attention_digest.update(layer_name.encode())
                    target_attention_digest.update(payload)
                    target_attention_bytes += len(payload)
                    target_attention_layers[layer_name] = layer_digest

        logger.info(
            "KDA_TRANSPORT_CHECKSUM location=%s req=%s groups=%s kda=%s "
            "kda_bytes=%d target_attention_tail=%s "
            "target_attention_tail_bytes=%d draft_attention_tail=%s "
            "draft_attention_tail_bytes=%d target_attention_layers=%s "
            "draft_attention_layers=%s",
            location,
            request_id,
            selected_groups,
            kda_digest.hexdigest(),
            kda_bytes,
            target_attention_digest.hexdigest(),
            target_attention_bytes,
            draft_attention_digest.hexdigest(),
            draft_attention_bytes,
            target_attention_layers,
            draft_attention_layers,
        )

    def debug_entry_checksums(
        self,
        location: str,
        request_id: str,
        boundary_tokens: int,
        group_index: int,
        block_id: int,
        chunk_hash: bytes,
    ) -> None:
        """Log the canonical KDA bytes for one semantic Store entry.

        Unlike :meth:`debug_checksums`, this deliberately excludes attention
        pages and never combines multiple logical blocks.  A save and load can
        therefore be paired by ``(boundary, group, chunk_hash)`` even when the
        physical block IDs differ between producer and consumer.
        """
        if os.getenv("VLLM_KDA_TRANSPORT_ENTRY_CHECKSUM") != "1":
            return
        block_ids = [block_id]
        digest = hashlib.blake2b(digest_size=16)
        region_digests: list[tuple[str, str, str, int]] = []
        total_bytes = 0
        with self._lock:
            layers = self._layers_by_group.get(group_index, ())
            devices = {layer.cache.device for layer in layers}
            for device in devices:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            for layer in sorted(layers, key=lambda item: item.layer_name):
                indices = layer._indices(block_ids)
                if indices is None:
                    continue
                for region in layer.regions:
                    selected = region.tensor.index_select(0, indices).contiguous()
                    payload = selected.view(torch.uint8).cpu().numpy().tobytes()
                    region_digest = hashlib.blake2b(
                        payload, digest_size=16
                    ).hexdigest()
                    digest.update(layer.layer_name.encode())
                    digest.update(region.kind.encode())
                    digest.update(payload)
                    total_bytes += len(payload)
                    region_digests.append(
                        (layer.layer_name, region.kind, region_digest, len(payload))
                    )

        logger.info(
            "KDA_STORE_ENTRY_CHECKSUM location=%s req=%s boundary=%d "
            "group=%d block=%d hash=%s digest=%s bytes=%d regions=%s",
            location,
            request_id,
            boundary_tokens,
            group_index,
            block_id,
            chunk_hash.hex(),
            digest.hexdigest(),
            total_bytes,
            region_digests,
        )


def kda_target_state_transport_enabled(extra_config: dict[str, object]) -> bool:
    return extra_config.get("kda_transport_policy") == KDA_TARGET_STATE_TRANSPORT
