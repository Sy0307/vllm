# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import async_tensor_h2d
import os as _k3_os
_K3_PP_DRAFT_NOSYNC = _k3_os.environ.get("VLLM_K3_PP_DRAFT_NOSYNC", "0") == "1"
_K3_PP_EARLY_COMMIT = __import__('os').environ.get('VLLM_K3_PP_EARLY_COMMIT', '0') == '1'  # k3_early_commit
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event | None

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    combined: torch.Tensor  # [2, num_reqs]: broadcast backing storage
    # Views into `combined`; a deferred broadcast updates them in place.
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]
    restore_model_state: Callable[[], None] | None = None
    # `record_event()` returns None for the CPU stream placeholder, so event
    # presence cannot be used to determine whether collectives were posted.
    launched: bool = False
    event_sampled: torch.cuda.Event | None = None  # k3_early_commit


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""
    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


def _default_recv_launch_delay(pp_size: int, async_scheduling: bool | None) -> int:
    """Return the automatic sampled-token receive delay.

    Async scheduling keeps multiple pipeline steps in flight, so the receive
    can overlap later work and still be posted on the last step before it is
    consumed. Synchronous scheduling cannot use that overlap and retains the
    upstream immediate-posting behavior. Non-CUDA-like platforms do likewise
    because this implementation relies on CUDA stream ordering.
    """
    if async_scheduling and current_platform.is_cuda_alike():
        return max(0, pp_size - 1)
    return 0


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
        async_scheduling: bool | None,
    ):
        pp_group = get_pp_group()
        self.is_last_rank = pp_group.is_last_rank
        self.last_rank = pp_group.last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        self.deferred_recv_launch_delay = (
            0
            if self.is_last_rank
            else _default_recv_launch_delay(pp_group.world_size, async_scheduling)
        )
        self.post_model_recv_launch = self.deferred_recv_launch_delay > 0
        # Keep warmup timing unchanged; the worker enables this after warmup.
        self.recv_launch_delay = 0
        self.pending_post_model_receive: PendingRecv | None = None

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * pp_group.world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = pp_group.make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        self.aux_hidden_state_relay_keys: tuple[str, ...] = ()

    def enable_deferred_collectives(self) -> None:
        """Enable the automatic receive delay after worker warmup."""
        self.recv_launch_delay = self.deferred_recv_launch_delay

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def configure_aux_hidden_state_relay(self, model: torch.nn.Module) -> None:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            aux_hidden_state_relay_keys,
        )

        self.aux_hidden_state_relay_keys = aux_hidden_state_relay_keys(model)

    def relay_aux_hidden_states(
        self,
        intermediate_tensors: IntermediateTensors | None,
        output_intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        if not self.aux_hidden_state_relay_keys:
            return output_intermediate_tensors
        assert intermediate_tensors is not None
        return IntermediateTensors(
            output_intermediate_tensors.tensors
            | {
                key: intermediate_tensors[key]
                for key in self.aux_hidden_state_relay_keys
            }
        )

    def _launch_receive(self, slot: PendingRecv) -> None:
        """Post one receiver's broadcasts exactly once."""
        if slot.launched:
            return
        with torch.cuda.stream(self.broadcast_stream):
            # This preserves upstream's main-to-broadcast stream dependency for
            # immediate receives. Deferred callers invoke this after the model,
            # so the same dependency places the receive behind model kernels.
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                slot.sampled_tokens,
                src=self.last_rank,
                group=self.broadcast_group,
            )
            torch.distributed.broadcast(
                slot.combined,
                src=self.last_rank,
                group=self.broadcast_group,
            )
            if _K3_PP_EARLY_COMMIT:
                slot.event_sampled = self.broadcast_stream.record_event()
            if slot.draft_tokens is not None:
                torch.distributed.broadcast(
                    slot.draft_tokens,
                    src=self.last_rank,
                    group=self.broadcast_group,
                )
            slot.event = self.broadcast_stream.record_event()
            # The tensors are populated on the broadcast stream and consumed
            # later on the main stream.
            slot.sampled_tokens.record_stream(self.main_stream)
            slot.combined.record_stream(self.main_stream)
            if slot.draft_tokens is not None:
                slot.draft_tokens.record_stream(self.main_stream)
            slot.launched = True

    def _advance_receive_queue(self) -> PendingRecv | None:
        """Select the configured slot and return the slot due this step."""
        if self.recv_launch_delay:
            launch_slot = self.queue[-self.recv_launch_delay]
            if launch_slot is not None:
                if self.post_model_recv_launch:
                    if self.pending_post_model_receive is not None:
                        # A prior step missed its post-model launch. Post that
                        # older slot first to preserve collective FIFO order.
                        self._launch_receive(self.pending_post_model_receive)
                    self.pending_post_model_receive = launch_slot
                else:
                    self._launch_receive(launch_slot)

        due_slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        return due_slot

    def _queue_receive(self, slot: PendingRecv) -> None:
        """Associate a receive with the current step and launch if immediate."""
        self.queue[-1] = slot
        if self.recv_launch_delay == 0:
            self._launch_receive(slot)

    def launch_post_model_receive(self) -> bool:
        """Post the selected receive behind this step's model kernels."""
        if not self.post_model_recv_launch or self.is_last_rank:
            return False
        slot = self.pending_post_model_receive
        if slot is None:
            return False
        self.pending_post_model_receive = None
        self._launch_receive(slot)
        return True

    def flush_pending_collectives(self) -> None:
        """Post all deferred receives before an idle boundary or shutdown.

        Every non-last rank in the PP group must call this at the same logical
        boundary. The matching broadcasts are ordered collectives, so a
        rank-local flush condition could desynchronize the group and hang it.
        """
        if self.is_last_rank:
            return

        if self.pending_post_model_receive is not None:
            pending_slot = self.pending_post_model_receive
            self.pending_post_model_receive = None
            self._launch_receive(pending_slot)
        for queued_slot in self.queue:
            if queued_slot is not None and not queued_slot.launched:
                self._launch_receive(queued_slot)

    def get_prev_sampled_outputs(
        self, draft_tokens_to_update: torch.Tensor | None = None, defer_drafts: bool = False
    ) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self._advance_receive_queue()
        if slot is None:
            return None

        # Post the receive before filtering. The last PP rank has already
        # posted the matching broadcasts, so even an all-excluded local step
        # must participate to preserve the group's collective order.
        self._launch_receive(slot)
        assert slot.launched

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_tensor_h2d(idx_mapping_np, device=self.device)

        _k3_early = (_K3_PP_EARLY_COMMIT and defer_drafts and slot.event_sampled is not None
                     and slot.draft_tokens is not None and draft_tokens_to_update is not None)
        _k3_ev = slot.event_sampled if _k3_early else slot.event
        if _k3_ev is not None:
            self.main_stream.wait_event(_k3_ev)
        if slot.restore_model_state is not None:
            slot.restore_model_state()
        if _k3_early:
            # Drafts are applied after postprocess (apply_deferred_drafts).
            self._k3_deferred_drafts = (slot, draft_tokens_to_update, exclude_mask, idx_mapping)
            draft_tokens_to_update = None
        if slot.draft_tokens is not None and draft_tokens_to_update is not None:
            draft_tokens = slot.draft_tokens
            draft_idx_mapping = slot.idx_mapping
            if exclude_mask.any() and _K3_PP_DRAFT_NOSYNC:
                # Rows map to distinct request slots: excluded rows are written
                # back with their current value (no-op) instead of being
                # compacted, which needed a blocking mask copy + boolean gather.
                current = draft_tokens_to_update[draft_idx_mapping]
                draft_tokens = torch.where(
                    (idx_mapping >= 0).unsqueeze(1),
                    draft_tokens.to(current.dtype),
                    current,
                )
            elif exclude_mask.any():
                keep = ~exclude_mask
                keep_t = torch.as_tensor(keep, device=self.device)
                draft_tokens = draft_tokens[keep_t]
                draft_idx_mapping = async_tensor_h2d(
                    slot.idx_mapping_np[keep], device=self.device
                )
            draft_tokens_to_update[draft_idx_mapping] = draft_tokens
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )

    def apply_deferred_drafts(self) -> None:
        """k3_early_commit: wait the full receive event, then write the drafts (same code as inline)."""
        pend = getattr(self, '_k3_deferred_drafts', None)
        if pend is None:
            return
        self._k3_deferred_drafts = None
        slot, draft_tokens_to_update, exclude_mask, idx_mapping = pend
        if slot.event is not None:
            self.main_stream.wait_event(slot.event)
        draft_tokens = slot.draft_tokens
        draft_idx_mapping = slot.idx_mapping
        if exclude_mask.any() and _K3_PP_DRAFT_NOSYNC:
            current = draft_tokens_to_update[draft_idx_mapping]
            draft_tokens = torch.where(
                (idx_mapping >= 0).unsqueeze(1),
                draft_tokens.to(current.dtype),
                current,
            )
        elif exclude_mask.any():
            keep = ~exclude_mask
            keep_t = torch.as_tensor(keep, device=self.device)
            draft_tokens = draft_tokens[keep_t]
            draft_idx_mapping = async_tensor_h2d(
                slot.idx_mapping_np[keep], device=self.device
            )
        draft_tokens_to_update[draft_idx_mapping] = draft_tokens

    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        """Broadcast draft proposals so non-last ranks can embed real token ids."""
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens[input_batch.idx_mapping].contiguous()
            # Must record the idx_mapping tensor since it was allocated
            # on the main stream.
            input_batch.idx_mapping.record_stream(self.broadcast_stream)
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.broadcast_group
            )

    def receive(
        self,
        input_batch: InputBatch,
        restore_model_state: Callable[[], None] | None = None,
    ) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            # These are views of the broadcast target. Creating them before a
            # deferred broadcast is intentional: filling `combined` updates
            # both views in place.
            num_sampled, num_rejected = combined.unbind(dim=0)
            draft_tokens = None
            if self.num_speculative_steps > 0:
                draft_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
        slot = PendingRecv(
            None,
            sampled_tokens,
            combined,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
            restore_model_state,
        )
        self._queue_receive(slot)
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send_tokens = torch.nn.functional.pad(
                sampled_token_ids,
                (0, self.max_sample_len - sampled_token_ids.shape[-1]),
            )
            torch.distributed.broadcast(
                send_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
