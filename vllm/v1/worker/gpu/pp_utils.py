# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    # Recorded when the receive is posted; None until then.
    event: torch.cuda.Event | None

    sampled_tokens: torch.Tensor  # [num_reqs, sample_len]
    # [2, num_reqs] broadcast target; num_sampled/num_rejected are views of it.
    combined: torch.Tensor
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
    # Restores the model state's postprocess inputs of this step (see
    # ModelState.defer_postprocess_state) before its postprocess runs.
    restore_model_state: Callable[[], None] | None = None
    # Whether the matching broadcasts were posted on the broadcast stream.
    launched: bool = False


def default_recv_launch_delay(pp_size: int, async_scheduling: bool | None) -> int:
    """Steps by which a non-last rank defers posting its sampled-token receive.

    A receive posted right after a step's forward sits on the GPU until the
    last rank samples that step, and its NCCL kernel keeps SMs that whole-grid
    kernels of the next forward (e.g. MegaMoE) wait for, which serializes the
    pipeline stages. With async scheduling `pp_size` steps are in flight, so
    the receive can instead be posted behind a later step's forward and still
    complete before it is consumed. Synchronous scheduling keeps immediate
    posting.
    """
    if async_scheduling and current_platform.is_cuda_alike():
        return max(0, pp_size - 1)
    return 0


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    with outputs that might be needed in a subsequent (decode) step.
    Returns None if no sampled outputs are needed in the requests' next step."""
    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    need_sampled_mask = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    if input_batch.max_seq_len_np is not None:
        # Also exclude final prefill chunks whose single sampled token reaches
        # the request's length cap.
        finished_prefill = prefill_len + 1 >= input_batch.max_seq_len_np
        finished_prefill &= input_batch.is_prefilling_np
        need_sampled_mask &= ~finished_prefill
    return need_sampled_mask if need_sampled_mask.any() else None


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
        async_scheduling: bool | None = None,
    ):
        pp_group = get_pp_group()
        self.is_last_rank = pp_group.is_last_rank
        self.last_rank = pp_group.last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # Deferred receives are posted behind the model kernels of the step
        # before they are due. Warmup keeps immediate posting; the worker calls
        # `enable_deferred_collectives` afterwards.
        self.deferred_recv_launch_delay = (
            0
            if self.is_last_rank
            else default_recv_launch_delay(pp_group.world_size, async_scheduling)
        )
        self.recv_launch_delay = 0
        self.pending_post_model_receive: PendingRecv | None = None

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        # Only XPU can disable microbatching via VLLM_XPU_PP_MICROBATCH.
        ring_depth = get_pp_group().world_size
        if current_platform.is_xpu() and not envs.VLLM_XPU_PP_MICROBATCH:
            ring_depth = 1
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * ring_depth)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        # Whether sampled tokens were broadcast this step.
        self.sampled_tokens_broadcast = False
        self.aux_hidden_state_relay_keys: tuple[str, ...] = ()

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def _sample_len(self, input_batch: InputBatch) -> int:
        # Width of the sampled-token broadcast. Must be the same on all ranks.
        return self.max_sample_len if input_batch.num_draft_tokens > 0 else 1

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

    def enable_deferred_collectives(self) -> None:
        self.recv_launch_delay = self.deferred_recv_launch_delay

    def _launch_receive(self, slot: PendingRecv) -> None:
        """Post the broadcasts matching one step of the last rank, once."""
        if slot.launched:
            return
        with torch.cuda.stream(self.broadcast_stream):
            # A deferred receive is launched after this rank's model kernels,
            # so the dependency places it behind them.
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                slot.sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                slot.combined, src=self.last_rank, group=self.broadcast_group
            )
            if slot.draft_tokens is not None:
                torch.distributed.broadcast(
                    slot.draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
            slot.event = self.broadcast_stream.record_event()
            # Allocated on the broadcast stream but consumed on the main stream.
            slot.sampled_tokens.record_stream(self.main_stream)
            slot.combined.record_stream(self.main_stream)
            if slot.draft_tokens is not None:
                slot.draft_tokens.record_stream(self.main_stream)
        slot.launched = True

    def _advance_receive_queue(self) -> PendingRecv | None:
        """Select the receive to post after this step's model and pop the due
        one. Receives are always posted in step order (ordered collectives)."""
        if self.recv_launch_delay:
            launch_slot = self.queue[-self.recv_launch_delay]
            if launch_slot is not None:
                if self.pending_post_model_receive is not None:
                    # The previous step had no post-model launch point.
                    self._launch_receive(self.pending_post_model_receive)
                self.pending_post_model_receive = launch_slot
        due_slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        return due_slot

    def launch_post_model_receive(self) -> None:
        """Post the receive selected for this step behind its model kernels."""
        slot = self.pending_post_model_receive
        if slot is None:
            return
        self.pending_post_model_receive = None
        self._launch_receive(slot)

    def flush_pending_collectives(self) -> None:
        """Post every deferred receive (idle step or shutdown). All non-last
        ranks reach this at the same logical step, so collective order holds."""
        if self.is_last_rank:
            return
        # The pending slot is still queued, so FIFO order covers it.
        self.pending_post_model_receive = None
        for slot in self.queue:
            if slot is not None:
                self._launch_receive(slot)

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor | None] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self._advance_receive_queue()
        if slot is None:
            return None
        # Post before filtering: the last rank broadcast this step regardless
        # of which local requests still need it.
        self._launch_receive(slot)

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

        assert slot.event is not None
        self.main_stream.wait_event(slot.event)
        if slot.restore_model_state is not None:
            slot.restore_model_state()
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
            draft_tokens=slot.draft_tokens,
        )

    def broadcast_drafts(self, draft_tokens: torch.Tensor) -> None:
        """Broadcast draft proposals so non-last ranks can embed real token ids."""
        assert self.is_last_rank
        if self.sampled_tokens_broadcast:
            with torch.cuda.stream(self.broadcast_stream):
                self.broadcast_stream.wait_stream(self.main_stream)
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
                draft_tokens.record_stream(self.broadcast_stream)

    def receive(
        self,
        input_batch: InputBatch,
        recv_drafts: bool,
        restore_model_state: Callable[[], None] | None = None,
    ) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch. `recv_drafts` must match whether the last rank
        calls `broadcast_drafts` for this step. `restore_model_state` runs right
        before this step's postprocess, once its sampled tokens arrive."""
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
            sample_len = self._sample_len(input_batch)
            sampled_tokens = torch.empty(
                num_reqs, sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            draft_tokens = None
            if recv_drafts:
                draft_tokens = sampled_tokens.new_empty(
                    num_reqs, self.num_speculative_steps
                )
        # Views of the broadcast target, filled in place when it is posted.
        num_sampled, num_rejected = combined.unbind(dim=0)
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
        self.queue[-1] = slot
        if self.recv_launch_delay == 0:
            self._launch_receive(slot)
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
            self.sampled_tokens_broadcast = False
            return

        assert sampled_token_ids.dtype == torch.int64
        self.sampled_tokens_broadcast = True

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            # receive() allocates the same width on the other ranks.
            assert sampled_token_ids.shape[-1] == self._sample_len(input_batch)
            torch.distributed.broadcast(
                sampled_token_ids.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
