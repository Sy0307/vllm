# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import contextlib
import logging
import tempfile
from typing import Any

import torch
from packaging import version
from torch.cuda.memory import CUDAPluggableAllocator
from torch.utils.cpp_extension import load_inline

from vllm import envs
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.nccl import find_nccl_include_paths, find_nccl_library_paths

logger = init_logger(__name__)

nccl_allocator_source = """
#include <nccl.h>
#include <stdio.h>
extern "C" {

void* nccl_alloc_plug(size_t size, int device, void* stream) {
  // A failed ncclMemAlloc leaves ptr unset: return nullptr so the caching
  // allocator reports out-of-memory instead of handing out a garbage pointer
  // that would then be registered as a symmetric window.
  void* ptr = nullptr;
  ncclResult_t err = ncclMemAlloc(&ptr, size);
  if (err != ncclSuccess) {
    fprintf(stderr, "[vllm nccl_alloc_plug] ncclMemAlloc(%zu) on device %d "
            "failed: %d\\n", size, device, (int)err);
    return nullptr;
  }
  return ptr;
}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
  ncclResult_t err = ncclMemFree(ptr);
  if (err != ncclSuccess) {
    fprintf(stderr, "[vllm nccl_free_plug] ncclMemFree(%p, %zu) on device %d "
            "failed: %d\\n", ptr, size, device, (int)err);
  }
}

}
"""

_allocator = None
_allocator_wrapper = None
_mem_pool = None
_registered_base_addrs: dict[bytes, set] = {}
_graph_pool_id = None
_nccl_allocator_failed_to_compile = False
_cached_pool_snapshot = None


def is_symmetric_memory_enabled():
    global _nccl_allocator_failed_to_compile
    return envs.VLLM_USE_NCCL_SYMM_MEM and not _nccl_allocator_failed_to_compile


def is_symmetric_memory_tensor(tensor: torch.Tensor):
    if not is_symmetric_memory_enabled() or _cached_pool_snapshot is None:
        return False
    for segment in _cached_pool_snapshot:
        for block in segment["blocks"]:
            if block["address"] == tensor.untyped_storage().data_ptr():
                return True
    return False


def set_graph_pool_id(graph_pool_id: Any) -> None:
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def compile_nccl_allocator():
    global _allocator, _allocator_wrapper, _nccl_allocator_failed_to_compile
    if not current_platform.is_cuda():
        _nccl_allocator_failed_to_compile = True
        return
    try:
        out_dir = tempfile.gettempdir()
        nccl_allocator_libname = "nccl_allocator"
        nccl_include_paths = find_nccl_include_paths()
        ldflags = ["-l:libnccl.so.2"]
        nccl_lib_paths = find_nccl_library_paths()
        if nccl_lib_paths:
            ldflags = [f"-L{p}" for p in nccl_lib_paths] + ldflags
        load_inline(
            name=nccl_allocator_libname,
            cpp_sources=nccl_allocator_source,
            with_cuda=True,
            extra_ldflags=ldflags,
            verbose=logger.isEnabledFor(logging.DEBUG),
            is_python_module=False,
            build_directory=out_dir,
            extra_include_paths=nccl_include_paths,
        )
        _allocator_wrapper = CUDAPluggableAllocator(
            f"{out_dir}/{nccl_allocator_libname}.so",
            "nccl_alloc_plug",
            "nccl_free_plug",
        )
        _allocator = _allocator_wrapper.allocator()
    except Exception as e:
        _nccl_allocator_failed_to_compile = True
        logger.warning(
            "Failed to compile NCCL memory allocator. "
            "Symmetric memory will be disabled. "
            "This is expected if NCCL headers are not available. "
            "optionally set VLLM_NCCL_INCLUDE_PATH to point to a directory "
            "containing the NCCL header. "
            "Error: %s",
            str(e),
        )


def get_nccl_mem_pool():
    global _mem_pool, _nccl_allocator_failed_to_compile
    if _mem_pool is None and not _nccl_allocator_failed_to_compile:
        compile_nccl_allocator()
        if _allocator is not None:
            _mem_pool = torch.cuda.MemPool(_allocator)
    return _mem_pool


def _cleanup_nccl_mem_pool():
    global _mem_pool
    _mem_pool = None


def _cleanup_nccl_allocator_wrapper():
    global _allocator_wrapper
    _allocator_wrapper = None


atexit.register(_cleanup_nccl_mem_pool)
atexit.register(_cleanup_nccl_allocator_wrapper)


class nccl_symm_mem_context:
    def __init__(
        self,
        pynccl_comm: PyNcclCommunicator,
        disabled: bool = False,
    ):
        self.disabled = (
            disabled
            or not is_symmetric_memory_enabled()
            or pynccl_comm.world_size == 1
            or not current_platform.is_cuda()
            or get_nccl_mem_pool() is None
            or version.parse(torch.__version__) < version.parse("2.8.0.a0")
        )
        if self.disabled:
            self.pynccl_comm: PyNcclCommunicator | None = None
            self._mem_pool_ctx: contextlib.AbstractContextManager[Any] = (
                contextlib.nullcontext()
            )
            self.is_graph_capture = None
            self.device = None
        else:
            self.pynccl_comm = pynccl_comm
            self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
            self.is_graph_capture = torch.cuda.is_current_stream_capturing()
            self.device = torch.accelerator.current_device_index()

    def __enter__(self):
        if self.disabled:
            return self
        assert self.pynccl_comm is not None, (
            "Symmetric memory requires pynccl to be initialized"
        )
        assert self.pynccl_comm.nccl_version >= 22703, (
            "NCCL version 2.27.3 or higher is required for NCCL symmetric memory"
        )
        if self.is_graph_capture:
            assert _graph_pool_id is not None, (
                "graph_pool_id is not set under graph capture"
            )
            # Pause graph memory pool to use symmetric memory with cuda graph
            torch._C._cuda_endAllocateToPool(self.device, _graph_pool_id)
        self._mem_pool_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disabled:
            return
        global _cached_pool_snapshot
        global _registered_base_addrs
        self._mem_pool_ctx.__exit__(exc_type, exc_val, exc_tb)
        _pool = get_nccl_mem_pool()
        assert _pool is not None
        _cached_pool_snapshot = _pool.snapshot()
        assert self.pynccl_comm is not None
        comm_key = bytes(self.pynccl_comm.unique_id.internal)
        if comm_key not in _registered_base_addrs:
            _registered_base_addrs[comm_key] = set()
        for segment in _cached_pool_snapshot:
            if segment["address"] not in _registered_base_addrs[comm_key]:
                if envs.VLLM_NCCL_SYMM_MEM_LOG:
                    # Window registration is collective: every rank of the
                    # communicator must register matching segments in the same
                    # order. Log each one to compare ranks after the fact.
                    logger.info(
                        "NCCL symm window register: rank %d/%d comm %s "
                        "addr 0x%x size %d (pool segments %d, registered %d)",
                        self.pynccl_comm.rank,
                        self.pynccl_comm.world_size,
                        comm_key[:6].hex(),
                        segment["address"],
                        segment["total_size"],
                        len(_cached_pool_snapshot),
                        len(_registered_base_addrs[comm_key]),
                    )
                self.pynccl_comm.register_comm_window_raw(
                    segment["address"], segment["total_size"]
                )
                _registered_base_addrs[comm_key].add(segment["address"])
        if self.is_graph_capture:
            torch._C._cuda_beginAllocateCurrentThreadToPool(self.device, _graph_pool_id)


# NCCL symmetric kernels address a peer's buffer as (peer window base + this
# rank's offset in its own window): a collective is only correct when every
# rank's buffer sits at the same offset of a matching window. Tensors taken per
# call from the shared caching pool do not guarantee that: whether a freed block
# can be reused depends on when its stream events complete, which differs per
# rank, so the next allocation can land at a different offset on one rank. A
# peer then writes or reads the wrong bytes of that rank's window, or past it
# (an NVLink access the GPU rejects). Symmetric buffers therefore come only from
# grow-only scratch: shape-driven allocations that are never freed, so every
# rank issues the same allocation sequence on the pool and gets the same layout.
_symm_scratch: dict[tuple, torch.Tensor] = {}
# Outgrown scratch stays alive: captured CUDA graphs may still use its pointer.
_symm_scratch_retired: list[torch.Tensor] = []


def _symm_placement(tensor: torch.Tensor) -> tuple[int, int | None, int | None]:
    """(pool segments, size of the tensor's segment, offset in it)."""
    snapshot = _cached_pool_snapshot or []
    ptr = tensor.untyped_storage().data_ptr()
    for segment in snapshot:
        base, size = segment["address"], segment["total_size"]
        if base <= ptr < base + size:
            return len(snapshot), size, ptr - base
    return len(snapshot), None, None


def check_symmetric_placement(
    pynccl_comm: PyNcclCommunicator, tensor: torch.Tensor, what: str
) -> None:
    """Fail loudly unless ``tensor`` sits in a pool segment of the same size at
    the same offset on every rank of ``pynccl_comm`` (one CPU all-gather; only
    called when scratch grows). An asymmetric symmetric buffer corrupts
    collectives silently or faults a peer's NVLink, so this is not optional."""
    if not envs.VLLM_NCCL_SYMM_MEM_CHECK or pynccl_comm.group is None:
        return
    mine = _symm_placement(tensor) + (tensor.nbytes,)
    group = pynccl_comm.group
    if isinstance(group, torch.distributed.ProcessGroup):
        gathered: list[Any] = [None] * pynccl_comm.world_size
        torch.distributed.all_gather_object(gathered, mine, group=group)
    else:
        gathered = group.all_gather_obj(mine)
    if any(other[1:] != mine[1:] for other in gathered):
        raise RuntimeError(
            f"NCCL symmetric memory: {what} is not symmetric across ranks "
            f"(segments, segment size, offset, bytes per rank: {gathered}). "
            "Set VLLM_USE_NCCL_SYMM_MEM=0 to run without symmetric memory."
        )
    if envs.VLLM_NCCL_SYMM_MEM_LOG:
        logger.info(
            "NCCL symm scratch %s: rank %d/%d segments %d size %s offset %s "
            "bytes %d (symmetric)",
            what,
            pynccl_comm.rank,
            pynccl_comm.world_size,
            *mine,
        )


def get_symm_scratch(
    pynccl_comm: PyNcclCommunicator,
    role: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    ubatch: int = 0,
) -> torch.Tensor:
    """Grow-only symmetric scratch of at least ``shape[0]`` rows, keyed by the
    communicator, role, ubatch and row shape; returns a leading slice. Growth
    doubles, so outgrown buffers total less than the live one. The caller owns
    the slice only until the next call with the same key: copy results out."""
    assert shape, "symmetric scratch buffers require at least one dimension"
    key = (
        bytes(pynccl_comm.unique_id.internal),
        role,
        ubatch,
        tuple(shape[1:]),
        dtype,
        device,
    )
    buf = _symm_scratch.get(key)
    rows = shape[0]
    if buf is None or buf.shape[0] < rows:
        capacity = rows if buf is None else max(rows, 2 * buf.shape[0])
        with nccl_symm_mem_context(pynccl_comm):
            new_buf = torch.empty((capacity, *shape[1:]), dtype=dtype, device=device)
        check_symmetric_placement(pynccl_comm, new_buf, f"scratch {role}")
        if buf is not None:
            _symm_scratch_retired.append(buf)
        buf = new_buf
        _symm_scratch[key] = buf
    return buf[:rows]


_placement_mismatches = 0


def report_symmetric_placement(
    pynccl_comm: PyNcclCommunicator, tensors: tuple[torch.Tensor, ...], what: str
) -> None:
    """Diagnostic for VLLM_NCCL_SYMM_MEM_LEGACY_PERCALL: compare the placement
    of per-call symmetric tensors across ranks (one CPU all-gather per call)
    and log mismatches instead of raising, to observe the legacy behavior."""
    global _placement_mismatches
    if pynccl_comm.group is None or torch.cuda.is_current_stream_capturing():
        return
    mine = tuple(_symm_placement(t)[1:] for t in tensors)
    group = pynccl_comm.group
    if isinstance(group, torch.distributed.ProcessGroup):
        gathered: list[Any] = [None] * pynccl_comm.world_size
        torch.distributed.all_gather_object(gathered, mine, group=group)
    else:
        gathered = group.all_gather_obj(mine)
    if any(other != mine for other in gathered):
        _placement_mismatches += 1
        if _placement_mismatches <= 50 or _placement_mismatches % 1000 == 0:
            logger.warning(
                "NCCL symm placement MISMATCH #%d %s rank %d/%d: "
                "(segment size, offset) per tensor per rank: %s",
                _placement_mismatches,
                what,
                pynccl_comm.rank,
                pynccl_comm.world_size,
                gathered,
            )
