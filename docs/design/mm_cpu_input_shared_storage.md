# Optional local backing storage for CPU multimodal inputs

This is an alternative transport proposal extracted from PR #57368. It targets
full CPU tensor inputs sent by the multiprocess executor. It does not change
worker input lifetime, encoder computation, or processor-cache ownership.

Main already contains the opt-in tensor arena from PR #51207. This alternative
uses per-input COW files rather than reusable pinned slots: it supports mixed
local/remote readers and noncontiguous tensors, but adds file operations and
does not pin its mappings. The two transport options cannot be enabled together.
A current-main comparison and maintainer agreement on the transport design are
required before proposing this alternative for merge.

## Configuration and ownership

Set `VLLM_MM_INPUT_SHARED_STORAGE_PATH` to an existing, writable, node-local
POSIX directory visible at the same absolute path to the multiprocess executor
and its local workers. Leave it unset for the ordinary transport.

```bash
export VLLM_MM_INPUT_SHARED_STORAGE_PATH=/dev/shm/vllm-mm-inputs
```

Provision that directory and its capacity before launching the service. A
memory-backed mount consumes the pod's memory budget; sharing does not make the
backing bytes free. Do not use a network filesystem. The existing 64 GiB `/dev/shm`
allocation must not be assumed sufficient for an unrestricted image workload.

Only CPU tensors of at least 1 MiB inside newly admitted multimodal features
use this path. Smaller fields and unsupported tensor types retain the ordinary
serialization. Other RPC tensors are not opted in. Each selected tensor gets a
single backing file; all local workers map that file. The writer still retains
its EngineCore input for replay, independently of this worker-shared allocation.

PyTorch has no read-only Tensor type. The implementation uses private,
copy-on-write file mappings with an immutable-input contract: reads share the
same physical pages, while an accidental in-place write cannot corrupt a peer.
Batching and device transfer can still allocate transient buffers. Do not sum
worker RSS to estimate shared physical bytes; use PSS and cgroup memory usage.

Each reader acknowledges only after mapping. The last acknowledgement unlinks
the file; kernel mapping lifetime then follows the surviving tensor storages.
Ring-buffer reuse cannot overwrite these inputs. The file API handles writes,
so storage exhaustion raises an allocation/write error before publishing the
input, instead of causing a mapped-write SIGBUS or silently restoring N copies.

Normal executor teardown removes files for readers that failed before mapping.
An advisory owner lease protects active stores. After a writer is killed, a new
store in the same directory reaps abandoned stores with released leases. This
recovery happens at startup, not continuously while the service is down.

Remote readers receive the ordinary tensor bytes, never local file paths. In the
TP8 x PP2 deployment with all eight first-stage TP workers on the driver node,
this removes those eight independent retained CPU allocations. This version
does not introduce a second shared-storage broadcaster on remote nodes, so it
does not provide node-wide sharing for TP groups spanning multiple nodes.

## Validation

Transport tests cover eight separate readers mapping one inode, BF16 and
noncontiguous inputs, a slow reader, ring and overflow paths, remote fallback,
copy-on-write isolation, unlink with live views, serialization failure, and
writer-crash recovery without removing another live store. Arena tests are kept
and run to verify that the existing transport remains available when this
alternative is unset.

Earlier B200 validation combined shared storage with early input release. It
does not measure this transport independently or compare against the arena now
on main. Before rollout, compare deterministic model outputs, CPU PSS/cgroup
memory, H2D latency, throughput, and allocation failures with both current
transport options.
