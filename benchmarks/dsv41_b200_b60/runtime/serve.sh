#!/usr/bin/env bash
# pb role server: our vllm venv (VENV) + a per-role vllm overlay (canlin best3 patches ported) + shared Mooncake P store.
# usage: ROLE=prefill|decode GPUS=.. PORT=.. RPC_PORT=.. DP_MASTER_PORT=.. NIXL_SIDE_CHANNEL_PORT=.. NIXL_HOST=..
#        NUMA="0 0 0 0" [DP=4] [MC_CONFIG=..] serve.sh
set -euo pipefail
W=/gpfs/mszn/workspace/dsv41-di-vllm-20261021
PB=$W/pb
# VENV: the vllm venv the overlays mirror (default f4edf9f0; the 2026-10-25 rebase is rebase/94a29d31-20261025).
V=${VENV:-$W/rebase/f4edf9f0-20261023/.venv}
ROLE=${ROLE:?}
case "$ROLE" in prefill) OV=$PB/${OV_P:-ov-p};; decode) OV=$PB/${OV_D:-ov-d};; *) echo "bad ROLE" >&2; exit 2;; esac
export PATH=$V/bin:$W/venv-vllm/bin:/usr/local/cuda-13.1/bin:$PATH CUDA_HOME=/usr/local/cuda-13.1
export CPATH=$W/sysinc/pyheaders2/usr/include/python3.12:$W/sysinc/pyheaders2/usr/include:$W/sysinc/sysroot/usr/include${CPATH:+:$CPATH}
# The overlay must shadow site-packages' vllm; nixl 1.5.0 and Mooncake come from their own dirs.
export PYTHONPATH=${PB_EXTRA_PYTHONPATH:+$PB_EXTRA_PYTHONPATH:}$OV:$W/nixl-overlay:$PB/mcpath
export CUDA_VISIBLE_DEVICES=${GPUS:?} VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_RUST_FRONTEND=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1
export VLLM_DP_MASTER_PORT=${DP_MASTER_PORT:?} VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_RPC_TIMEOUT=600000
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/home/zetyun/.cache/vllm-pb-$ROLE}
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_SERVER_DEV_MODE=1
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=${VLLM_RANDOMIZE_DP_DUMMY_INPUTS:-1} VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_RS_TOKEN_IDS_AS_BYTES=1
export NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=1
export UCX_TLS=cuda_copy,cuda_ipc,rc,tcp UCX_MEMTYPE_CACHE=n UCX_MEMTYPE_REG_WHOLE=n UCX_RNDV_PIPELINE_ERROR_HANDLING=y
IB_NICS=$(sed -n 's/^IB_NICS=//p' $PB/runtime/nodes.env)
export UCX_NET_DEVICES=$(echo "$IB_NICS" | sed 's/,/:1,/g; s/$/:1/')
export NIXL_UCX_SGL_ENABLE=1 NIXL_LOG_LEVEL=${NIXL_LOG_LEVEL:-INFO}
export VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_SIDE_CHANNEL_PORT:?} VLLM_NIXL_SIDE_CHANNEL_HOST=${NIXL_HOST:?}
# canlin best3 host paths (env-gated in the overlays): int32 token ids, DP-synchronized GC.
export VLLM_COMPACT_TOKEN_IDS=${VLLM_COMPACT_TOKEN_IDS:-1} VLLM_GC_SYNC_STEPS=${VLLM_GC_SYNC_STEPS:-16}
SPECCFG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
NIXL='"kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"backends":["UCX"],"kv_lease_duration":300}'
if [ "$ROLE" = prefill ]; then
  export VLLM_PREFILL_STEP_COST=${VLLM_PREFILL_STEP_COST:-'{"ctx_budget_ms":40,"skip_scan":16,"e":6.83366,"a":0.00587,"b":1.32e-8,"b2":0,"d":0.94755,"g":4e-6}'}
  export VLLM_PD_HOSTOPT=${VLLM_PD_HOSTOPT:-1}
  export MOONCAKE_CONFIG_PATH=${MC_CONFIG:?} MOONCAKE_REQUESTER_LOCAL_HOSTNAME=$NIXL_HOST
  export MC_STORE_MEMCPY=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1 MC_SLICE_SIZE=1048576 MC_WORKERS_PER_CTX=4
  export VLLM_MOONCAKE_LOAD_RECV_THREADS=4
  # P_LOOKUP_PREFETCH_LIMIT/_TTL (#61292 input-thread lookup prefetch) and P_EARLY_RECV (#63936 early Store-load
  # reporting) exist only in the PR-synced P overlay (ov-p80+); unset, the JSON is unchanged.
  STORE="{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"load_async\":true,\"lookup_async\":true${P_LOOKUP_PREFETCH_LIMIT:+,\"lookup_prefetch_limit\":$P_LOOKUP_PREFETCH_LIMIT}${P_LOOKUP_PREFETCH_TTL:+,\"lookup_prefetch_ttl_s\":$P_LOOKUP_PREFETCH_TTL}}}"
  ROLE_ARGS=(
    --kv-transfer-config "{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",$([ "${P_EARLY_RECV:-0}" = 1 ] && echo '"enable_early_recv":true,')\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",$NIXL},$STORE]}}"
    --engram-config "{\"cpu_offload\":$([ "${P_ENGRAM_CPU:-1}" = 1 ] && echo true || echo false),\"embedding_across_dp\":true,\"use_thp\":false${P_ENGRAM_PREFETCH:+,\"host_prefetch_threads\":$P_ENGRAM_PREFETCH}}"
    --max-num-seqs 64 --max-num-batched-tokens ${MAX_BATCHED:-16384}
    --enable-prefix-caching --long-prefill-token-threshold ${LONG_PREFILL:-8192} --prefix-cache-retention-interval 32768
    --prefill-schedule-interval 1
    --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046],"decoder_replay_cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2048],"max_cudagraph_capture_size":2046}'
  )
else
  export VLLM_FAST_BLOCK_HASH=${VLLM_FAST_BLOCK_HASH:-1} VLLM_DSV41_FUSED_DRAFT_SAMPLE=${VLLM_DSV41_FUSED_DRAFT_SAMPLE:-1}
  MS=${MAX_SEQS:-256}
  SIZES=$(seq -s, 6 6 $(( 6 * MS )))
  DPC=()
  [ "${D_PREFIX_CACHE:-1}" = 1 ] && DPC=(--enable-prefix-caching --prefix-cache-retention-interval 32768) || DPC=(--no-enable-prefix-caching)
  ROLE_ARGS=(
    --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",$NIXL$([ "${D_EARLY_RECV:-0}" = 1 ] && echo ',"enable_early_recv":true')}"
    --engram-config "{\"cpu_offload\":$([ "${D_ENGRAM_CPU:-0}" = 1 ] && echo true || echo false),\"embedding_across_dp\":true,\"use_thp\":false}"
    "${DPC[@]}"
    --max-num-seqs $MS --max-num-batched-tokens ${D_MBT:-$(( 10 * MS ))}
    --prefill-schedule-interval ${D_SCHED_INTERVAL:-2}
    --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"max_cudagraph_capture_size\":$(( 6 * MS )),\"cudagraph_capture_sizes\":[$SIZES]}"
  )
fi
EXTRA=()
[ -n "${EPLB_CFG:-}" ] && EXTRA+=(--enable-eplb --eplb-config "$EPLB_CFG")
[ -n "${SCHED_POLICY:-}" ] && EXTRA+=(--scheduling-policy "$SCHED_POLICY")
[ -n "${DP_SYNC_INTERVAL:-}" ] && EXTRA+=(--dp-sync-interval "$DP_SYNC_INTERVAL")
# GRPC_PORT: Rust-frontend gRPC endpoint for a Dynamo sidecar. Dynamo parses tool calls and reasoning itself and
# refuses an engine that runs its own reasoning parser, so the engine then runs without parsers (as canlin's serve_pd.sh).
PARSERS=(--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41)
[ -n "${GRPC_PORT:-}" ] && { EXTRA+=(--grpc-port "$GRPC_PORT"); PARSERS=(); }
# EXTRA_ARGS: extra vllm serve flags (word-split), e.g. --no-async-scheduling.
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)
# NSYS_OUT=<prefix>: run under nsys; each POST /start_profile captures NSYS_ITERS engine steps per DP rank.
NSYS=() PROF=()
if [ -n "${NSYS_OUT:-}" ]; then
  NSYS=(/gpfs/mszn/workspace/dsv41-agentx-profile-20260923/tools/2025.5.2/target-linux-x64/nsys profile
    -t cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=repeat
    --kill=none --cuda-graph-trace=node -o "$NSYS_OUT" --force-overwrite=true)
  PROF=(--profiler-config "{\"profiler\":\"cuda\",\"detailed_trace_annotation\":true,\"ignore_frontend\":true,\"max_iterations\":${NSYS_ITERS:-200}}")
fi
cd /tmp
exec "${NSYS[@]}" $V/bin/vllm serve /gpfs/mszn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --host 0.0.0.0 --port ${PORT:?} \
  --data-parallel-size ${DP:-4} --data-parallel-rpc-port ${RPC_PORT:?} --tensor-parallel-size 1 \
  --enable-expert-parallel --enable-ep-weight-filter \
  --language-model-only --tokenizer-mode deepseek_v41 "${PARSERS[@]}" \
  --kv-cache-dtype fp8 --max-model-len 1048576 --gpu-memory-utilization ${GPU_UTIL:-0.95} \
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kernel-config '{"enable_flashinfer_autotune":true,"moe_backend":"deep_gemm_mega_moe","sparse_indexer_topk_backend":"deep_select"}' \
  --speculative-config "$SPECCFG" \
  --numa-bind --numa-bind-nodes ${NUMA:?} \
  --disable-uvicorn-access-log --enable-prompt-tokens-details \
  "${ROLE_ARGS[@]}" "${EXTRA[@]}" "${PROF[@]}"
