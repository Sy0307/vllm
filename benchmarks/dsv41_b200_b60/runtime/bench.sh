#!/usr/bin/env bash
# bench.sh RUN DUR [CONC]: AgentX C2048 against the Dynamo frontend on this node (P or D host), background-safe.
set -euo pipefail
RUN=${1:?}; DUR=${2:-1800}; CONC=${3:-2048}
PB=/gpfs/mszn/workspace/dsv41-di-vllm-20261021/pb
H=/gpfs/mszn/workspace/agentx-harness-89b2186
OUT=$PB/runs/$RUN/aiperf-$(date +%Y%m%dT%H%M%S)
mkdir -p $OUT; echo $OUT > $PB/runs/$RUN/LATEST_AIPERF
export HF_HOME=$H/hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/gpfs/mszn/data/k3-benchmarks/ref440542/tokenizer-path-compat
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800 TERM=xterm-256color
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
. $PB/runtime/nodes.env
export NO_PROXY=127.0.0.1,localhost,$P_IP,$D_IP no_proxy=127.0.0.1,localhost,$P_IP,$D_IP
exec $H/venv/bin/aiperf profile --scenario agentx \
  --url http://127.0.0.1:8300 --endpoint /v1/chat/completions \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer /gpfs/mszn/models/deepseek-ai/DeepSeek-V4.1-Flash \
  --concurrency $CONC --benchmark-duration $DUR \
  --failed-request-threshold 0.10 --warmup-requests-per-lane 10 \
  --trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 \
  --tokenizer-trust-remote-code --unsafe-override --random-seed 42 \
  --server-metrics http://$P_IP:8301/metrics http://$P_IP:8302/metrics http://$D_IP:8311/metrics \
  --output-artifact-dir "$OUT/aiperf_artifacts" \
  --public-dataset semianalysis_cc_traces_weka_062126 > $OUT/aiperf.log 2>&1
