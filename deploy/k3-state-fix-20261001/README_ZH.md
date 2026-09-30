# 2026-10-01 状态修复版本：修复说明、发布与部署

本文件随分支发布，面向开发与运维。飞书运维手册开头的“2026-10-01 更新”一节是本文件部署部分的摘要；两者不一致时以本文件和实际发布清单为准。

## 1. 版本

| 项目 | 值 |
| --- | --- |
| 源码分支 | `k3/unified-spec55-frozen-mm57368-20260928` |
| 修复提交 | `918d60999c3e4849319cffd487f7afd4b928ad73` |
| 基线 | `c5f2e6f65`（convhist3，2026-09-29 发布说明中的版本）+ `df878b8`（仅文档） |
| 运行时基线镜像 | `localhost/kimi-k3-unified:mm57368-convhist3-c5f2e6f65-20260929-r1`（9-30 回滚前生产所用） |
| 发布方式 | 在基线镜像 `/opt/k3/overlay/vllm/` 中替换 10 个源文件，构建新的不可变镜像 |

## 2. 修复内容

按严重性排序。每项都给出影响范围和修改位置。

### 2.1 RecoverSSM 提交写错状态块（最严重）

- 现象：投机解码（DSpark K4 + `--use-replayssm`）时，某一步接受后的 token 数恰好落在 Mamba 块边界（`N + A` 是块大小 `B` 的整数倍），状态被提交到下一列，而这一列还不属于该请求。
  - 下一列为空时：本步接受的 recurrent/conv 状态没有保存，之后从旧状态继续，输出逐渐错乱。
  - 下一列残留别的块号时：状态写进不属于当前请求的块。
  - 残留块含 NaN 时：logits 全 NaN，采样固定选出 token 1023（` @`）或 8191（`itable`），配合 block 验证接受草稿，表现为 `@ting. The` 这类固定片段反复出现。
- 触发频率：每次跨越块边界时，约有 1/平均接受长度 的概率恰好停在边界上，属于高频事件。
- 修改：`vllm/models/kimi_k3/nvidia/ops/recoverssm.py`（提交计划的 final column）与 `vllm/v1/worker/gpu/model_states/recoverssm.py`（running column），均由 `(N+A)//B` 改为 `(N+A-1)//B`；无效源块或零接受长度时屏蔽块表读取，零提交时不推进状态。
- 上游状态：vLLM main 截至 2026-10-01 仍是原写法（来源 #51855，2026-08-17）。上游 K3 RecoverSSM + PP（#56121）尚未合入，因此上游用户基本不会跑到该组合。

### 2.2 RecoverSSM 提交后又执行一次通用 align 复制

- 现象：RecoverSSM 已把接受后的状态写好，随后 `mamba_hybrid.py` 仍无条件执行通用 `run_fused_postprocess_align`。在 PP 延后提交下，两次操作读到的块表可能属于不同批次，第二次复制会覆盖另一条正在运行请求的状态。完整模型实测捕获到跨请求覆盖。
- 修改：`vllm/v1/worker/gpu/model_states/mamba_hybrid.py` 在 `self.recoverssm is not None` 时跳过通用复制。不开 RecoverSSM 的路径不变。

### 2.3 MLA context split 空分段参与合并

- 现象：`VLLM_K3_CONTEXT_SPLIT_KV=1` 时，一个 chunk 的分段数由最长历史决定。短历史请求会被切出空分段；空分段的输出没有被 kernel 写入，仍是未初始化内存，合并时 `NaN * 0 = NaN`。多模态编码预算截断等情况可以产生这种形状。
- 修改：`vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py`，当 chunk 内最短历史小于分段数时退回不分段路径（3 行，用 CPU 长度判断，不引入 GPU 同步）。安全形状的结果与原来逐位相同。

### 2.4 align 状态复制按批次行号取块表（公共路径）

- 现象：V2 runner 在 PP 下延后执行的状态复制，使用的是“按批次排列、每步重新 gather”的块表视图，按批次行号取行。延后执行时，这些行可能已经对应别的请求。
- 修改：复制 kernel 统一按请求槽位（`req_idx`）取持久块表；metadata 使用独立的按批次 context，两者分开初始化与清理。与上游 #57807 方案一致。
- 文件：`vllm/v1/worker/gpu/model_runner.py`、`vllm/v1/worker/mamba_utils.py`、`vllm/v1/worker/gpu/model_states/mamba_hybrid.py`、`vllm/v1/worker/gpu/cudagraph_utils.py`。
- 注意：这是公共路径，不开投机时也会执行。V1 runner 调用时映射为空，行为不变。

### 2.5 取消请求后回收的 Mamba/KDA 块未清零

- 修改：`vllm/v1/core/single_type_kv_cache_manager.py` 记录新分配的 Mamba 块，`vllm/v1/worker/utils.py` 的 `KVBlockZeroer` 同时处理 conv、recurrent 与 RecoverSSM 记录张量。清零发生在状态预复制之前，不会清掉复制进来的状态。属于防御性修复。

### 2.6 Mooncake 同一批读取中出现重复 key 时漏写

- 现象：Mooncake SDK（实测 `0.3.13.post1`，PyPI 最新版）在一次 `batch_get_into_multi_buffers` 中含重复 key 时，只写最后一个目标地址，其余地址保留旧内容，却全部返回成功。上游修复 Mooncake #3929 于 2026-09-11 合入 main，截至 2026-10-01 没有任何正式版或 TestPyPI CUDA 13 包包含它。
- 修改：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` 在调用 SDK 前把批次切成若干段，每段内 key 不重复，按原顺序合并返回值。没有重复 key 时仍是一次调用。
- 后续：Mooncake 发布含 #3929 的正式版后，可在隔离环境验收后升级；本补丁保留无害。

### 2.7 本次不包含

- APC 短预填充卷积写共享初始块的修复（`causal_conv1d.py`）：K3 当前不走该路径（`IS_APC_ENABLED=False`），而该 kernel 为所有 Mamba/GDN 模型共用，本次不带入。补丁单独保存。
- Mooncake SDK 升级、FlashKDA 版本变更、采样参数默认值变更：均不在本次发布内。

## 3. 发布物（冻结）

### 3.1 本版本修改的 10 个文件及 SHA256

```text
80f49e711026109a2c7f8201289edeee7fa87c4f3e73be3f62d08fc0e4d8a46d  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
504142e02119eca23d514f18a1ef2f05b3904cf1e9ef714952f1fcdb8709cda2  vllm/models/kimi_k3/nvidia/ops/recoverssm.py
864f41e1647c2fbb44e280279b695bce1f149ae6910c6a9610d0fd8131c881f2  vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py
04741d2fa6ae89829246d416356422fce7da97d5c76bb0decb635aa5674cf7ac  vllm/v1/core/single_type_kv_cache_manager.py
3cbf167a2793fb3fa32b56be4ded030ce686cfd165cdc7d2cfd93d638b1413e9  vllm/v1/worker/gpu/cudagraph_utils.py
32b9df9ae931aa0f4284a3590c6a62ae37129098cd4cf93015f7a81030eb8897  vllm/v1/worker/gpu/model_runner.py
e9c71ba147d5c9ad9d412b3ebfadde7d56929917d42c0f3ebc007df95bc10006  vllm/v1/worker/gpu/model_states/mamba_hybrid.py
c0391f60e3a1e0a0140379562124277456706787598a16e8e3c6a215e2a19a4a  vllm/v1/worker/gpu/model_states/recoverssm.py
53fc9a9abc66088b0f136a0a8b89ee77f50817c0370a9b36e7da042c5946e0cd  vllm/v1/worker/mamba_utils.py
fbb80a7b223a4984898f9980089bfae8e759c92d300afe6104b64aae9f166c9a  vllm/v1/worker/utils.py
```

镜像中以下文件保持基线版本，发布时应核对不变：

```text
3cbfb5898c3f0eccff602007217040b1a94a31798cfc49246477004f7d74e8c7  vllm/models/kimi_k3/nvidia/kda.py
561bc23d0d3691b154fdec0cd8990eff86521f843145dfab053684a3f1522eb0  vllm/model_executor/layers/mamba/kda_checkpoint.py
cb16cc9250c4195c09d6d83b43bba607c2d749b621d8d135a920443d1577265f  vllm/model_executor/layers/mamba/ops/causal_conv1d.py
```

### 3.2 冻结发布目录（GPFS）

路径：`/gpfs/mszn/data/k3-benchmarks/k3-ops-statefix-918d6099-20261001/`

```text
overlay/vllm/                  冻结 overlay（5160 个文件）= convhist3 镜像 overlay + 3.1 的 10 个文件，无诊断插桩；即 6.2 验证所用的代码
overlay-SHA256SUMS             逐文件 sha256；本文件 sha256 = cdae37bd51d94fb5f0d4696930c9b8038848295180dc6cfa2e56f46b76d1a4a1
image-build/
  context/Containerfile        FROM convhist3 镜像 f3b530a2ada1…，COPY rootfs/ /，写入 VLLM_BUILD_COMMIT 与 revision 标签
  context/rootfs/opt/k3/       10 个修复文件、更新后的 SNAPSHOT-SHA256SUMS（a04a316a…）与 source.lock.json
  build.py                     在构建 convhist3 的 podman 主机上执行：构建、镜像内全量校验 overlay、导出 image.oci.tar、生成 lws-template.json
  runtime.json                 LWS 运行配置（release_integrity 已更新为新哈希；cache_prefix = statefix-918d6099-20261001-r1-{GROUP}）
  launcher-configmap.json      新 ConfigMap k3-statefix-launch-04d05ce6c038（launcher.py 与 9-29 相同，runtime.json 为上面的新版本）
  lws-template.PENDING-DIGEST.json  9-29 LWS 模板，仅替换 ConfigMap 名；镜像引用由 build.py 填入后生成 lws-template.json
  inputs.json / prepare_inputs.py   生成上述输入的记录与脚本（已执行并通过全部校验）
release-files/                 10 个修复文件及 3 个应保持不变的基线文件，RELEASE-SHA256SUMS
```

`prepare_inputs.py` 已校验：新 SNAPSHOT 中 overlay 部分与 `overlay-SHA256SUMS` 逐文件一致；`runtime.json` 含 `--use-replayssm` 与 `experimental_recoverssm_store=true`，不含诊断参数。

## 4. 运行配置

`image-build/runtime.json` 由 9-29 生产配置生成，只改了 4 处：`release`、`source.release_commit`、`release_integrity`（12 个文件哈希 + SNAPSHOT 摘要 `a04a316a3ae523c3be7f8bb97eb11484d697ed4254e9b10c6754100edc44f99d`）、`cache_prefix`。其余参数不变：

| 配置 | 值 |
| --- | --- |
| 并行 | TP8、PP2（48/45）、DCP8 a2a、EP deepep_v2 |
| 投机 | DSpark K4、draft greedy、`rejection_sample_method=block`、`--use-replayssm`（开启 RecoverSSM） |
| KV 与调度 | FP8 KV、prefix-match-unit 128、max-num-seqs 192、batched tokens 8768、scheduled tokens 8192 |
| Mooncake | 每 worker 160 GB segment、4 GB buffer，`experimental_recoverssm_store=true`，`cache_prefix=statefix-918d6099-20261001-r1-{GROUP}` |
| 环境变量 | `VLLM_K3_*` 与 9-29 相同，`VLLM_K3_CONTEXT_SPLIT_KV=1` 保持开启 |

LWS 启动器启动时会校验 `release_integrity` 中的文件哈希和 SNAPSHOT 摘要，镜像与 ConfigMap 必须同时使用本版本，混用会直接启动失败。

## 5. 部署步骤

流程与 2026-09-29 发布相同（参考 `/gpfs/mszn/data/k3-benchmarks/k3-haochi-convhist3-20260929-r1/` 中的 `imports.py`、`ops.py`、`run-rollout.py`）。

1. **构建镜像**：在构建 convhist3 镜像的 podman 主机上执行 `python3 /gpfs/mszn/data/k3-benchmarks/k3-ops-statefix-918d6099-20261001/image-build/build.py`。输出中必须出现 `IMAGE_OVERLAY_PASS` 与 `BUILD_COMPLETE`；记录 `build-result.json` 中的 image_id、image_ref（manifest digest）与 archive_sha256。
2. **导入镜像**：按 9-29 `imports.py` 的方式，在 4 个生产节点（0011/0019/0023/0029）用 k0s ctr 导入 `image-build/image.oci.tar` 并按 digest 打标签；任务名与标签换成本版本（如 `statefix-918d6099-import-<node>-1001`）。
3. **创建 ConfigMap**：`kubectl create -f image-build/launcher-configmap.json`（immutable）。
4. **更新 LWS 模板**：以 `image-build/lws-template.json`（build.py 生成，镜像为新 digest、ConfigMap 为 `k3-statefix-launch-04d05ce6c038`）替换 `kimi-k3-haochi` 的 `leaderWorkerTemplate`，用 partition 控制逐组滚动。先确认当前 LWS 模板与回滚后的实际状态，再做 JSON patch（带 uid/resourceVersion 校验）。
5. **逐组滚动**：每次一个两节点组，顺序与 9-29 相同：从路由摘除 → 等待无运行请求 → 释放 → 新 Pod 启动 → admission 准入 → 验收 → 重新加入路由 → 观察，然后再处理下一组。新 Pod 使用新的 `cache_prefix`，Mooncake 池为空，不复用旧快照。
6. **核对新 Pod**：镜像 digest 为新版本；日志无 Traceback；`use_replayssm=True`、`num_speculative_tokens=4`；启动器 precheck 通过（哈希不符会报 `Release source hash mismatch`）。

## 6. 验收

### 6.1 发布前必须通过

- 启动日志无 Traceback / EngineDeadError；日志中 `use_replayssm=True`、`num_speculative_tokens=4`。
- 固定输入冷算、GPU 本地前缀命中、Mooncake 外部恢复三者比较（与 9-29 版本相同的门禁，覆盖 11648、23168 等边界）。外部恢复必须观察到该请求的外部加载计数增长。
- 图片输入、混批并发、请求取消与继续命中。
- 高并发压测：输出全文扫描无 token 1023/8191 连续片段；`Corrupted` 计数为 0；记录吞吐、TTFT、ITL、运行中请求数、抢占次数。

### 6.2 已完成的隔离验证（prod24/25，2026-10-01）

使用 3.2 的冻结 overlay（与镜像 overlay 逐文件一致）和第 4 节运行参数，TP8×PP2×DCP8、K4 + RecoverSSM、FP8 KV、MooncakeStore kv_both、CUDA graph，在 prod24/25 两机启动：

| 阶段 | 请求 | 结果 |
| --- | --- | --- |
| 启动 | 16 rank | 就绪，无 Traceback；`use_replayssm=True`；KV 容量 36,390,353 tokens |
| 64 并发，~6K 输入 / 1024 输出 | 64 | 全部成功；生成吞吐 2328 tok/s；接受长度 2.54；无抢占 |
| 384 并发，~6K 输入 / 1024 输出 | 384 | 全部成功；运行中请求 192（max-num-seqs 上限）；生成吞吐 3662 tok/s；接受长度 2.54；无抢占；KV 使用峰值 26% |
| 输出检查 | 全部 452 条 | `Corrupted: 0 reqs`；同一 token 最长连续 2 次；无 `@ting`/`itable` 固定片段 |

本次隔离验证不含 Mooncake 外部恢复门禁与图片输入，这两项按 6.1 在新 Pod 上执行。

## 7. 上线后监控

- `vllm:num_preemptions_total` 增长速度、运行中/等待请求数、`kv_cache_usage_perc`。
- 服务日志中 `Corrupted: N reqs` 必须保持 0。
- 抽样扫描输出中 ` @` / `itable` 连续出现、长周期重复。

## 8. 回退

先撤回新副本流量，切回已验收的副本；或以已验证版本和新的 `cache_prefix` 重启。不同发布版本不得接入同一缓存空间。记录失败版本的镜像、配置与日志。
