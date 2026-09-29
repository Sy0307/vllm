# 2026-09-29 更新：发布版本与重新部署

本节作为部署手册开头的最新操作说明。源码采用下列版本；既有手册中的旧镜像、旧源码分支和裸机 200 GB Mooncake 配置不作为本次发布输入。本节是待执行的发布方案：新镜像尚未构建，完整模型验收和生产发布尚未执行。

## 1. 本次需要更新的内容

| 项目 | 发布要求 |
| --- | --- |
| 源码分支 | [k3/unified-spec55-frozen-mm57368-20260928](https://github.com/Sy0307/vllm/tree/k3/unified-spec55-frozen-mm57368-20260928) |
| 代码提交 | [c5f2e6f65e2ff8be05540b461f239051634be5bc](https://github.com/Sy0307/vllm/commit/c5f2e6f65e2ff8be05540b461f239051634be5bc)，包含原 fb513459 多模态 CPU 输入释放与共享存储改动 |
| KDA 层 | 初始化 FlashKDA 检查点导出器时，显式传入 `state_len=self.conv_size-1` |
| 检查点导出器 | `state_len` 改为必填参数，按逻辑卷积历史长度导出 |
| 本模型状态规格 | 卷积宽度 4、有效历史 3；DSpark block4 下物理缓冲区仍为 7 格 |
| 发布方式 | 构建新的不可变镜像；完整启动两节点新副本，使用新的 GPU 缓存及独立 Mooncake 缓存空间 |

运行时替换的源码文件共两处，必须作为同一版本同时打包：

```text
vllm/models/kimi_k3/nvidia/kda.py
vllm/model_executor/layers/mamba/kda_checkpoint.py
```

## 2. 构建发布物

1. 从上述代码提交导出两处源码。保留当前多模态补丁、已冻结的原生库、Python 环境、FlashInfer cubin 和库加载顺序。
2. 基于当前 `mm57368-fb513459-20260928-candidate` 镜像构建新镜像，将两处文件放入 `/opt/k3/overlay/vllm/` 对应目录。使用已经核对的基础镜像内容 ID 或仓库 digest；不能只依赖可被覆盖的 tag。保留现有 entrypoint 与 LWS 启动方式。
3. 重新生成 `/opt/k3/SNAPSHOT-SHA256SUMS` 中对应文件的记录及发布清单；更新 `/opt/k3/source.lock.json`、`VLLM_BUILD_COMMIT`、镜像 revision 标签，并同步启动校验使用的 manifest 摘要。保留原基础版本记录。
4. 使用新的唯一镜像标签，例如 `mm57368-convhist3-c5f2e6f65-20260929-r1`。记录构建后的镜像 digest、归档校验和、源码提交及两处源码哈希；这些值以实际构建结果为准。
5. 将新镜像分发到新副本的两个节点，确认内容一致。运行时 `PYTHONPATH` 指向 `/opt/k3/overlay`；不能用旧 overlay 的挂载覆盖新文件。依赖随原镜像保留，不执行 `pip install -U`。

对应代码提交的文件 SHA256：

```text
3cbfb5898c3f0eccff602007217040b1a94a31798cfc49246477004f7d74e8c7  vllm/models/kimi_k3/nvidia/kda.py
561bc23d0d3691b154fdec0cd8990eff86521f843145dfab053684a3f1522eb0  vllm/model_executor/layers/mamba/kda_checkpoint.py
```

发布记录应区分镜像内容 ID、仓库 manifest digest 与 OCI 归档 SHA256，三者不能混写。

## 3. 准备新副本与缓存配置

在空闲且已核对资源的独立两节点上创建新 LWS 副本和独立 Service；可使用 prod24/prod25。新副本的标签不能提前被现有生产 Service 或网关选中。原 G0/G1 保持运行状态和配置，先完成新副本验收。

从当前生产 LWS 与运行时配置复制出新发布资源，使用新的 ConfigMap 名称和发布标识；不要直接重放旧裸机手册。

| 配置 | 本次沿用值 |
| --- | --- |
| 并行与分层 | TP8、PP2、DCP8/a2a、EP；PP 分层 48/45 |
| 投机 | DSpark K4、draft greedy、rejection block；开启 RecoverSSM |
| KV 与调度 | FP8 KV、prefix-match-unit 128；max-seqs 192、batched tokens 8768、scheduled tokens 8192 |
| 图模式 | FULL_AND_PIECEWISE；沿用当前图桶 |
| Mooncake | 每 worker 160 GB segment、4 GB buffer；异步 lookup/load；`experimental_recoverssm_store=true` |
| 图片输入 | 保留当前多模态配置和共享存储参数，不添加 `--language-model-only` |

在原 `kv_connector_extra_config` 中增加版本与副本专用的 `cache_prefix`，保留其余原有字段。以下仅为新 canary 的示例：

```json
{
  "load_async": true,
  "lookup_async": true,
  "enable_offload": false,
  "experimental_recoverssm_store": true,
  "cache_prefix": "k3-h3-c5f2e6f65-20260929-r1-canary"
}
```

同一两节点副本的全部 rank 使用相同前缀。其他独立副本使用不同后缀；后续发布使用新的发布标识。该前缀只隔离 Mooncake，GPU APC 由全新的 worker 进程重新建立。

新副本启动自己的 Mooncake master/store，指向本副本 leader，初始为空；不连接原副本的池，不恢复之前的状态快照。保留当前容器内存限制下的 160 GB 配置，不能直接套用裸机 200 GB 配置。确认新 Pod 实际 cgroup 内存余量、16 个 rank、IB、端口和两级节点资源均满足启动要求。

## 4. 启动与接流量前验收

1. 启动新副本的 leader 与 worker，等待全部 16 个 rank 就绪。核对两个 Pod 的实际镜像 digest、进程加载路径、两处源码哈希、命令行、投机参数和 cache_prefix。只更新分支或只重启 API 进程不构成完整发布。
2. 按现有 LWS admission 机制对新 Pod UID 完成准入。准入文件与服务路由分别检查；健康检查通过不等于已经通过接流量验收。
3. 在新副本使用固定输入、贪心采样和真实 block 验证，分别验证冷算、GPU 本地前缀命中与 Mooncake 外部恢复；覆盖分块预填充，包含 11648、23168 等缓存边界。比较生成 token 与 logprob/logits；差异超过既定容差时停止发布并保留结果。
4. 本地热请求应有实际缓存命中；外部恢复必须观察到该次请求对应的外部加载 token/读取计数增长，不能只凭总 cached_tokens 判断。必要的 GPU 缓存重置仅在尚未接生产流量的新副本执行。
5. 再验证图片输入、混批并发、请求取消和继续命中。禁止使用 synthetic acceptance 代替真实采样验证。检查请求错误、输出有效性、TTFT、ITL、单请求 TPS、总体吞吐、GPU KV 与主机内存占用。
6. 将新镜像和新配置的验收结果填入发布记录。尚未完成上述步骤时保持“待完整模型验收”，不沿用历史版本的门禁或吞吐成绩作为本次验收结果。

## 5. 切流与回退

新副本验收通过后，先少量切流，观察稳定后再逐组扩大。每次操作以完整两节点组为单位，避免两级混用不同镜像或缓存前缀。已有生产副本的替换另行安排，不能在新副本验收前重启原 G0/G1。

如需回退，先撤回新副本流量，切回已经完成验收的独立副本，或以已验证版本和独立新缓存重新启动。记录失败发布的镜像、配置与日志；不要将不同发布版本重新接入同一个缓存空间。
