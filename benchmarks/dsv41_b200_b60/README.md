# DeepSeek V4.1 Flash B200 / B60 复现归档

本分支冻结 2026-11-02 的 B60（`pb-b60-1102a`）现有代码和参数，不修改原实现，不包含新跑分。

- **成绩**：16 张 B200（8P+8D）、C2816；全程 **100,225.93 tok/s**，稳态 **112,512.96 tok/s**；稳态 TTFT 均值/p90 **1.5545/2.5075 s**，ITL **10.7933 ms**，正式错误 **0**。
- **口径**：全程 usage completion tokens / 实际约 1840 s；稳态吞吐按 [600,1800) 完成请求，TTFT 按该窗口发起请求。B60 全程 TTFT 均值 3.163 s，突发段 11.555 s。
- **基座**：`94a29d316e8c1d6dcf6ff75645c2e61c7e3c83fd`；依赖版本和成绩见 [results.json](results.json)。
- **代码**：[P 的 45 个快照文件](overlays/prefill/vllm) + [D 的 38 个快照文件](overlays/decode/vllm)，来源和 SHA256 见 [source_manifest.json](source_manifest.json)。这是此前已核对的本地部署快照，本次未再次连接 GPU 节点。
- **范围**：仅完整保存 B60。B56 指标是参考，它使用 `ov-p84`，不能把这里的 P 快照当成 B56 的精确代码。

- **详细分位数**：[request_metrics.json](request_metrics.json) 保存两组正式全程的 per-user TPS、TTFT、ITL 和请求总耗时，含 p1/p5/p10/p25/p50/p75/p90/p95/p99。全程包含突发阶段，与稳态口径分开。

## 优化与对应提交

提交按归档文件主题分组；共享文件可能包含多个机制。表中提交是实测文件的归档入口，**不是独立消融或可逐项相加的性能补丁**。完整代码取分支最终版本。

| 优化 | 做了什么 | 已有证据与限制 | 归档 commit |
|---|---|---|---|
| P 调度 | 短任务优先（residual SJF）、动态 token 上限、DP 共享整步预算，缩短排队和过长的计算步。 | 与默认先来先服务、关闭这三项优化相比，平均首 token 等待从 1.84 秒降到 1.55 秒，90% 请求的等待从 3.28 秒降到 2.51 秒。三项一起测试，各自贡献尚不清楚。 | [eb0e76832](https://github.com/Sy0307/vllm/commit/eb0e76832c31955a49bceb4fa3d54f212c55b7a2) |
| KV 数据通路 | 输入线程 lookup prefetch、MultiConnector、early receive 通知线程、Mooncake load kick；把查找、加载及通知前移。 | 与关闭 load kick 相比，打开后平均首 token 等待降低约 6.3%，吞吐基本不变。 | [af39cf4ce](https://github.com/Sy0307/vllm/commit/af39cf4ce163bddec144c9b0345152b74185fec5) |
| Engram 主机行预取 | P 的 CPU offload 配合 8 线程行预取，与 GPU 执行重叠。 | 在单独开关行预取的端到端测试中，平均首 token 等待降低约 8.5%，输入处理耗时降低约 10.6%，吞吐提高约 0.33%。 | [a0af0237f](https://github.com/Sy0307/vllm/commit/a0af0237ff0e1c4b2ec007164c34f5aa7982b480) |
| D 分配与准入 | 跳过无操作 KV allocation、defer promote；配合 host core 中的 input gate。 | 在当前配置下，这一项能提升多少还没有单独测出。 | [b6892dc74](https://github.com/Sy0307/vllm/commit/b6892dc748b592d106eee7fbce81c15ceeeb0a2f) |
| 融合 draft sampling | 保留融合 kernel 与 NaN、边界修正，减少 draft 采样开销。 | 在当前配置下，这一项能提升多少还没有单独测出；固定模拟接受率也不能说明生成质量。 | [f29936ccc](https://github.com/Sy0307/vllm/commit/f29936cccce36c7ff931c02a4eb4697d33a5f56d) |
| MXFP8 预编译 | 提前编译可能使用的 GEMM tactic，避免服务中编译停顿。 | 已有运行时 compile stall 被消除的记录；没有独立吞吐结论。 | [597d66921](https://github.com/Sy0307/vllm/commit/597d6692195d1b831d0876dfe3aeb9dcde8cbadf) |
| 主机与模型基础优化 | compact int32 TokenIds、fast hash、每 16 步同步 GC、input gate，以及模型/runner 配套文件。 | 包含继承的 Canlin 基础优化与其他同期文件；未拆分单项收益。 | [f184ccabf](https://github.com/Sy0307/vllm/commit/f184ccabfe55305c4aa9577b08ada0542e2cc15b) |
| NIXL 断连恢复 | 失效 peer 清理、退避和重新握手，避免一次断连变成持续请求失败。 | 可靠性修正；网络根因未消除。 | [040885c57](https://github.com/Sy0307/vllm/commit/040885c570c0d5020dc950cda49260c6f5f837b0) |
| 部署参数 | frontend 放 D、绑核限线程、路由策略、B60 P/D knob。 | 性能取决于完整配置及外部依赖。 | [f8661ae9f](https://github.com/Sy0307/vllm/commit/f8661ae9f149519e8ac9c154c079887e3d45147e) |

公开 PR 来源：[#63932](https://github.com/vllm-project/vllm/pull/63932)、[#63933](https://github.com/vllm-project/vllm/pull/63933)、[#63934](https://github.com/vllm-project/vllm/pull/63934)、[#63936](https://github.com/vllm-project/vllm/pull/63936)、[#63938](https://github.com/vllm-project/vllm/pull/63938)、[#63940](https://github.com/vllm-project/vllm/pull/63940)、[#64500](https://github.com/vllm-project/vllm/pull/64500)。这些是开发来源；归档中还包括 PR 移植和本地修正，不能用 PR 当前 head 替代。

本地修正来源为 `4ac0ab768e12107a313aeedd3748fc81148e6c1f`（lookup 前截断 P prompt）与 `3c65a5dbafb04369ceb51503546d2943ccca8c31`（通知式 early receive）；相应实现已保存在本分支快照。最新上游已合入的 BF16 DeepSelect、compressed KV gather 复用、MegaMoE replica mapping、MegaAttn positions，不应继续算成相对当前 main 的私有优势。

## 使用方式

根目录 `vllm/` 保持冻结基座；归档在 `benchmarks/dsv41_b200_b60/`，不会因为 checkout 分支而自动生效。P、D 同名文件不同，应分别构造 overlay。

在已有匹配 vLLM wheel、原生扩展和依赖的 Linux 环境中，用该环境的 Python 执行：

```bash
VLLM_SITE=$($VENV/bin/python -c 'import importlib.util; print(importlib.util.find_spec("vllm").origin.rsplit("/", 1)[0])')
$VENV/bin/python benchmarks/dsv41_b200_b60/materialize_overlay.py \
  --role prefill --base-vllm "$VLLM_SITE" --output "$P_OVERLAY"
$VENV/bin/python benchmarks/dsv41_b200_b60/materialize_overlay.py \
  --role decode --base-vllm "$VLLM_SITE" --output "$D_OVERLAY"
```

`VENV` 是匹配基座的 venv；`P_OVERLAY`、`D_OVERLAY` 为尚不存在的目录。脚本复制归档文件，其余内容链接到已安装包，保留 `__init__.py` 和原生扩展。启动时将相应 overlay 加入 `PYTHONPATH` 最前面。

部署历史原件见 [runtime/](runtime/)：`serve.sh`、`pb.py`、`bench.sh`、路由策略及 B60 knob。它们保留原路径与日志开关，**不是通用的一键安装脚本**。换环境须设置基座/overlay/模型/Mooncake/Dynamo/harness 路径，复制 `nodes.env.example` 填写已分配的节点，并修改 run ID 和所有嵌入的日志路径；不要复用 `pb-b60-1102a`。远程登录和上传工具不在归档中。

关键参数：P 两个 DP4/EP4 实例、D 一个 DP8/EP8 实例，TP1；P seq/MBT=64/16384、D=256/2560；P 的 Engram CPU offload/8 线程，D GPU resident；Mooncake 为 8×80GB + 1192GB；P FULL_AND_PIECEWISE、D FULL_DECODE_ONLY。P/D 启动 MemAvailable 门槛为 1300/1200 GiB，冷启动曾约 36 分钟。

## 必须保留的限制

- 上述调度对照只关闭短任务优先、动态 token 上限和 DP 共享预算，其他优化保留，不能代表整套系统相对原版 vLLM 的收益。两次使用不同机器，约 1.5% 吞吐差与机器性能差接近，无法判断调度是否提高吞吐。
- 尾延迟有代价：包含突发恢复期时，默认调度让 99% 请求在 6.12 秒内收到首 token，当前优化需 11.05 秒；与 SJF 等待满 10 秒才提权有关。去掉恢复期后，两者 p99 为 5.43 / 5.72 秒，但当前优化的 p99.9 仍高约 38%。
- dspark k=5、synthetic AL=3.51、adaptive verification 关闭；这些成绩不证明真实接受率或输出质量。
- B60 未满足原预注册聚合锁步效率门槛，采用仍待定；C2048 尚未测试该配置。
- 保留源文件现状与哈希。本次只归档、检查打包一致性和发布链接，没有运行新性能、模型质量或功能测试，也未对快照自动格式化。
