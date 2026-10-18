# Kimi K3 + DSpark B200 冻结版本与部署说明｜2026-10-19

适用对象：维护 K3 双节点 B200 服务、接入 OpenAI Chat Completions 和复现实验的工程师。维护人：Sy03。本文说明冻结源码、依赖、配置和接口契约；当前为待 GPU 验收的源码交付版。按本次安排，先完成打包和文档，再执行新增实验；已有一小时实验异步运行，不作为本版打包的等待条件。

飞书同版文档：<https://acnc6zeentra.feishu.cn/docx/LE2mdI7ZUovrnGxBjKucKKDWn7S>。

## 版本与交付范围

| 项目 | 内容 |
| --- | --- |
| 源码分支 | <https://github.com/Sy0307/vllm/tree/k3/b200-tail-release-20261019> |
| 源码冻结提交 | 53710fce3d146b821edc4e059ffbf580287a813c |
| 交付目录 | deploy/k3-b200-20261019/ |
| 交付形态 | Git 分支、完整源码归档、环境锁、原生二进制依赖归档、模型元数据锁、启动与验收脚本 |
| 运行范围 | prod30（PP0/API）与 prod24（PP1），各 8 张物理 B200 GPU 0–7；不涉及 H200 |
| 状态 | 117 项接口 CPU 单测通过；当前依赖解析检查通过；组装后的 GPU HTTP/SSE、真实拒绝采样与完整性能验收待执行 |
| 生产状态 | 本次没有构建新容器镜像，没有修改 Kubernetes/LWS 或生产路由，也没有完成生产切流 |

正式分支保存所有已有收益的实现；配置按收益和代价分别保留。封装后的组合尚未重测，因此下文成绩是已有配置的参考成绩，不是新交付包的实测成绩。

## 分支如何组成

| 层次 | 来源与作用 |
| --- | --- |
| 原 K3 分支 | 父提交 984c1efeaeaca19942ade2f57b171213e4101d88，保留原 K3、DSpark、RecoverSSM 及多模态接入历史 |
| 上游合并 | 父提交 a2fb4827324bf023075799ef3d70d347834b8137；隔离合并快照 f779262c188f7ba406fab8528df8ee86a22f03e1 |
| 实测运行时回填 | 以 sy-final2-20261017/mgG 冻结运行时逐文件回填 Python 与资源；tokenspeed_mla 原样放入 vendor 并保留原许可证 |
| 可选实现 | 加入严格单请求预填充、CuTe 历史注意力、预填充成本上限 v2；分别由配置开关控制 |
| API 修复 | 4 个运行时文件：消息 schema、K3 renderer、统一 parser、tool parser；保留正常思考和工具解析路径 |
| 锁与二进制 | source.lock.json 锁定 3563 个运行时文本/资源文件及 17 个原生库；原生 wheel 元数据提交为 94f4170df37fe29d68bd2f7e5a496501b78d55ae |

原先仍在合并中的开发工作区没有被改写。本分支使用独立 worktree。源码归档不包含 GPU 权重和依赖二进制；完整运行环境由本节源码、依赖归档、模型目录以及指定系统环境共同组成。不能把本分支直接当作已验证的重新编译 wheel 或容器镜像。

## 已打包的优化及默认状态

| 优化组 | 包含的实现 | 状态与已知代价 |
| --- | --- | --- |
| 缓存与 KDA 状态 | checkpoint 原地写入、REFRESH/HOLD、Mooncake 最新轮 key 修复、本地前缀跳过、两段式 lookup | 默认配置保留；#1–#3 的历史吞吐收益最大，各次结果不能相加 |
| MLA 与 DSpark | split-KV 工作量重排、fold、FP8 历史 KV 传输、合并融合、k_pe FP8 直通、草稿本地 argmax | 默认保留；不引入被否决的 dense-prefill FP8 GEMM 改动 |
| 投影与主机开销 | 按 M 选 SP 通信路径、KDA 跨组 metadata、workspace view 缓存 | 默认保留；部分小优化没有独立端到端收益量化 |
| MoE 与共享专家 | 共享专家副流重叠、MegaMoE、共享专家 TP4 | 默认参考配置采用 TP4；TP4 不保证逐位一致，完整语义质量评估尚未通过 |
| 显存与 KV 容量 | 本卡行数缓冲、TP/DCP/EP 共享通信器、PP1 graph/接收缓冲复用、KDA 分组、两级显存比例、PP1 视觉塔处理 | 默认保留；PP0/PP1 利用率分别 0.95/0.91，容量随所选配置变化 |
| 调度 | 服务端 max-num-seqs=192；预填充配对 WAIT=16；默认预算 9216/9792 | 默认参考组合；客户端为 96 个 AgentX lane，不等于服务端最多 96 个在途请求 |
| 严格单请求预填充 | 扩大预算至 12288/12864，保留严格的单请求上限 | 可选 ttft-strict；TTFT P99 明显改善，但 TPOT P99 退步 |
| CuTe 历史注意力 | 仅 B200 支持形状走 CuTe，其余回退原后端；包含 indptr 对齐修复 | 可选 cute-history，默认关闭；有性能收益，也有输出数值差异，不能当作质量通过 |
| 预填充成本上限 v2 | 按历史长度和成本模型限制每请求块，并保留最小块及有限尾块余量 | 可选 prefill-cost-v2，默认目标为 0；现有实验异步，结果未用于本次打包选择 |
| 历史缓存与均匀切块 | 历史 K/V 复用实现留在源码；未把均匀切块提升为默认方案 | 历史缓存默认关闭；其约 12 GiB 代价与小收益不适合默认启用。均匀切块没有稳定吞吐收益，且 ITL 有代价 |

此前单卡内核、短测与一小时的收益来自不同实验条件。配置文件是互斥的完整选择，不应把多个 profile 的收益相加，也不应直接叠加所有开关。GB300/B300 的算法或远端 KV 长尾结果不构成本机 B200 的性能承诺；本包没有另起分片远端 KV 的结构改造。

## 现有一小时成绩与长尾取舍

以下均为 C96/S192、每 lane 请求预热 10 次、正式发送 3600 秒、seed=42、393 条数据、DSpark K4 合成接受率 [0.8,0.6,0.45,0.35] 的既有结果。吞吐按成功输出 token / 原生计分窗口计算，包含正式发送后的 drain。各配置为单轮长测，闭环请求到达时间并不完全相同。

| 配置 | 吞吐 tok/s | 成功 / 错误 | 正式记录错误率 |
| --- | --- | --- | --- |
| 默认参考 9216/9792 | 2228.93 | 8657 / 81 | 0.93% |
| 严格单请求上限 12288/12864 | 2222.84 | 8659 / 77 | 0.88% |
| CuTe 历史注意力 9216/9792 | 2257.95 | 8782 / 83 | 0.94% |

| TTFT（秒） | 平均 | P50 | P90 | P95 | P99 |
| --- | --- | --- | --- | --- | --- |
| 默认参考 | 2.523 | 1.540 | 3.872 | 6.664 | 26.528 |
| 严格上限 | 2.078 | 1.520 | 3.460 | 5.174 | 12.008 |
| CuTe 历史注意力 | 2.325 | 1.504 | 3.505 | 6.019 | 21.031 |

| TPOT（毫秒） | 平均 | P50 | P90 | P95 | P99 |
| --- | --- | --- | --- | --- | --- |
| 默认参考 | 38.041 | 35.997 | 52.969 | 63.060 | 94.985 |
| 严格上限 | 39.493 | 36.081 | 55.280 | 66.668 | 125.176 |
| CuTe 历史注意力 | 37.108 | 35.367 | 50.544 | 59.851 | 94.434 |

TPOT 对应 full_response_inter_token_latency：每个成功请求的完整生成时长除以输出 token 数减一，再对这些请求平均值取分位数；它不是逐 token 到达间隔的 P99。错误率分母为正式阶段成功加错误记录，不包含预热错误、协议取消等其他类别。

默认参考的 81 条正式错误均为 AIPerf 未收到实际正文；严格上限的 77 条中包含 75 条无实际正文和 2 条连接重置。这些错误仍是已知限制，不能写成零错误，也不能用本次两项 API 修复宣称它们已全部解决。预热设置为每 lane 10 个请求，不代表 960 个请求全部成功；默认参考实际记录了 218 条预热成功和 843 条预热错误，严格上限为 211/850。

取舍：严格上限使 TTFT P99 降低约 54.7%，同时 TPOT P99 上升约 31.8%；因此保留为可选 profile，不直接替换默认。CuTe 具有较好的现有综合指标，但其串行门禁 23 例只有 13 例 token 完全一致，仍缺完整语义评估。本次不等待成本上限实验，不宣布新的综合冠军。

## 依赖环境与归档

| 组件 | 实际锁定值 |
| --- | --- |
| 系统 / Python | Ubuntu 24.04.3；Python 3.12.3；环境解释器为 main-env/bin/python |
| GPU / 驱动 | 每台 8×B200；驱动 590.48.01 |
| CUDA | PyTorch 编译 CUDA 为 13.0；系统 nvcc 为 13.1.115，二者分别记录 |
| PyTorch / Triton | torch 2.13.0+cu130；triton 3.7.1 |
| Transformers | 5.17.0 |
| FlashInfer / CUTLASS DSL | FlashInfer 0.7.0.post1；CUTLASS DSL 4.8.0，由额外 Python 路径覆盖环境内较旧版本 |
| 通信与 KV | DeepEP 2.0.0+local；Mooncake 0.3.13.post1 CUDA 13；显式加载锁定 NCCL 库 |
| vLLM 原生库 | wheel 元数据 0.29.1rc1.dev569+g94f4170df；17 个 .so 的精确 SHA256 见 source.lock.json |

实际环境并非单一虚拟环境：PYTHONPATH 先提供 overlay、up/deps、deps/cdsl48 和交付目录；main-env 再通过 .pth 引用 baseline-env 与 nightly-9521-env。锁文件保留解析顺序、模块 origin、.pth 原文和 221 个有效分发包的版本。无序 pip freeze 无法替代这些内容。DeepGEMM/FlashKDA 已随源码或运行时提供，不能因缺少同名顶层 pip 包就判断缺依赖。

```text
/gpfs/mszn/data/k3-benchmarks/
  k3-pd-four-node-20260923/main-env/
  k3-mm-main-e2e-20260917/baseline-env/
  nightly-9521-env/
  k3-opt-20261013/up/deps/
  k3-opt-20261013/deps/cdsl48/
  k3-opt-20261013/campaigns/sy-final2-20261017/mgG/
  ref440542/{harness-venv,aiperf-source,inferencex-source,python}/
```

归档目录为 /gpfs/mszn/data/k3-benchmarks/k3-opt-20261013/releases/k3-b200-closeout-20261019/。runtime-dependencies.tar.gz 保存上述服务依赖、原生运行时和 benchmark harness；replay-assets.tar.gz 保存离线数据缓存与 tokenizer 路径兼容层。模型权重、系统驱动、操作系统及 /usr/local/cuda 系统工具链由目标环境单独提供。mmap-cache 为派生缓存，可从归档的离线数据重新生成。

服务依赖与离线回放数据归档均已完成。runtime-dependencies.tar.gz 为 11,386,171,046 字节，SHA256 为 5df73b99f1f597d2ec94b3116c8447cf9b315cf796cc85efd076835cec01482f；replay-assets.tar.gz 为 938,209,675 字节，SHA256 为 5b8e55019fb64d746790fd2836b6f1ccd365aba3897dc8a01aa1295cda8f04ce。对应清单 environment-bundle.json、replay-bundle.json 已随分支保存。恢复目标应是新的隔离 Linux 环境并保持锁定绝对前缀；不要覆盖现有共享环境。跨机器从空环境恢复尚未完成验收。

主模型目录 /gpfs/mszn/models/moonshotai/Kimi-K3；草稿目录 /gpfs/mszn/models/Inferact/Kimi-K3-DSpark。model.lock.json 记录配置、tokenizer Python 与 tiktoken.model 哈希，以及 96 个主模型分片、1 个草稿分片的文件名和大小。没有可核实的 Hub revision，未对所有权重内容做全量哈希；启动前的文件大小检查不等于权重内容完整性证明。

## 目录、配置和启动器契约

| 文件 | 作用 |
| --- | --- |
| source.lock.json / environment.lock.json / model.lock.json | 源码与原生库、实际依赖解析环境、模型元数据三份锁 |
| prod_runtime.json + serving.json | 基础参数与真实 block 拒绝采样配置；启动时合成最终参数 |
| benchmark.json + benchmark_protocol.json | 合成接受率性能口径与默认 benchmark 配置 |
| profiles/ | 严格上限、CuTe 历史注意力、成本上限 v2 的独立 serving/benchmark 配置 |
| assemble_runtime.py / verify_release.py | 新建只读 overlay，并校验锁定源码、原生库、模型元数据和分片大小 |
| launch.py / health.py / stop.py | 两节点启动、prod30 就绪检查、按记录的进程身份停止本次实例 |
| probe_api.py / run_agentx.py | 真实接口验收与固定协议的 AgentX 性能入口 |
| provenance.json / release-status.json / evidence/ | 来源、验收状态、既有指标及 CPU 检查证据 |

以下为当前启动器的输入/输出契约与待执行示例，新增脚本尚未完成 GPU 代表性试跑。先完成本文，再执行隔离验收；HEALTHY 仅代表 /health 就绪，不能代替功能与质量验收。

```bash
git clone --branch k3/b200-tail-release-20261019 git@github.com:Sy0307/vllm.git k3-b200-release
cd k3-b200-release
# 在已还原且通过锁定环境核对的主机上，将 .venv 指向 main-env。
ln -s /gpfs/mszn/data/k3-benchmarks/k3-pd-four-node-20260923/main-env .venv
.venv/bin/python deploy/k3-b200-20261019/assemble_runtime.py
```

assemble_runtime.py 仅在 overlay 不存在时运行；复制锁定原生运行时，再用分支源码替换对应文本文件并逐项校验。期望末尾输出 ASSEMBLY_COMPLETE。它不会安装 pip 包或启动 GPU 服务。已有路径、哈希不符或缺少模型文件时应停止，保留错误证据。

```bash
# prod30 与 prod24 各自先检查身份、GPU 与监听端口
hostname
id
nvidia-smi -L
nvidia-smi --query-compute-apps=pid --format=csv,noheader
ss -ltn
# 在两台相同检出目录分别执行；同一 run 名称只能使用一次
.venv/bin/python deploy/k3-b200-20261019/launch.py release-20261019 deploy/k3-b200-20261019/serving.json
# 仅 prod30
.venv/bin/python deploy/k3-b200-20261019/health.py release-20261019
```

启动器只接受 prod30/prod24 和 UID 1000；启动前拒绝任何已有 GPU 计算进程及服务端口占用，逐文件验证 overlay，再用真实 PYTHONPATH 检查依赖解析。两机映射为 10.18.1.30/10.18.1.24，API 18995、分布式主端口 29795；Mooncake 使用 18796/18797/19796。每次 run 创建独立缓存前缀与记录目录，不复用旧的 Mooncake 池。

最终参数、环境、物理 GPU 和进程 PID/PGID/start_ticks 写入 deploy/k3-b200-20261019/state/runs/<RUN>/<HOST>/。profiler 与 worker 诊断扩展在 benchmark 配置保留；serving 配置移除它们，并关闭开发接口。serving.json 与 benchmark.json 不可互换使用：前者真实 block 验证，后者合成接受率。

## Tool Call 与 Partial Mode 接口契约

本节适用于模型名 kimi-k3 的 /v1/chat/completions。K3 常开思考；官方规定开启思考时指定某个 function 应返回 400。本版在请求校验处拒绝该选择，而不是简单把 stop 改成 tool_calls。auto、required、none 的原有解析路径保留。官方文档来源和本地 CPU 证据见文末。

| 输入 | 本版契约 | 验证状态 |
| --- | --- | --- |
| tool_choice={type:function,function:{name:...}} | HTTP 400，message 包含 tool_choice 'specified' is incompatible with thinking enabled；流式/非流式相同 | CPU 单测通过；GPU HTTP/SSE 待执行 |
| tool_choice=auto / required | 按真实 XTML 生成解析工具调用；实际调用应包含函数名和有效 JSON 参数，结束原因为 tool_calls | CPU parser 用例通过；真实工具生成待执行 |
| 最后一条 assistant，文本 content，partial=true | 把正文保留为未闭合前缀，续写结果只返回新增后缀；可保留 reasoning 历史 | 实际 K3 tokenizer 与 parser 单测通过；GPU 续写待执行 |
| user 消息 partial=true 或非最后一条 assistant partial=true | 本版明确返回 HTTP 400，提示 final assistant | 位置校验单测通过；官方错误位置的线上返回尚未核验 |
| Partial 携带 tool_calls 或非文本 content | 本版拒绝，限定为正文文本前缀续写 | 不声明支持工具调用的 Partial 扩展 |

```json
{
  "model": "kimi-k3",
  "messages": [
    {
      "role": "user",
      "content": "1 + 1 等于几？只说答案。"
    },
    {
      "role": "assistant",
      "reasoning_content": "1 加 1 等于 2。",
      "content": "答案是",
      "partial": true
    }
  ],
  "max_tokens": 128
}
```

Partial 修复没有改写模型目录里的 tokenizer。renderer 保留其分段 token 化和思考历史，只移除最后 assistant 的结束边界，使生成从 response 正文继续；parser 根据 prompt 通道状态把后缀识别为正文，避免把正常续写误归入 reasoning。验证必须检查实际后缀和通道，不能仅凭返回 200 或输出被截断就认定 Partial 生效。

普通多轮请求应完整回传 assistant 的 reasoning、content、tool_calls；历史兼容 reasoning_content。流式客户端分别累积 delta.reasoning、delta.content 与工具增量，再保存完整 assistant。两项接口修复不代表所有官方 API 细节已完全对齐，也不证明既有空正文错误均已消除。

## 文档完成后的验收与实验入口

目前已完成：117 项 CPU 单测，包含实际 K3 tokenizer 的中文、空白、空前缀、控制标记和通道状态；新建 overlay 的 3563 个运行时文件和 17 个原生库通过校验；两台主机各自通过 221 个有效包的解析、.pth 与 torch 导入核对；新增 API/部署文件检查通过。文档完成后已异步启动 release-api-20261019 做真实 HTTP/SSE 验收，结果尚未补录；本次自有服务在验收完成或 30 分钟到期后自动停止。冻结的原始运行时与第三方代码仍有继承的 lint 问题，已记录并保留原字节，不宣称分支已满足上游 PR 的全量 lint 要求。

下一步先做真实 block 采样的隔离 HTTP/SSE 验收，再决定完整计分配置。probe_api.py 包含 12 项请求：指定 function 拒绝、两类错误 Partial 位置拒绝、有效 Partial 后缀续写、auto/required 工具调用，各覆盖流式与非流式。输出写入新文件，遇到任何断言失败即停止并保留已有响应。

```bash
# prod30，真实 serving 服务就绪后；输出路径不能已存在
.venv/bin/python deploy/k3-b200-20261019/probe_api.py --output /tmp/k3-release-api-20261019.json
```

真实接口验收必须检查 HTTP 400 的响应体、SSE [DONE]、有效 Partial 不重复前缀且不新增 reasoning、工具名与 JSON 参数。不得只检查 finish_reason。TP4 和 CuTe 的语义质量仍需专门评估，不能用这 12 项验收或合成性能替代。

```bash
# 新性能实验仅在文档完成后安排；先用 benchmark.json 启动独立服务
.venv/bin/python deploy/k3-b200-20261019/run_agentx.py release-score-20261019 h1c96s192 --duration 3600 --concurrency 96 --warmup 10
```

此入口保留原生 AIPerf 的预热、正式发送、drain 与导出口径。客户端 96 表示会话树 lane；子 agent 会派生请求，服务端运行数可以超过 96。只将 max-num-seqs 改成 192 不能保证负载打满；应同步记录 running/waiting、KV 用量、TTFT/TPOT 分布、成功/错误、请求长度和有效接受长度。结果选择以两项 P95/P99、错误率和吞吐共同判断。

## 停止、保留证据与恢复边界

```bash
# 在 prod30、prod24 分别执行同一 RUN
.venv/bin/python deploy/k3-b200-20261019/stop.py release-20261019
```

stop.py 只对本 run 记录的 UID、PID start_ticks、SID/PGID 匹配会话发送 TERM；等待后仍存活才处理同一自有进程组。PID 被复用时拒绝发送信号。stopped.json 记录 owned_remaining 和当前 GPU 计算进程；期望本 run owned_remaining 为空。其他用户的 GPU 进程不得被停止。

本次不进行生产切流，旧服务环境、旧 overlay 和原开发 worktree 均保留。新版本验收失败时停止本次自有会话并保留日志，修正后使用新的 run 名重新验收。若未来变更线上路由，须另行记录当前已验收版本、镜像 digest、路由与恢复验证；参考文档中的旧镜像标签不能直接视为当前生产状态。

## 来源与证据索引

原部署文档：<https://acnc6zeentra.feishu.cn/wiki/UMFKwfKCZiWz77kF9kQcgtG7nlE。本文沿用其“版本来源—冻结输入—部署入口—验收—恢复”的组织方式，更新为本次> B200 裸机交付，不复用旧镜像或旧 LWS 的生产结论。

官方 Tool Choice：<https://platform.kimi.com/docs/guide/use-tool-choice；官方> Partial Mode：<https://platform.kimi.com/docs/guide/use-partial-mode-feature-of-kimi-api；K3> 接入：<https://platform.kimi.com/docs/guide/kimi-k3-quickstart。>

仓库证据位于 deploy/k3-b200-20261019/evidence/，包括已完成测试、依赖检查、已有成绩和官方文档快照。原始长测审计保存在 k3_deployment/reports/2026-10-17/k3-final2-followup/ 下的 budget-headroom-score-comparison.json 与 cute-context-score-comparison.json。本轮异步实验名 cx-final2costv2-candidate-h1-r1；本版没有等待或引用它的新成绩。
