# Kimi K3 + DSpark：B200 部署说明

[发布分支](https://github.com/Sy0307/vllm/tree/k3/b200-tail-release-20261019) · [飞书同版](https://acnc6zeentra.feishu.cn/docx/LE2mdI7ZUovrnGxBjKucKKDWn7S)

源码与依赖已冻结。117 项 CPU 单测、两机依赖核对及 12 项真实 HTTP/SSE 验收通过；本交付组合尚未完成一小时性能、完整质量和稳定性验收。部署内容位于 `deploy/k3-b200-20261019/`，以下简称部署目录。

## 1. 环境与文件

默认两台主机：prod30（`xb01-gpu-200b-0030`，PP0/API，10.18.1.30）和 prod24（`xb01-gpu-200b-0024`，PP1，10.18.1.24），各 8×B200，使用 UID 1000。启动前 GPU 和服务端口必须空闲。prod09/prod25 使用第 6 节的独立配置，不修改默认主机对。

环境：Ubuntu 24.04.3、Python 3.12.3、驱动 590.48.01；torch 2.13.0+cu130、Triton 3.7.1、Transformers 5.17.0、FlashInfer 0.7.0.post1、CUTLASS DSL 4.8.0、DeepEP 2.0.0+local、Mooncake 0.3.13.post1。系统 nvcc 为 13.1.115。

归档目录：

```text
/gpfs/mszn/data/k3-benchmarks/k3-opt-20261013/releases/k3-b200-closeout-20261019/
```

- `runtime-dependencies.tar.gz`：服务依赖与冻结原生运行时。
- `replay-assets.tar.gz`：仅复现 AgentX 时需要的离线数据。
- `k3-b200-release-20261019-r2.tar.gz`、`package-manifest-r2.json`：本版源码及提交、大小、SHA256 清单。

仅在新的隔离 Linux 环境恢复依赖，并保留归档中的绝对路径；不要覆盖共享环境。模型另行准备：主模型 `/gpfs/mszn/models/moonshotai/Kimi-K3`，草稿 `/gpfs/mszn/models/Inferact/Kimi-K3-DSpark`。跨机器从空环境恢复尚未验收。

`source.lock.json` 校验源码与 17 个原生库；`environment.lock.json` 校验有效依赖及路径顺序；`model.lock.json` 校验模型元数据和权重分片大小（不含权重全量内容哈希）。不要用一次普通 pip install 替代锁定环境。

## 2. 组装与启动

两机使用相同检出目录。依赖恢复后执行：

```bash
git clone --branch k3/b200-tail-release-20261019 git@github.com:Sy0307/vllm.git k3-b200-release
cd k3-b200-release
ln -s /gpfs/mszn/data/k3-benchmarks/k3-pd-four-node-20260923/main-env .venv
.venv/bin/python deploy/k3-b200-20261019/assemble_runtime.py
```

组装只运行一次（共享文件系统无需重复），期望输出 `ASSEMBLY_COMPLETE`。它创建只读 overlay，已有路径或校验失败时退出。

在 prod30、prod24 各执行一次，使用同一个、从未使用过的 RUN 名：

```bash
.venv/bin/python deploy/k3-b200-20261019/launch.py release-r2-20261019 deploy/k3-b200-20261019/serving.json
```

仅在 prod30 等待就绪：

```bash
.venv/bin/python deploy/k3-b200-20261019/health.py release-r2-20261019
```

API 地址为 `http://10.18.1.30:18995/v1`，模型名 `kimi-k3`。启动器校验主机、UID、GPU 空闲、端口及依赖；分布式端口 29795，Mooncake 端口 18796/18797/19796。最终命令、环境、日志及进程身份保存在部署目录的 `state/runs/<RUN>/<HOST>/`。

Mooncake 默认每 worker 的 store segment 为 160 GiB，单机 8 个 worker 共 1,280 GiB，另有每 worker 4 GiB 缓冲。启动器根据 `/proc/meminfo` 的 `MemAvailable` 检查上述分配并预留 64 GiB；不足时在加载权重前退出。该预检不预留内存，其他任务仍可能改变可用容量。

## 3. 默认配置

TP8 / PP2（48/45 层）/ DCP8 / EP8；DSpark K4，真实 block 拒绝采样；服务端上限 192，调度/批量 token 预算 9216/9792；预填充配对 WAIT=16，共享专家 TP4；PP0/PP1 显存利用率 0.95/0.91。历史缓存、CuTe 历史注意力和预填充成本限制默认关闭。

使用 `serving.json` 启动实际服务；`benchmark.json` 使用合成接受率，只供计分。`profiles/` 中的配置为可选实验，不叠加启用。TP4 存在数值差异，完整语义质量评估未完成。

保留 `--use-replayssm` 与 `experimental_recoverssm_store=true`。已核对旧版 RecoverSSM 边界/状态归属、新块清零、卷积逻辑历史、Mooncake 重复 key 读取、思考历史与 token 统计修复均在当前运行源码中；核对明细见 `evidence/historical-fixes-audit.json`。

## 4. 接口验收与接入

服务就绪后，仅 prod30 执行，输出文件须不存在：

```bash
.venv/bin/python deploy/k3-b200-20261019/probe_api.py --output /tmp/k3-release-r2-api.json
```

应通过全部 12 项检查，覆盖流式/非流式：指定 function 返回 400；错误 Partial 位置返回 400；正确 Partial 返回新增正文后缀；auto/required 返回实际工具名、JSON 参数及 `tool_calls` 结束原因。

客户端必须完整保存 assistant 的 `reasoning`、`content`、`tool_calls`；流式分别累积对应增量。历史兼容 `reasoning_content`，保持 `include_reasoning=true`。网关若改字段名，需双向完整映射。

K3 常开思考，不支持指定某个 function。`partial=true` 只用于最后一条 assistant 的文本正文前缀，不能放在 user 消息，也不能带 tool_calls。错误位置返回 400 是本版校验契约；不据此声明与官方所有细节一致。网关自身的修复需在 AGW 侧部署，本仓库不包含网关代码。

## 5. 停止与回退

在 prod30、prod24 各执行：

```bash
.venv/bin/python deploy/k3-b200-20261019/stop.py release-r2-20261019
```

只停止本 RUN 记录且进程身份匹配的会话。检查 `stopped.json` 中 `owned_remaining` 为空；验收失败时保留日志，修复后换新 RUN。旧环境与旧版本保留，尚未进行生产切流；本交付不是新的容器镜像或 Kubernetes 发布。

测试和历史指标见 `evidence/`，当前验收状态见 `release-status.json`。

## 6. prod09/prod25 部署与 JSON 验收（2026-10-21）

使用更新后的发布分支；10 月 19 日的 r2 源码归档不包含本节新增的启动器和配置，原依赖与模型锁保持不变。

prod09（`xb01-gpu-200b-0009`，10.18.1.9）为 PP0/API，prod25（`xb01-gpu-200b-0025`，10.18.1.25）为 PP1。在两台各执行：

```bash
.venv/bin/python deploy/k3-b200-20261019/launch.py json-api-new deploy/k3-b200-20261019/profiles/serving-prod09-prod25.json
```

仅 prod09 执行健康检查与 JSON 验收，输出目录必须不存在：

```bash
.venv/bin/python deploy/k3-b200-20261019/health.py json-api-new
.venv/bin/python deploy/k3-b200-20261019/probe_json.py --output /tmp/k3-json-acceptance
```

该配置通过 `nodes` 显式选择这对主机，其余模型和采样设置继承 `serving.json`，仅将 Mooncake `mc_segment` 改为每 worker 64 GiB（单机共 512 GiB）。第一轮使用 160 GiB 时，prod09 的 worker 在挂载 segment 阶段被主机 OOM killer 杀死，尚未进入 HTTP 验收。较小的 store 会减少外部缓存容量；本次接口验证不证明吞吐或长时间缓存命中率与 160 GiB 相同。

`probe_json.py` 包含普通请求对照，以及 JSON Object、strict JSON Schema、嵌套 Schema 和反向提示的流式/非流式检查，共 9 项。它保存原始请求、HTTP/SSE 响应和逐项结果；只解析最终 `message.content` 或拼接后的 `delta.content`，不将思考字段当作 JSON。任一检查失败时返回非零退出码。

停止本次实例时，在两台各执行 `stop.py json-api-r2`；它仍只处理本 RUN 记录且身份匹配的进程。

实测结果：`json-api-r2` 上两轮 JSON 验收各 9/9 通过，原有 HTTP/SSE 接口验收 12/12 通过，新增内存检查单测本地及远端均 3/3 通过。服务保留运行，API 为 `http://10.18.1.9:18995/v1`，模型名 `kimi-k3`。上述启动示例用新 RUN 名 `json-api-new`，不要重复使用已有 RUN。

本轮未修改冻结的推理源码；JSON 模式本身已可用，修复范围是主机选择、Mooncake 容量和启动前内存检查。内存检查是在 r2 启动后加入并单测的，不声称 r2 已执行该预检。客户端应仅解析最终 `content`；流式先拼接 `delta.content`，思考字段单独处理。结果及原始响应见 `evidence/json-prod09-prod25-20261021/`。本次直接访问后端，不覆盖网关转发、全部 JSON Schema 关键字或性能与质量评估。

## 7. allowed_tools 更新（2026-10-26）

新增 `tool_choice.allowed_tools`，支持 `auto` 和 `required`，工具调用限定在请求指定的子集中。语法关闭、parser 不支持（包括 `hf`）、启用 beam search，或 `auto` 模式叠加额外输出约束时返回 400，避免接受请求后静默忽略限制。同时删除重复归一化，合并精简回归测试。

代码已合入原[发布分支](https://github.com/Sy0307/vllm/tree/k3/b200-tail-release-20261019)；本次更新尚未部署。
