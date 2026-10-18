# Kimi K3 + DSpark：B200 部署说明

[发布分支](https://github.com/Sy0307/vllm/tree/k3/b200-tail-release-20261019) · [飞书同版](https://acnc6zeentra.feishu.cn/docx/LE2mdI7ZUovrnGxBjKucKKDWn7S)

源码与依赖已冻结。117 项 CPU 单测、两机依赖核对及 12 项真实 HTTP/SSE 验收通过；本交付组合尚未完成一小时性能、完整质量和稳定性验收。部署内容位于 `deploy/k3-b200-20261019/`，以下简称部署目录。

## 1. 环境与文件

限定两台主机：prod30（`xb01-gpu-200b-0030`，PP0/API，10.18.1.30）和 prod24（`xb01-gpu-200b-0024`，PP1，10.18.1.24），各 8×B200，使用 UID 1000。启动前 GPU 和服务端口必须空闲。

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
