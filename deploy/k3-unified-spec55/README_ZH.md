# Kimi K3 Unified + DSpark 投机解码：运维部署手册

版本：`k3-unified-spec55-20260927`（2026-09-27；2026-09-28 修订：去掉 `--language-model-only`）。面向运维与运维 Agent。

## 0. 部署前必读

- **本版保持 ViT 开启，不要添加 `--language-model-only`**（2026-09-28 修改）。9-27 版的 `runtime.json` 原本带有这个参数。实测它不影响 ViT 加载：worker 日志里视觉编码器照常构建，并因注意力头数不能被 TP8 整除，自动改为数据并行。但它会把每个请求的图片数上限设为 0，API 日志为 `running in text-only mode`，按代码带图片的请求会被拒绝。现已从 `runtime.json` 删除（见第 3 节第 5 条）。
- 这是一个两节点实例：TP8 × PP2 × direct DCP8 + EP + 模型级 SP，开启 DSpark K4 投机解码和 MooncakeStore。每个请求的 prefill 和 decode 都由两台一起完成，不是 P/D 两座岛。
- 交付物分三部分：
  1. **冻结的 vLLM overlay**：GPFS 上的只读目录，带逐文件校验和；
  2. **代码分支**：用于审阅和重建；
  3. **本目录的启动器**：`runtime.json`、`launch.py`、`verify.py`、`gate_mooncake.py`。
- **运行时实际加载的是冻结 overlay**，不是分支源码。分支里只有 Python 源码改动，编译产物（`.so`、`third_party/`、`vllm_flash_attn/cute`）只在冻结 overlay 里。不要用分支重新 `pip install` 替代 overlay。
- 与 9 月 14 日的 1214.90 手册不同：本版不用 k0s 容器，而是在两台机器上直接用 GPFS 上的 Python 环境启动进程；依赖版本见第 2 节。
- 不要 `pip install -U`，不要改动第 2 节列出的任何共享环境目录，不要自行杀别人的进程。发现 GPU 或端口被占用就停下来报告。

## 1. 目标配置

| 项目 | 配置 |
| --- | --- |
| 机器 | rank0 = prod25（xb01-gpu-200b-0025，10.18.1.25）：API + PP 第0级（层 1–48）<br>rank1 = prod24（xb01-gpu-200b-0024，10.18.1.24）：headless，PP 第1级（层 49–93） + 采样与草稿 |
| API | `http://10.18.1.25:18984`，served-model-name `kimi-k3` |
| 并行 | TP8、PP2（`VLLM_PP_LAYER_PARTITION=48,45`）、direct DCP8 / a2a、EP（deepep_v2）、模型级 SP |
| MoE | deep_gemm_mega_moe |
| Model Runner | V2 |
| 投机解码 | DSpark K4（草稿模型 `/gpfs/mszn/models/Inferact/Kimi-K3-DSpark`，草稿 TP8，贪心起草，**block 验证**），RecoverSSM（`--use-replayssm`） |
| 缓存 | GPU APC + MooncakeStore（每 worker 200 GB DRAM + 4 GB buffer，RDMA，前 4 路 IB），`experimental_recoverssm_store=true`（同一引擎自存自取 KDA 页） |
| 调度 | max-num-seqs 192，max-num-batched-tokens 8768，max-num-scheduled-tokens 8192，long-prefill-token-threshold 0 |
| 显存 / KV | gpu-memory-utilization 0.88，FP8 KV，max-model-len 1048576，prefix-match-unit 128 |
| 图模式 | FULL_AND_PIECEWISE，捕获档位见 `runtime.json`（decode 每 4 请求一档 × 5，混批 1280–8192） |

## 2. 冻结产物与依赖

**发布目录**（GPFS，只读）：

```text
/gpfs/mszn/data/k3-benchmarks/k3-ops-unified-spec55-20260927/
  overlay/vllm/            完整 vLLM 包（767 MB，5159 个文件），PYTHONPATH 指向 overlay/
  overlay-SHA256SUMS       逐文件 sha256
  deploy/                  本目录的副本
  run/<deployment>/        启动器生成的日志、launch.json、编译缓存
```

`overlay-SHA256SUMS` 的 sha256：`38906276662dbc03c61e4ca1bd1747825ae97a033d598736b7510d91499546bf`

**代码分支** `k3/unified-spec55-opt-20260927`（<https://github.com/Sy0307/vllm/tree/k3/unified-spec55-opt-20260927>），基线为 vLLM `c9611215195e`：

| commit | 内容 |
| --- | --- |
| `06f7cb59d` | Codex unified 候选 overlay 的 23 个源文件，原样导入：模型级 SP 输入投影、MRV2 PP/DCP 修复、RecoverSSM 状态、KDA RecoverSSM transport |
| `cb02153a1` | TokenSpeed MLA decode：按请求变长切分、验证步头组折叠、折叠拷贝，以及小 q 长历史 prefill 的 split-KV |
| `41b08e9b0` | RecoverSSM 与 MooncakeStore 同开；KDA commit kernel 换 tile；RecoverSSM 混批放开 piecewise 图 |
| `16460502b` | 去掉 PP 接收草稿时的主机同步；几个默认关闭的 PP 往返实验开关 |
| `6eb6879f2` | 融合 all-gather + GEMM 的形状清单（按真实权重逐形状实测） |
| `8407050f2` | cohort 平衡（仅 no-spec 用，本版关闭） |

冻结 overlay 中这 29 个源文件的 sha256，与分支 `8407050f2` 逐一核对一致。`8407050f2` 之后的提交只改 `deploy/` 下的启动器和手册，`vllm/` 源码与冻结 overlay 相同。

另有两个分支，**都不在冻结 overlay 里，运维部署不要用**：

| 分支 | 内容 | 状态 |
| --- | --- | --- |
| `k3/unified-spec55-frozen-mm57368-20260928`（<https://github.com/Sy0307/vllm/tree/k3/unified-spec55-frozen-mm57368-20260928>） | 在 `8407050f2` 上叠加 PR 57368（图片 CPU 输入用完即释放 + 共享存储），针对多图 CPU OOM | CPU 单测通过；尚未做 GPU 验证，也没打进 overlay |
| `k3/unified-spec55-cpudisp-20260928`（<https://github.com/Sy0307/vllm/tree/k3/unified-spec55-cpudisp-20260928>） | 混批 KDA metadata 去重 + state 回写改 `index_copy_`，减少 CPU 下发时间，与图片无关 | 门禁逐位一致；端到端 +0.31%，在噪声内 |

**共享依赖**（GPFS 上现有目录，不要修改）：

| 用途 | 路径 | 版本 |
| --- | --- | --- |
| Python 环境 | `/gpfs/mszn/data/k3-benchmarks/k3-pd-four-node-20260923/main-env` | torch 2.13.0+cu130，flashinfer 0.6.18.post1，nvidia-cutlass-dsl 4.7.1，mooncake-transfer-engine-cuda13 0.3.13.post1 |
| NCCL | `VLLM_NCCL_SO_PATH` → `/gpfs/mszn/data/k3-benchmarks/nightly-9521-env/.../nvidia/nccl/lib/libnccl.so.2` | 以运行时 `NCCLLibrary().ncclGetVersion()` 为准 |
| Mooncake master | `/gpfs/mszn/data/k3-benchmarks/nightly-9521-env/bin/mooncake_master` | — |
| 其余动态库 | `LD_LIBRARY_PATH` 中的 nightly-9521-env 与 k3-mm-main-e2e-20260917/baseline-env | 见 `runtime.json` |
| FlashInfer cubin | `/gpfs/mszn/data/k3-benchmarks/k3-pd-four-node-20260923/unified-spec-scaling/flashinfer-cubins` | 离线，`FLASHINFER_NO_DOWNLOAD=1` |
| 模型 | `/gpfs/mszn/models/moonshotai/Kimi-K3`、`/gpfs/mszn/models/Inferact/Kimi-K3-DSpark` | — |

## 3. 本版相对基线打开的优化

以下开关都由环境变量控制，取值写在 `runtime.json`，启动器会自动设置；`verify.py` 会读运行中进程的环境逐项核对。

| 环境变量 | 作用 | 验证 |
| --- | --- | --- |
| `VLLM_K3_CONTEXT_SPLIT_KV=1`，`VLLM_K3_SPLIT_MEMO=1` | 小 q 长历史 prefill 的历史注意力切分 KV，并缓存切分索引 | 精度与未切分同级，graph 回放一致 |
| `VLLM_K3_TS_VAR_SPLIT=1`，`VLLM_K3_TS_VAR_SPLIT_Q=1`，`VLLM_K3_TS_VS_ALPHA=2.0`，`VLLM_K3_TS_VS_MEMO=1` | MLA decode 按请求 KV 长度分配切分，消除长请求拖尾 | 数值与原版同 |
| `VLLM_K3_TS_HEAD_FOLD=1`，`VLLM_K3_TS_HEAD_FOLD_MAP=5:4,4:3`，`VLLM_K3_HF_FASTCOPY=1` | 验证步 96 头分组折叠（KV 读 5 次降为 4 次）；折叠拷贝改用 Triton | 36 ≤ B ≤ 64 保护；与生产路径 LSE ≤ 4e-6；拷贝逐位相同 |
| `VLLM_K3_PP_DRAFT_NOSYNC=1` | 去掉 PP 接收草稿时的同步 H2D 拷贝 | CPU 等价测试 1831 例全同；门禁 PASS |
| `VLLM_K3_RECOVERSSM_WIDE_PW=1` | RecoverSSM 下超过 960 token 的混批也走 piecewise 图 | 与 eager 冷算逐 token 相同 |
| `VLLM_K3_KDA_TILE=commit=16x4` | KDA commit kernel tile | 逐位相同 |
| `VLLM_K3_KDA_SP_SHAPES`，`VLLM_K3_SMLP_SP_SHAPES` | 融合 all-gather + GEMM 的形状清单 | 真实权重逐形状逐位相同 |
| `--additional-config` 中 `shared_mlp_sp_input_projection` 与 `kda_sp_extended_shapes` 为 true | 打开共享 MLP 和扩展形状的融合投影 | 同上 |

**必须保持关闭**（`runtime.json` 已设为关闭，不要改）：

- `VLLM_K3_TS_SKIP_CORR=0.0`：大于 0 时结果错误。
- `VLLM_K3_SMALLQ_*=0`：spec 下变慢。
- `VLLM_K3_PP_EARLY_COMMIT=0`、`VLLM_K3_LAST_LATE_COMMIT=0`、`VLLM_K3_PP_SEND_DBUF=0`、`VLLM_K3_COMMIT_SIDESTREAM=0`：测过，无收益。
- `VLLM_K3_HF_VALIDATE=0`、`VLLM_K3_HF_VALIDATE_CTRL=0`：仅用于校验，开启后 decode 每步慢约 18 ms。
- `VLLM_K3_COHORT_BALANCE=0`。

**与测试时的运行配置相比，本版做了五处删改**：

1. 去掉测量探针 `--worker-extension-cls`。
2. 去掉 `--profiler-config` 和 `K3_SCALING_OUTPUT`。
3. `PYTHONPATH` 只保留冻结 overlay（测试时多出的两个目录只放探针模块）。
4. 编译缓存改到发布目录下，首次启动需要重新编译。
5. 去掉 `--language-model-only`（2026-09-28），让图片输入可用。

其余参数和环境变量与 1h 测试完全一致。注意：第 4 节的性能数字和门禁都是带着 `--language-model-only`、只有文本输入时测的。

## 4. 性能与正确性依据

- **受控接受率 55%**（草稿接受率 = 被接受 / 草稿数；逐位累计 0.80 / 0.60 / 0.45 / 0.35，AL 3.2）下，InferenceX AgentX C96 1h（aiperf，seed 42，3600 s）实测 **1900.96 tok/s**（aiperf 自报 1895.56）。
    - 对照：无投机解码的最好 1h 成绩 1449.84，同口径历史冠军 1214.90。
- **测法校准**：用 synthetic 采样喂入真实运行实测的逐位接受率，1h 分数比真实运行高 1.07%（按步定价对比为 1.19%）。因此真实接受率 55% 时，预期约 1880 tok/s（推算）。
- 附注：InferenceX 数据集的 prompt 是随机哈希拼接的，草稿模型几乎猜不中，这份数据上实测 AL 只有 1.71，1h 为 1302.73。这是测试集本身的特点，不代表生产接受率。
- **正确性**：各项优化的核对方式和结果见第 3 节。block 模式门禁的做法是：同一 prompt 分别冷算、本地命中、Mooncake 命中，比较 64 个贪心 token 和 logprob。优化期间每个组合都跑过这个门禁并 PASS。本版的新部署需要按第 7 节重新执行一次。
- **图片输入尚未验证**：去掉 `--language-model-only` 后，预期文本路径不变：ViT 本来就会加载；K3 的 config 没有 mm_prefix，注意力后端的选择也不变。但有两点没测过：启动时会多做视觉编码器的显存 profile，KV 容量可能略有变化；图片与 DSpark、PP2、DCP8、RecoverSSM、Mooncake 同时开的组合没在本栈跑过。接图片流量前，需要补一次带图片的门禁，并重新确认启动日志里的 KV 容量。

## 5. 预检（两台都要执行）

```bash
R=/gpfs/mszn/data/k3-benchmarks/k3-ops-unified-spec55-20260927/deploy
cp $R/site.example.json /tmp/k3-site.json     # 核对主机名、IP、overlay_manifest_sha256
python3 $R/launch.py --site /tmp/k3-site.json  # 只做检查，打印将要执行的命令与环境
```

检查内容：

- 8 张 B200 空闲，无未纠正的 ECC 错误；
- 端口空闲：18984、29734，rank0 另加 18784 / 19784 / 18785；
- MemAvailable ≥ 1750 GiB；
- Python、模型、Mooncake master 存在；
- overlay 清单校验和一致，且本版改动的 29 个文件与清单一致。

任何一项失败都停止并报告，不要修改检查条件。

## 6. 启动（每台一条命令）

先在 prod25（rank0）执行，随后**立即**在 prod24（rank1）执行。不要等 rank0 健康检查通过再启动 rank1，两级必须一起起来。

```bash
# prod25
python3 /gpfs/mszn/data/k3-benchmarks/k3-ops-unified-spec55-20260927/deploy/launch.py --site /tmp/k3-site.json --launch
# prod24
python3 /gpfs/mszn/data/k3-benchmarks/k3-ops-unified-spec55-20260927/deploy/launch.py --site /tmp/k3-site.json --launch
```

rank0 会先启动 Mooncake master，然后启动 vLLM。日志在 `run/<deployment>/<hostname>/server.log`，进程号在 `server-process.json`，实际参数在 `launch.json`。

首次启动要编译 CUDA graph 和 Triton 缓存，耗时明显长于后续启动；超时时间是 `VLLM_ENGINE_READY_TIMEOUT_S=3600`。

## 7. 验收（接流量之前）

两台都执行 `verify.py`；`gate_mooncake.py` 只在 rank0 上执行。

```bash
curl -fsS --max-time 10 http://10.18.1.25:18984/health          # rank0
python3 $R/verify.py --site /tmp/k3-site.json --full               # 两台都跑，输出 VERIFY <host> PASS
python3 $R/gate_mooncake.py /tmp/k3-gate.json 11 http://10.18.1.25:18984   # rank0，输出 GATE PASS
```

**`verify.py` 核对**：

- 运行中进程的实际命令行和环境变量（读 `/proc/<pid>`），逐项与 `launch.json` 对照，包括全部 `VLLM_K3_*` 变量；
- 关键参数：TP8 / PP2 / DCP8 a2a、EP + deepep_v2、MegaMoE、DSpark K4、`rejection_sample_method=block`、MooncakeStore + RecoverSSM store、192 / 8768 / 0.88 / FP8 KV；
- 日志：启动阶段没有 Traceback，没有 EngineDeadError，rank0 上模型级 SP 已启用；
- overlay 清单校验和，加 `--full` 时全量重算哈希；
- rank0 额外发一个 16 token 的贪心请求，并确认投机解码计数器在增长。

不使用 `/server_info`：这套运行环境里它收集系统信息会失败，返回 500，与部署本身无关。

**`gate_mooncake.py`**：会清空 GPU 前缀缓存，并请求 top-5 logprobs，**只能在未接流量的新部署上运行**。它用 30K / 90K / 150K 三个长度，比较冷算、本地命中和 Mooncake 命中。

交付时如果还没跑门禁，状态必须写"基础验收通过，Mooncake 门禁待执行"。

**已做的验证**（2026-09-27，prod24 / prod25，按本手册步骤执行）：

- 19:03 启动，19:15 `/health` 通过，首次启动约 12 分钟；
- 两台 `verify.py --full` 全部 PASS：命令行、环境变量、27 个 `VLLM_K3_*` 开关、5159 个 overlay 文件哈希全部一致；
- `gate_mooncake.py` 输出 GATE PASS。冷算的 64 个 token 和 logprob 在三种长度上都与测试期各臂的门禁逐位相同，Mooncake 命中与冷算完全一致。
- 按第 8 节停机，32 秒内 GPU 全部释放；
- 用本发布包加 `--synthetic-acceptance 0.80,0.60,0.45,0.35` 复现受控 55% 的 1h 性能（19:34 就绪，两台 verify PASS）：官方分 **1903.48 tok/s**。与测试时的 1900.96 相差 +0.13%；300 / 900 / 1800 / 3600 s 各时点完成量差 +0.4 / +0.7 / +0.2 / +0.5%；AL 3.202；ITL 37.66 ms；全程两台零报错。

## 8. 停止

启动器使用独立进程组，进程组号就是 `server-process.json` 里的 pid：

- 在各自节点上执行 `kill -TERM -<pid>`，停掉 vLLM；
- rank0 再对 `mooncake-master.json` 里的 pid 执行同样操作。

只停本部署记录的进程，不要按进程名批量杀。

## 9. 已知限制

- 依赖第 2 节的共享 GPFS 环境。如果长期运行，建议把 main-env、nightly-9521-env 和冻结 overlay 一起封装成不可变镜像，再重新验收。
- 只在 prod24 / prod25 上验证过：IB 网卡 `ib7s400p0`–`ib7s400p7`、GID 0、上述 IP。换机器需要重新核对映射。
- `VLLM_SERVER_DEV_MODE=1` 与测试时一致，`/server_info`、`/reset_prefix_cache` 等开发端点处于开放状态，需要在网络层限制访问。
- 同一个服务进程只能做一次 Kineto profile，第二次会段错误。
- `--synthetic-acceptance` 仅用于压测：它会用随机接受代替真实验证，输出不是模型的真实结果，**生产严禁使用**。
- 本版不含 PR 57368（图片 CPU 输入用完即释放 + 共享存储）。多图、长输出的高并发负载下，PP 第0级的 8 个 worker 各保留一份图片张量直到请求结束，可能 CPU OOM；9-16/17 distill 事故就是这个根因。修复已叠在冻结代码上，见第 2 节的 `k3/unified-spec55-frozen-mm57368-20260928`；要用它需要重建 overlay，并做带图片的 GPU 验证。
- MooncakeStore 每节点固定占用 8 ×（200 + 4）= 1632 GiB 主机内存，启动即全额记账，只适用于本手册的裸机环境。放进 1700 / 1725 GiB 上限的容器里放不下，需要重新定段大小。另外，第 5 节的 MemAvailable 检查读的是 `/proc/meminfo`，在容器里拿到的是宿主机的内存，不能代替容器上限检查。
