# 2026-10-09 K3 思考历史接入与 token 统计回补

适用分支：`k3/unified-spec55-frozen-mm57368-20260928`。本次源码更新保留官方模型模板，回补上游 K3 reasoning token 统计，并说明客户端和网关的历史消息要求。当前是源码和 CPU 验证状态，尚未构建镜像或部署生产。

## 1. 多轮消息要求

[HF 官方使用说明](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/README.md)要求保留完整 assistant 历史，包括思考内容和工具调用。接入 K3 时应保存 API 返回的完整消息，再回填到下一轮 `messages`。

本分支的 Chat API 使用以下字段：

| 场景 | 字段及要求 |
| --- | --- |
| 非流式输出 | `choices[i].message.reasoning`、`content`、`tool_calls` |
| 流式输出 | 累积 `choices[i].delta.reasoning`、`content` 和工具调用增量 |
| 历史请求 | 支持 `reasoning`；请求模型也会把 `reasoning_content` 归一化为 `reasoning` |
| 网关兼容 | 若对外使用 `reasoning_content`，在请求和响应两侧完整映射；不要同时填入两个不一致的字段 |
| 返回控制 | 多轮保存历史需要 `include_reasoning=true`（本分支默认值）；它只控制返回已生成的思考 |

非流式客户端可直接保存 SDK 消息：

```python
response = client.chat.completions.create(
    model="kimi-k3",
    messages=messages,
    stream=False,
    extra_body={
        "chat_template_kwargs": {"thinking": True},
        "include_reasoning": True,
    },
)
assistant = response.choices[0].message.model_dump(
    mode="json", exclude_none=True
)
messages.append(assistant)
```

需要执行工具调用时，先保存这条 assistant，再按 `tool_call_id` 添加工具结果。不要将返回消息重新构造成只有 `role` 和 `content` 的字典。

流式客户端必须在结束后组装完整 assistant：分别累积 reasoning 和 content；按工具调用的 `index` 合并 `id`、函数名和参数片段，并保存最终 `tool_calls`。不要只保存页面展示的正文，也不要把所有 reasoning 转移到 content。

已有的纯正文历史、手工示例或其他模型历史，应保留其来源。无法恢复的思考不要编造。官方模板在思考模式下会为缺失 reasoning 的 assistant 写入空 think 块；保留完整历史是正常接入路径。对此类历史是否采用其他序列化策略，需要独立验证质量、工具续接和性能，不能默认删除所有历史 think 块。

## 2. vLLM 统计回补

源码基线：`0ed095c8fb47444e3babd9f6397f19430f374ef2`。回补来源为上游 [PR #58372](https://github.com/vllm-project/vllm/pull/58372)，提交 `9bf44c40134c024956df9134036b1d5b3d4eef7b`。

本分支代码提交：[`7ef487078ccf4abc634773012caa2b53505ab1d1`](https://github.com/Sy0307/vllm/commit/7ef487078ccf4abc634773012caa2b53505ab1d1)。parser 文件 SHA256 为 `c19ebe65d3a00a7abeef66b45971184d307563aca78b80512a1d4362ebe5b153`，与本次核对的 upstream 文件一致。

修改文件：`vllm/reasoning/kimi_k3_reasoning_parser.py`，补充 K3 三 token 标记下的 `count_reasoning_tokens`；同步回补原上游单元测试。

统计对象仅为本轮生成 token，排除 think 结构标记和 response 内容；生成前缀已包含 think 开头时，从输出第一个 token 起计数。空 think 的计数仍应为 0。此回补修正 `usage.completion_tokens_details.reasoning_tokens`，不会改变 prompt、采样或输出解析，也不会保证每次生成非空思考。

## 3. 已完成验证与发布前验收

已完成 CPU 检查：

- K3 reasoning parser 既有测试及上游新增测试：51 项通过。
- K3 renderer 测试：23 项通过，包含异步路径。
- SDK 消息保存 → 请求校验 → 实际 renderer → 官方首发/最新版 serializer 的 segment 检查：16 个组合通过，覆盖两种字段名、工具历史、同步和异步。tokenizer 在此检查中返回占位 ID，没有运行真实 BPE 或模型。
- 8 条归档输出的真实 token ID：统计与第一个 think 关闭位置一致，空思考为 0，有思考为正数。

发布前应完成以下接口与模型验收，记录模型 revision、代码提交、镜像 digest、请求参数及网关版本：

1. 直连与网关分别验证非流式和流式：reasoning 字段内容一致、工具增量完整、请求 ID 可追踪。
2. 完整保存真实 K3 回复并续接多轮和工具结果；核对下一轮请求及实际模板中的历史 reasoning 未丢失。
3. 验证 usage：非空思考的 token 统计为正，立即关闭 think 的统计为 0；`include_reasoning=false` 仅隐藏返回内容，不改变生成 token 的统计。
4. 对纯正文或导入历史单独评估。若业务要求每次必须有非空思考，要记录原始生成 token；首个输出直接关闭 think 时，解析器没有可补出的内容，须继续验证模型与输入策略。
5. 验证长思考流式请求的延迟与 API CPU 使用。当前上游实现会对累计生成序列计数，单元测试没有覆盖其性能。

## 4. 镜像与部署

本次修改不在 2026-10-01 冻结的 10 文件发布包中。旧镜像、旧 `SNAPSHOT-SHA256SUMS` 和旧 ConfigMap 不能表示已包含统计回补。

如需发布，按照现有不可变镜像流程加入新的 parser 文件，更新完整 overlay 清单、`SNAPSHOT-SHA256SUMS`、`source.lock.json` 和 `runtime.json` 的 `release_integrity`，记录新源码提交并生成匹配的不可变 ConfigMap。镜像内校验通过后，再做隔离 serving 验收和逐组滚动。

`/model` 上的官方 remote code 来自只读模型 PVC，本次无需修改。客户端历史保存和网关字段透传可独立于 vLLM 镜像发布。每一侧更新均应记录实际版本，完成端到端验收后再宣告接入完成。
