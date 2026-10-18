# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.renderers import ChatParams
from vllm.renderers.kimi_k3 import KimiK3Renderer, _merge_k3_media_io_kwargs
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry


class StubTokenizer:
    """Stands in for the model's TikTokenTokenizer.

    Records the kwargs it is called with and returns fixed token ids, so tests
    can assert how the renderer drives ``apply_chat_template`` without the real
    (git-LFS) tiktoken vocabulary.
    """

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.calls: list[dict[str, Any]] = []
        self.conversations: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, conversation, **kwargs) -> list[int]:
        self.conversations.append(conversation)
        self.calls.append(kwargs)
        return list(self.token_ids)


@dataclass
class MockHFConfig:
    model_type: str = "kimi_k3"


@dataclass
class MockModelConfig:
    runner_type: str = "generate"
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    multimodal_config: Any = None
    hf_config: MockHFConfig = field(default_factory=MockHFConfig)
    allowed_local_media_path: str = ""
    allowed_media_domains: Any = None
    enable_prompt_embeds: bool = False
    renderer_num_workers: int = 1


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


def _make_renderer(tokenizer: StubTokenizer) -> KimiK3Renderer:
    config = MockVllmConfig(MockModelConfig(), MockParallelConfig())
    return KimiK3Renderer(config, tokenizer)


def test_kimi_k3_registered():
    assert RENDERER_REGISTRY.load_renderer_cls("kimi_k3").__name__ == "KimiK3Renderer"
    assert (
        TokenizerRegistry.load_tokenizer_cls("kimi_k3").__name__ == "CachedHfTokenizer"
    )


def test_k3_media_io_defaults_preserve_original_mode():
    # Default: K3 keeps the original image mode (no background flattening).
    assert _merge_k3_media_io_kwargs(None) == {"image": {"image_mode": None}}

    # Server-/request-level values take precedence over the K3 default.
    assert _merge_k3_media_io_kwargs({"image": {"image_mode": "RGB"}}) == {
        "image": {"image_mode": "RGB"}
    }

    # Unrelated image kwargs are merged with the default.
    assert _merge_k3_media_io_kwargs(
        {"image": {"rgba_background_color": (0, 0, 0)}}
    ) == {"image": {"image_mode": None, "rgba_background_color": (0, 0, 0)}}


def test_apply_chat_template_forces_tokenize_and_pins_return_dict():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    tools = [{"type": "function", "function": {"name": "search"}}]
    params = ChatParams(
        chat_template_kwargs={"tools": tools, "tokenize": False, "thinking": True}
    )

    token_ids = renderer._apply_chat_template(
        [{"role": "user", "content": "hi"}], params
    )

    assert token_ids == [7, 8, 9]
    kwargs = tokenizer.calls[-1]
    # tokenize is forced on even though the request asked for False, so K3 keeps
    # the special-vs-ordinary token distinction instead of re-tokenizing a string.
    assert kwargs["tokenize"] is True
    # return_dict is pinned False so we always get a flat list of ids.
    assert kwargs["return_dict"] is False
    assert kwargs["tools"] == tools
    assert kwargs["thinking"] is True


def test_apply_chat_template_translates_standard_thinking_kwargs():
    # Standard enable_thinking/reasoning_effort kwargs must be translated
    # to K3's native thinking/thinking_effort.
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "none"}
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["thinking"] is False
    assert "thinking_effort" not in kwargs
    assert "enable_thinking" not in kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("reasoning_effort", ["low", "high", "max"])
def test_apply_chat_template_translates_supported_reasoning_effort(
    reasoning_effort: str,
):
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(chat_template_kwargs={"reasoning_effort": reasoning_effort})

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["thinking_effort"] == reasoning_effort
    assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("reasoning_effort", ["minimal", "medium", "xhigh"])
def test_apply_chat_template_rejects_unsupported_reasoning_effort(
    reasoning_effort: str,
):
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(chat_template_kwargs={"reasoning_effort": reasoning_effort})

    with pytest.raises(VLLMValidationError, match="thinking_effort") as exc_info:
        renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert exc_info.value.parameter == "thinking_effort"
    assert exc_info.value.value == reasoning_effort
    assert tokenizer.calls == []


def test_apply_chat_template_validates_canonical_native_thinking_effort():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={
            "thinking_effort": "low",
            "reasoning_effort": "medium",
        }
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert tokenizer.calls[-1]["thinking_effort"] == "low"


@pytest.mark.parametrize("thinking_effort", ["none", "minimal", "medium", "xhigh"])
def test_apply_chat_template_rejects_unsupported_native_thinking_effort(
    thinking_effort: str,
):
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(chat_template_kwargs={"thinking_effort": thinking_effort})

    with pytest.raises(VLLMValidationError, match="thinking_effort") as exc_info:
        renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert exc_info.value.parameter == "thinking_effort"
    assert exc_info.value.value == thinking_effort
    assert tokenizer.calls == []


def test_apply_chat_template_native_k3_kwargs_take_precedence():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={
            "thinking": True,
            "enable_thinking": False,
            "thinking_effort": "low",
            "reasoning_effort": "high",
        }
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["thinking"] is True
    assert kwargs["thinking_effort"] == "low"


def test_apply_chat_template_adds_k3_api_metadata():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    response_format = {"type": "json_object"}
    params = ChatParams(
        tool_choice="required",
        response_format=response_format,
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["tool_choice"] == "required"
    assert kwargs["response_format"] == response_format


def test_apply_chat_template_auto_tool_choice_keeps_template_kwarg():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={"tool_choice": "required"},
        tool_choice="auto",
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert tokenizer.calls[-1]["tool_choice"] == "required"


def test_apply_chat_template_omits_tool_choice_without_tools():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)

    renderer._apply_chat_template(
        [{"role": "user", "content": "hi"}], ChatParams(tool_choice=None)
    )

    assert "tool_choice" not in tokenizer.calls[-1]


def test_render_messages_returns_token_prompt():
    renderer = _make_renderer(StubTokenizer([1, 2, 3]))

    conversation, prompt = renderer.render_messages(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt == {"prompt_token_ids": [1, 2, 3]}
    assert "multi_modal_data" not in prompt
    assert conversation[0]["role"] == "user"


@pytest.mark.parametrize("stream", [False, True])
def test_k3_rejects_named_function_before_tokenization(stream):
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
        stream=stream,
    )
    tokenizer = StubTokenizer([1])
    renderer = _make_renderer(tokenizer)
    with pytest.raises(VLLMValidationError, match="specified.*thinking enabled"):
        renderer.render_messages(
            request.messages, ChatParams(tool_choice=request.tool_choice)
        )
    assert not tokenizer.calls


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "hi", "partial": True}],
        [
            {"role": "assistant", "content": "prefix", "partial": True},
            {"role": "user", "content": "hi"},
        ],
    ],
)
@pytest.mark.asyncio
async def test_k3_rejects_misplaced_partial_after_request_validation(messages):
    request = ChatCompletionRequest(model="kimi-k3", messages=messages)
    tokenizer = StubTokenizer([1])
    renderer = _make_renderer(tokenizer)
    for render in (renderer.render_messages, renderer.render_messages_async):
        with pytest.raises(VLLMValidationError, match="final assistant"):
            if render == renderer.render_messages_async:
                await render(request.messages, ChatParams())
            else:
                render(request.messages, ChatParams())
    assert not tokenizer.calls


def test_partial_false_does_not_change_normal_chat_rendering():
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "hi", "partial": False}],
    )
    tokenizer = StubTokenizer([1])
    renderer = _make_renderer(tokenizer)
    _, prompt = renderer.render_messages(request.messages, ChatParams())
    assert prompt == {"prompt_token_ids": [1]}
    assert len(tokenizer.calls) == 1


@pytest.mark.parametrize(
    "prefix", ["结论：", "  prefix\n\t", "", "<|open|>think<|sep|>"]
)
def test_partial_uses_real_k3_response_channel_and_preserves_prefix(prefix):
    """Optional CPU integration against the pinned deployment tokenizer."""
    model_path = os.environ.get("K3_TOKENIZER_PATH")
    if not model_path:
        pytest.skip("Set K3_TOKENIZER_PATH to the pinned K3 tokenizer")
    from transformers import AutoTokenizer

    from vllm.parser.parser_manager import ParserManager

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    renderer = _make_renderer(tokenizer)
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[
            {"role": "user", "content": "Continue the answer."},
            {
                "role": "assistant",
                "content": prefix,
                "partial": True,
                "reasoning_content": "Existing reasoning.",
            },
        ],
    )
    _, prompt = renderer.render_messages(
        request.messages, ChatParams(tool_choice="none")
    )
    ids = prompt["prompt_token_ids"]
    rendered = tokenizer.decode(ids)
    assert rendered.endswith("<|open|>response<|sep|>" + prefix)
    assert "Existing reasoning." in rendered
    assert "tool_choice=none" in rendered
    assert rendered.count('message role="assistant"') == 1
    prefix_ids = tokenizer.encode(prefix, allow_special_tokens=False)
    if prefix_ids:
        assert ids[-len(prefix_ids) :] == prefix_ids
    parser_cls = ParserManager.get_parser(reasoning_parser_name="kimi_k3")
    parser = parser_cls(tokenizer)
    assert parser.is_reasoning_end(ids)
    parser.set_prompt_token_ids(ids)
    reasoning, content, _ = parser.parse("continued", request)
    assert reasoning is None and content == "continued"


def test_render_messages_derives_private_xtml_tool_attrs():
    tokenizer = StubTokenizer([1, 2, 3])
    renderer = _make_renderer(tokenizer)

    conversation, _ = renderer.render_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "lookup:0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                    {
                        "id": "lookup:1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "lookup:1",
                "tool": "client-supplied-name",
                "index": 99,
                "content": "second",
            },
            {
                "role": "tool",
                "tool_call_id": "lookup:0",
                "content": "first",
            },
        ],
        ChatParams(),
    )

    assert [message["content"] for message in conversation[1:]] == [
        "first",
        "second",
    ]
    assert conversation[1]["tool"] == "lookup"
    assert conversation[1]["index"] == 1
    assert conversation[2]["tool"] == "lookup"
    assert conversation[2]["index"] == 2
    assert tokenizer.conversations[-1] == conversation


def test_render_messages_ignores_client_supplied_xtml_tool_attrs():
    tokenizer = StubTokenizer([1, 2, 3])
    renderer = _make_renderer(tokenizer)

    conversation, _ = renderer.render_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "lookup:0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "unknown",
                "tool": "lookup",
                "index": 1,
                "content": "result",
            },
        ],
        ChatParams(),
    )

    assert "tool" not in conversation[1]
    assert "index" not in conversation[1]


@pytest.mark.asyncio
async def test_render_messages_async_returns_token_prompt():
    renderer = _make_renderer(StubTokenizer([4, 5]))

    conversation, prompt = await renderer.render_messages_async(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt == {"prompt_token_ids": [4, 5]}
    assert conversation[0]["role"] == "user"
