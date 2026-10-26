# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionAllowedToolChoiceParam,
    ChatCompletionAllowedToolFunction,
    ChatCompletionAllowedToolRef,
    ChatCompletionAllowedToolsParam,
    ChatCompletionRequest,
)
from vllm.exceptions import VLLMValidationError

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
            },
        },
    },
]


def _allowed_choice(mode: str, names: list[str]) -> dict:
    return {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": mode,
            "tools": [
                {"type": "function", "function": {"name": name}} for name in names
            ],
        },
    }


def _request(tool_choice: dict) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": TOOLS,
            "tool_choice": tool_choice,
        }
    )


@pytest.mark.parametrize("mode", ["auto", "required"])
def test_allowed_tools_collapses_to_mode_and_filters_tools(mode: str):
    request = _request(_allowed_choice(mode, ["get_weather"]))

    assert request.tool_choice == mode
    assert [tool.function.name for tool in request.tools] == ["get_weather"]
    # The renderer only understands plain mode strings; it must not receive
    # the allowed_tools object.
    params = request.build_chat_params(
        default_template=None, default_template_content_format="auto"
    )
    assert params.tool_choice == mode


def test_allowed_tools_keeps_declaration_order():
    request = _request(_allowed_choice("auto", ["get_time", "get_weather"]))

    assert [tool.function.name for tool in request.tools] == [
        "get_weather",
        "get_time",
    ]


def test_allowed_tools_ignores_duplicate_references():
    request = _request(_allowed_choice("required", ["get_weather", "get_weather"]))

    assert [tool.function.name for tool in request.tools] == ["get_weather"]


def test_allowed_tools_accepts_flat_function_references():
    request = _request(
        {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "auto",
                "tools": [{"type": "function", "name": "get_time"}],
            },
        }
    )

    assert request.tool_choice == "auto"
    assert [tool.function.name for tool in request.tools] == ["get_time"]


def test_allowed_tools_accepts_typed_choice_model():
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS,
        tool_choice=ChatCompletionAllowedToolChoiceParam(
            allowed_tools=ChatCompletionAllowedToolsParam(
                mode="required",
                tools=[
                    ChatCompletionAllowedToolRef(
                        function=ChatCompletionAllowedToolFunction(name="get_weather")
                    )
                ],
            )
        ),
    )

    assert request.tool_choice == "required"
    assert [tool.function.name for tool in request.tools] == ["get_weather"]


def test_allowed_tools_sets_internal_grammar_flag():
    request = _request(_allowed_choice("auto", ["get_weather"]))

    assert request.enforce_tool_call_grammar is True
    assert "enforce_tool_call_grammar" not in request.model_dump()


def test_plain_auto_does_not_set_internal_grammar_flag():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": TOOLS,
            "tool_choice": "auto",
        }
    )

    assert request.enforce_tool_call_grammar is False


def test_allowed_tools_auto_keeps_structured_outputs_compatible():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": TOOLS,
            "tool_choice": _allowed_choice("auto", ["get_weather"]),
            "structured_outputs": {"json": {"type": "object"}},
        }
    )

    assert request.tool_choice == "auto"
    assert request.structured_outputs is not None


@pytest.mark.parametrize(
    ("tool_choice", "match"),
    [
        ({"type": "allowed_tools"}, "allowed_tools.*must be an object"),
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {"mode": "always", "tools": []},
            },
            "mode.*must be",
        ),
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {"mode": "auto", "tools": []},
            },
            "non-empty array",
        ),
        (
            _allowed_choice("auto", ["not_declared"]),
            "do not match any",
        ),
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "auto",
                    "tools": [{"type": "custom", "name": "shell"}],
                },
            },
            "only supports function tools",
        ),
        (
            {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "auto",
                    "tools": [{"type": "function", "function": {}}],
                },
            },
            "must name a function",
        ),
    ],
)
def test_allowed_tools_rejects_invalid_choices(tool_choice: dict, match: str):
    with pytest.raises(VLLMValidationError, match=match):
        _request(tool_choice)


def test_allowed_tools_still_requires_tools():
    with pytest.raises(VLLMValidationError, match="`tools` must be set"):
        ChatCompletionRequest.model_validate(
            {
                "model": "kimi-k3",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": _allowed_choice("auto", ["get_weather"]),
            }
        )
