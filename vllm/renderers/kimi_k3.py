# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media.connector import merge_media_io_kwargs
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

# Keep the original image mode (including the alpha channel) for K3 instead of
# flattening images onto a background color. Server-level (--media-io-kwargs)
# and request-level media_io_kwargs still take precedence over this default.
_K3_MEDIA_IO_DEFAULTS: dict[str, dict[str, Any]] = {"image": {"image_mode": None}}
_K3_THINKING_EFFORTS = ("low", "high", "max")


def _merge_k3_media_io_kwargs(
    media_io_kwargs: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    return merge_media_io_kwargs(_K3_MEDIA_IO_DEFAULTS, media_io_kwargs)


def _dump_k3_template_value(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    return value


def _apply_k3_thinking_kwargs(kwargs: dict[str, Any]) -> None:
    if (enable_thinking := kwargs.pop("enable_thinking", None)) is not None:
        kwargs.setdefault("thinking", enable_thinking)

    reasoning_effort = kwargs.pop("reasoning_effort", None)
    if reasoning_effort == "none":
        kwargs.setdefault("thinking", False)
    elif reasoning_effort is not None:
        kwargs.setdefault("thinking_effort", reasoning_effort)

    thinking_effort = kwargs.get("thinking_effort")
    if thinking_effort is not None and thinking_effort not in _K3_THINKING_EFFORTS:
        supported = ", ".join(_K3_THINKING_EFFORTS)
        raise VLLMValidationError(
            f"Kimi K3 supports thinking_effort values: {supported}",
            parameter="thinking_effort",
            value=thinking_effort,
        )


def _normalize_k3_tool_messages(
    conversation: list[ConversationMessage],
) -> list[dict[str, Any]]:
    """Reorder tool-result messages to match assistant tool_call order.

    Supports matching by ``tool_call_id`` or by the synthetic
    ``"{tool}:{zero_based_index}"`` alias. When any tool message in a
    block cannot be resolved, the whole block is left in its original
    order (graceful fallback).

    Returns a new list; caller-owned message dicts are not mutated.
    """
    normalized: list[dict[str, Any]] = []
    i = 0

    while i < len(conversation):
        message = conversation[i]
        normalized.append(dict(message))
        i += 1

        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue

        # Build the lookup table from the assistant's tool_calls.
        targets_by_id: dict[str, tuple[int, str]] = {}
        for position, tool_call in enumerate(tool_calls):
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue

            call_target = (position, name)
            aliases = [f"{name}:{position}"]
            if tool_call_id := tool_call.get("id"):
                aliases.insert(0, str(tool_call_id))
            for alias in aliases:
                targets_by_id.setdefault(alias, call_target)

        # Collect consecutive tool messages.
        block_start = i
        while i < len(conversation) and conversation[i].get("role") == "tool":
            i += 1
        if i == block_start:
            continue

        # Try to resolve every tool message in the block.
        resolved: list[tuple[int, int, dict[str, Any]]] = []
        for order, tool_message in enumerate(conversation[block_start:i]):
            tool_call_id = tool_message.get("tool_call_id")
            resolved_target = (
                targets_by_id.get(str(tool_call_id))
                if tool_call_id is not None
                else None
            )
            if resolved_target is None:
                normalized.extend(dict(item) for item in conversation[block_start:i])
                break

            position, name = resolved_target
            enriched = dict(tool_message)
            enriched["tool"] = name
            enriched["index"] = position + 1
            resolved.append((position, order, enriched))
        else:
            resolved.sort(key=lambda item: (item[0], item[1]))
            normalized.extend(item for _, _, item in resolved)

    return normalized


def _validate_k3_partial(messages: list[ChatCompletionMessageParam]) -> bool:
    for index, message in enumerate(messages):
        if not message.get("partial", False):
            continue
        if index != len(messages) - 1 or message["role"] != "assistant":
            raise VLLMValidationError(
                "partial=true is only supported on the final assistant message",
                parameter=f"messages[{index}].partial",
            )
        if not isinstance(message.get("content"), str) or message.get("tool_calls"):
            raise VLLMValidationError(
                "Kimi K3 Partial Mode requires a text content prefix "
                "without tool_calls",
                parameter=f"messages[{index}]",
            )
        return True
    return False


class KimiK3Renderer(BaseRenderer[HfTokenizer]):
    """Render chat prompts with Kimi K3's Python XTML encoding.

    K3 ships no Jinja chat template; its tokenizer renders messages through
    ``encoding_k3`` instead. We tokenize eagerly so the structural markers keep
    their special-token ids while user- and tool-supplied text stays ordinary.
    """

    def __init__(self, config: VllmConfig, tokenizer: HfTokenizer | None) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        params: ChatParams,
    ) -> list[int]:
        # Tokenize eagerly: K3 encodes structural markers as special tokens and
        # user/tool text as ordinary tokens, so we cannot defer to a plain
        # re-tokenization of the rendered string downstream.
        kwargs = params.get_apply_chat_template_kwargs()
        _apply_k3_thinking_kwargs(kwargs)
        if params.tool_choice not in (None, "auto"):
            kwargs["tool_choice"] = _dump_k3_template_value(params.tool_choice)
        if params.response_format is not None:
            kwargs["response_format"] = _dump_k3_template_value(params.response_format)
        choice = kwargs.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function":
            raise VLLMValidationError(
                "tool_choice 'specified' is incompatible with thinking enabled",
                parameter="tool_choice",
            )
        kwargs["tokenize"] = True
        tokenizer = self.get_tokenizer()
        if not conversation or not conversation[-1].get("partial", False):
            return tokenizer.apply_chat_template(conversation, **kwargs)

        # Render request-level instructions before the unfinished assistant turn.
        # The frozen K3 tokenizer otherwise appends them after that turn and
        # ignores continue_final_message. Preserve its segment-wise tokenization.
        kwargs["add_generation_prompt"] = False
        prefix = tokenizer.apply_chat_template(conversation[:-1], **kwargs)
        final = tokenizer.apply_chat_template(
            [conversation[-1]],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
            thinking=kwargs.get("thinking", True),
            thinking_effort=None,
        )
        closing = tokenizer.encode(
            "<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>",
            add_special_tokens=False,
        )
        if not closing or final[-len(closing) :] != closing:
            raise VLLMValidationError(
                "Kimi K3 tokenizer does not expose the expected Partial Mode boundary",
                parameter="messages[-1].partial",
            )
        return prefix + final[: -len(closing)]

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        partial = _validate_k3_partial(messages)
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_conversation = _normalize_k3_tool_messages(conversation)
        if partial:
            rendered_conversation[-1]["partial"] = True
        prompt = parse_dec_only_prompt(
            self._apply_chat_template(rendered_conversation, params)
        )
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return cast(list[ConversationMessage], rendered_conversation), prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        partial = _validate_k3_partial(messages)
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=_merge_k3_media_io_kwargs(params.media_io_kwargs),
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered_conversation = _normalize_k3_tool_messages(conversation)
        if partial:
            rendered_conversation[-1]["partial"] = True
        token_ids = await self._apply_chat_template_async(rendered_conversation, params)
        prompt = parse_dec_only_prompt(token_ids)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return cast(list[ConversationMessage], rendered_conversation), prompt
