from __future__ import annotations

import base64
import json
import logging
import time
from typing import TYPE_CHECKING, Optional

import openai
from attrs import Factory, define, field
from schema import Schema

from griptape.artifacts import ActionArtifact, AudioArtifact, TextArtifact
from griptape.common import (
    ActionCallDeltaMessageContent,
    ActionCallMessageContent,
    ActionResultMessageContent,
    AudioDeltaMessageContent,
    AudioMessageContent,
    BaseDeltaMessageContent,
    BaseMessageContent,
    DeltaMessage,
    ImageMessageContent,
    Message,
    PromptStack,
    TextDeltaMessageContent,
    TextMessageContent,
    ToolAction,
    observable,
)
from griptape.configs.defaults_config import Defaults
from griptape.drivers.prompt import BasePromptDriver
from griptape.tokenizers import BaseTokenizer, OpenAiTokenizer
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai.types.chat.chat_completion_chunk import ChoiceDelta
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from griptape.drivers.prompt.base_prompt_driver import StructuredOutputStrategy
    from griptape.tools import BaseTool


logger = logging.getLogger(Defaults.logging_config.logger_name)


@define
class OpenAiChatPromptDriver(BasePromptDriver):
    """OpenAI Chat Prompt Driver.

    Attributes:
        base_url: An optional OpenAi API URL.
        api_key: An optional OpenAi API key. If not provided, the `OPENAI_API_KEY` environment variable will be used.
        organization: An optional OpenAI organization. If not provided, the `OPENAI_ORG_ID` environment variable will be used.
        client: An `openai.OpenAI` client.
        model: An OpenAI model name.
        tokenizer: An `OpenAiTokenizer`.
        user: A user id. Can be used to track requests by user.
        response_format: An optional OpenAi Chat Completion response format. Currently only supports `json_object` which will enable OpenAi's JSON mode.
        seed: An optional OpenAi Chat Completion seed.
        ignored_exception_types: An optional tuple of exception types to ignore. Defaults to OpenAI's known exception types.
        parallel_tool_calls: A flag to enable parallel tool calls. Defaults to `True`.
    """

    base_url: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    api_key: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": False})
    organization: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    model: str = field(kw_only=True, metadata={"serializable": True})
    tokenizer: BaseTokenizer = field(
        default=Factory(lambda self: OpenAiTokenizer(model=self.model), takes_self=True),
        kw_only=True,
    )
    user: str = field(default="", kw_only=True, metadata={"serializable": True})
    response_format: Optional[dict] = field(
        default=None,
        kw_only=True,
        metadata={"serializable": True},
    )
    seed: Optional[int] = field(default=None, kw_only=True, metadata={"serializable": True})
    tool_choice: str = field(default="auto", kw_only=True, metadata={"serializable": False})
    use_native_tools: bool = field(default=True, kw_only=True, metadata={"serializable": True})
    structured_output_strategy: StructuredOutputStrategy = field(
        default="native", kw_only=True, metadata={"serializable": True}
    )
    parallel_tool_calls: bool = field(default=True, kw_only=True, metadata={"serializable": True})
    ignored_exception_types: tuple[type[Exception], ...] = field(
        default=Factory(
            lambda: (
                openai.BadRequestError,
                openai.AuthenticationError,
                openai.PermissionDeniedError,
                openai.NotFoundError,
                openai.ConflictError,
                openai.UnprocessableEntityError,
            ),
        ),
        kw_only=True,
    )
    modalities: list[str] = field(default=Factory(lambda: ["text"]), kw_only=True, metadata={"serializable": True})
    audio: dict = field(
        default=Factory(lambda: {"voice": "alloy", "format": "pcm16"}), kw_only=True, metadata={"serializable": True}
    )
    _client: openai.OpenAI = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @lazy_property()
    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            organization=self.organization,
        )

    @observable
    def try_run(self, prompt_stack: PromptStack) -> Message:
        params = self._base_params(prompt_stack)
        logger.debug(params)
        result = self.client.chat.completions.create(**params)

        logger.debug(result.model_dump())
        if len(result.choices) == 1:
            message = result.choices[0].message

            return Message(
                content=self.__to_prompt_stack_message_content(message),
                role=Message.ASSISTANT_ROLE,
                usage=Message.Usage(
                    input_tokens=result.usage.prompt_tokens,
                    output_tokens=result.usage.completion_tokens,
                ),
            )
        else:
            raise Exception("Completion with more than one choice is not supported yet.")

    @observable
    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]:
        params = self._base_params(prompt_stack)
        logger.debug({"stream": True, **params})
        result = self.client.chat.completions.create(**params, stream=True)

        for chunk in result:
            logger.debug(chunk.model_dump())
            if chunk.usage is not None:
                yield DeltaMessage(
                    usage=DeltaMessage.Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    ),
                )
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta

                content = self.__to_prompt_stack_delta_message_content(delta)

                if content is not None:
                    yield DeltaMessage(content=content)

    def _base_params(self, prompt_stack: PromptStack) -> dict:
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "user": self.user,
            "seed": self.seed,
            "modalities": self.modalities,
            "audio": self.audio,
            **({"stop": self.tokenizer.stop_sequences} if self.tokenizer.stop_sequences else {}),
            **({"max_tokens": self.max_tokens} if self.max_tokens is not None else {}),
            **({"stream_options": {"include_usage": True}} if self.stream else {}),
            **self.extra_params,
        }

        if prompt_stack.tools and self.use_native_tools:
            params["tool_choice"] = self.tool_choice
            params["parallel_tool_calls"] = self.parallel_tool_calls

        if prompt_stack.output_schema is not None:
            if self.structured_output_strategy == "native":
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Output",
                        "schema": prompt_stack.output_schema.json_schema("Output"),
                        "strict": True,
                    },
                }
            elif self.structured_output_strategy == "tool" and self.use_native_tools:
                params["tool_choice"] = "required"

        if self.response_format is not None:
            if self.response_format == {"type": "json_object"}:
                params["response_format"] = self.response_format
                # JSON mode still requires a system message instructing the LLM to output JSON.
                prompt_stack.add_system_message("Provide your response as a valid JSON object.")
            else:
                params["response_format"] = self.response_format

        if prompt_stack.tools and self.use_native_tools:
            params["tools"] = self.__to_openai_tools(prompt_stack.tools)

        messages = self.__to_openai_messages(prompt_stack.messages)

        params["messages"] = messages

        return params

    def __to_openai_messages(self, messages: list[Message]) -> list[dict]:
        openai_messages = []

        for message in messages:
            # If the message only contains textual content we can send it as a single content.
            if message.has_all_content_type(TextMessageContent):
                openai_messages.append({"role": self.__to_openai_role(message), "content": message.to_text()})
            # Action results must be sent as separate messages.
            elif action_result_contents := message.get_content_type(ActionResultMessageContent):
                openai_messages.extend(
                    {
                        "role": self.__to_openai_role(message, action_result_content),
                        "content": self.__to_openai_message_content(action_result_content),
                        "tool_call_id": action_result_content.action.tag,
                    }
                    for action_result_content in action_result_contents
                )

                if message.has_any_content_type(TextMessageContent):
                    openai_messages.append({"role": self.__to_openai_role(message), "content": message.to_text()})
            else:
                openai_message = {
                    "role": self.__to_openai_role(message),
                    "content": [],
                }

                for content in message.content:
                    if isinstance(content, ActionCallMessageContent):
                        if "tool_calls" not in openai_message:
                            openai_message["tool_calls"] = []
                        openai_message["tool_calls"].append(self.__to_openai_message_content(content))
                    elif (
                        isinstance(content, AudioMessageContent)
                        and message.is_assistant()
                        and time.time() < content.artifact.meta.get("expires_at", float("inf"))
                    ):
                        openai_message["audio"] = {
                            "id": content.artifact.meta["audio_id"],
                        }
                    else:
                        openai_message["content"].append(self.__to_openai_message_content(content))

                # Some OpenAi-compatible services don't accept an empty array for content
                if not openai_message["content"]:
                    openai_message["content"] = ""

                openai_messages.append(openai_message)

        return openai_messages

    def __to_openai_role(self, message: Message, message_content: Optional[BaseMessageContent] = None) -> str:
        if message.is_system():
            return "system"
        elif message.is_assistant():
            return "assistant"
        else:
            if isinstance(message_content, ActionResultMessageContent):
                return "tool"
            else:
                return "user"

    def __to_openai_tools(self, tools: list[BaseTool]) -> list[dict]:
        return [
            {
                "function": {
                    "name": tool.to_native_tool_name(activity),
                    "description": tool.activity_description(activity),
                    "parameters": (tool.activity_schema(activity) or Schema({})).json_schema("Parameters Schema"),
                },
                "type": "function",
            }
            for tool in tools
            for activity in tool.activities()
        ]

    def __to_openai_message_content(self, content: BaseMessageContent) -> str | dict:
        if isinstance(content, TextMessageContent):
            return {"type": "text", "text": content.artifact.to_text()}
        elif isinstance(content, ImageMessageContent):
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{content.artifact.mime_type};base64,{content.artifact.base64}"},
            }
        elif isinstance(content, AudioMessageContent):
            artifact = content.artifact

            # We can't send the audio if it's expired.
            if int(time.time()) > artifact.meta.get("expires_at", float("inf")):
                return {
                    "type": "text",
                    "text": artifact.meta.get("transcript"),
                }
            else:
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(artifact.value).decode("utf-8"),
                        "format": artifact.format,
                    },
                }
        elif isinstance(content, ActionCallMessageContent):
            action = content.artifact.value

            return {
                "type": "function",
                "id": action.tag,
                "function": {"name": action.to_native_tool_name(), "arguments": json.dumps(action.input)},
            }
        elif isinstance(content, ActionResultMessageContent):
            return content.artifact.to_text()
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    def __to_prompt_stack_message_content(self, response: ChatCompletionMessage) -> list[BaseMessageContent]:
        content = []

        if response.content is not None:
            content.append(TextMessageContent(TextArtifact(response.content)))
        if response.audio is not None:
            content.append(
                AudioMessageContent(
                    AudioArtifact(
                        value=base64.b64decode(response.audio.data),
                        format="wav",
                        meta={
                            "audio_id": response.audio.id,
                            "transcript": response.audio.transcript,
                            "expires_at": response.audio.expires_at,
                        },
                    )
                )
            )
        if response.tool_calls is not None:
            content.extend(
                [
                    ActionCallMessageContent(
                        ActionArtifact(
                            ToolAction(
                                tag=tool_call.id,
                                name=ToolAction.from_native_tool_name(tool_call.function.name)[0],
                                path=ToolAction.from_native_tool_name(tool_call.function.name)[1],
                                input=json.loads(tool_call.function.arguments),
                            ),
                        ),
                    )
                    for tool_call in response.tool_calls
                ],
            )

        return content

    def __to_prompt_stack_delta_message_content(self, content_delta: ChoiceDelta) -> Optional[BaseDeltaMessageContent]:
        if content_delta.content is not None:
            return TextDeltaMessageContent(content_delta.content)
        elif content_delta.tool_calls is not None:
            tool_calls = content_delta.tool_calls

            if len(tool_calls) == 1:
                tool_call = tool_calls[0]
                index = tool_call.index

                if tool_call.function is not None:
                    # Tool call delta either contains the function header or the partial input.
                    if tool_call.id is not None and tool_call.function.name is not None:
                        return ActionCallDeltaMessageContent(
                            index=index,
                            tag=tool_call.id,
                            name=ToolAction.from_native_tool_name(tool_call.function.name)[0],
                            path=ToolAction.from_native_tool_name(tool_call.function.name)[1],
                        )
                    else:
                        return ActionCallDeltaMessageContent(index=index, partial_input=tool_call.function.arguments)
                else:
                    raise ValueError(f"Unsupported tool call delta: {tool_call}")
            else:
                raise ValueError(f"Unsupported tool call delta length: {len(tool_calls)}")
        # OpenAi doesn't have types for audio deltas so we need to use hasattr and getattr.
        elif hasattr(content_delta, "audio") and getattr(content_delta, "audio") is not None:
            audio_chunk: dict = getattr(content_delta, "audio")
            return AudioDeltaMessageContent(
                id=audio_chunk.get("id"),
                data=audio_chunk.get("data"),
                expires_at=audio_chunk.get("expires_at"),
                transcript=audio_chunk.get("transcript"),
            )
        return None
