from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from attrs import Factory, define, field

from griptape.artifacts import ActionArtifact, TextArtifact
from griptape.common import (
    ActionCallDeltaMessageContent,
    ActionCallMessageContent,
    ActionResultMessageContent,
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
from griptape.configs import Defaults
from griptape.drivers.prompt import BasePromptDriver
from griptape.tokenizers import SimpleTokenizer
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

logger = logging.getLogger(Defaults.logging_config.logger_name)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ollama import ChatResponse, Client

    from griptape.tokenizers.base_tokenizer import BaseTokenizer
    from griptape.tools import BaseTool


@define
class OllamaPromptDriver(BasePromptDriver):
    """Ollama Prompt Driver.

    Attributes:
        model: Model name.
    """

    model: str = field(kw_only=True, metadata={"serializable": True})
    host: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})
    tokenizer: BaseTokenizer = field(
        default=Factory(
            lambda self: SimpleTokenizer(
                characters_per_token=4,
                max_input_tokens=2000,
                max_output_tokens=self.max_tokens,
            ),
            takes_self=True,
        ),
        kw_only=True,
    )
    options: dict = field(
        default=Factory(
            lambda self: {
                "temperature": self.temperature,
                "stop": self.tokenizer.stop_sequences,
                "num_predict": self.max_tokens,
            },
            takes_self=True,
        ),
        kw_only=True,
    )
    use_native_tools: bool = field(default=True, kw_only=True, metadata={"serializable": True})
    _client: Client = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @lazy_property()
    def client(self) -> Client:
        return import_optional_dependency("ollama").Client(host=self.host)

    @observable
    def try_run(self, prompt_stack: PromptStack) -> Message:
        params = self._base_params(prompt_stack)
        logger.debug(params)
        response = self.client.chat(**params)
        logger.debug(response.model_dump())

        return Message(
            content=self.__to_prompt_stack_message_content(response),
            role=Message.ASSISTANT_ROLE,
        )

    @observable
    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]:
        params = {**self._base_params(prompt_stack), "stream": True}
        logger.debug(params)
        stream: Iterator = self.client.chat(**params)

        tool_index = 0
        for chunk in stream:
            logger.debug(chunk)
            message_content = self.__to_prompt_stack_delta_message_content(chunk)
            # Ollama provides multiple Tool calls as separate chunks but with no index to differentiate them.
            # So we must keep track of the index ourselves.
            if isinstance(message_content, ActionCallDeltaMessageContent):
                message_content.index = tool_index
                tool_index += 1
            yield DeltaMessage(content=message_content)

    def _base_params(self, prompt_stack: PromptStack) -> dict:
        messages = self._prompt_stack_to_messages(prompt_stack)

        params = {
            "messages": messages,
            "model": self.model,
            "options": self.options,
            **self.extra_params,
        }

        if prompt_stack.output_schema is not None and self.structured_output_strategy == "native":
            params["format"] = prompt_stack.to_output_json_schema()

        # Tool calling is only supported when not streaming
        if prompt_stack.tools and self.use_native_tools:
            params["tools"] = self.__to_ollama_tools(prompt_stack.tools)

        return params

    def _prompt_stack_to_messages(self, prompt_stack: PromptStack) -> list[dict]:
        ollama_messages = []
        for message in prompt_stack.messages:
            action_result_contents = message.get_content_type(ActionResultMessageContent)

            # Function calls need to be handled separately from the rest of the message content
            if action_result_contents:
                ollama_messages.extend(
                    [
                        {
                            "role": self.__to_ollama_role(message, action_result_content),
                            "content": self.__to_ollama_message_content(action_result_content),
                        }
                        for action_result_content in action_result_contents
                    ],
                )

                text_contents = message.get_content_type(TextMessageContent)
                if text_contents:
                    ollama_messages.append({"role": self.__to_ollama_role(message), "content": message.to_text()})
            else:
                ollama_message: dict[str, Any] = {
                    "role": self.__to_ollama_role(message),
                    "content": message.to_text(),
                }

                action_call_contents = message.get_content_type(ActionCallMessageContent)
                if action_call_contents:
                    ollama_message["tool_calls"] = [
                        self.__to_ollama_message_content(action_call_content)
                        for action_call_content in action_call_contents
                    ]

                image_contents = message.get_content_type(ImageMessageContent)
                if image_contents:
                    ollama_message["images"] = [
                        self.__to_ollama_message_content(image_content) for image_content in image_contents
                    ]

                ollama_messages.append(ollama_message)

        return ollama_messages

    def __to_ollama_message_content(self, content: BaseMessageContent) -> str | dict:
        if isinstance(content, TextMessageContent):
            return content.artifact.to_text()
        elif isinstance(content, ImageMessageContent):
            return content.artifact.base64
        elif isinstance(content, ActionCallMessageContent):
            action = content.artifact.value

            return {
                "type": "function",
                "id": action.tag,
                "function": {"name": action.to_native_tool_name(), "arguments": action.input},
            }
        elif isinstance(content, ActionResultMessageContent):
            return content.artifact.to_text()
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    def __to_ollama_tools(self, tools: list[BaseTool]) -> list[dict]:
        ollama_tools = []

        for tool in tools:
            for activity in tool.activities():
                ollama_tool = {
                    "function": {
                        "name": tool.to_native_tool_name(activity),
                        "description": tool.activity_description(activity),
                    },
                    "type": "function",
                }

                activity_schema = tool.activity_schema(activity)
                if activity_schema is not None:
                    ollama_tool["function"]["parameters"] = tool.to_activity_json_schema(activity, "Parameters Schema")[
                        "properties"
                    ]["values"]

                ollama_tools.append(ollama_tool)
        return ollama_tools

    def __to_ollama_role(self, message: Message, message_content: Optional[BaseMessageContent] = None) -> str:
        if message.is_system():
            return "system"
        elif message.is_assistant():
            return "assistant"
        else:
            if isinstance(message_content, ActionResultMessageContent):
                return "tool"
            else:
                return "user"

    def __to_prompt_stack_message_content(self, response: ChatResponse) -> list[BaseMessageContent]:
        content = []
        message = response["message"]

        if "content" in message and message["content"]:
            content.append(TextMessageContent(TextArtifact(response["message"]["content"])))
        if "tool_calls" in message:
            content.extend(
                [
                    ActionCallMessageContent(
                        ActionArtifact(
                            ToolAction(
                                tag=tool_call["function"]["name"],
                                name=ToolAction.from_native_tool_name(tool_call["function"]["name"])[0],
                                path=ToolAction.from_native_tool_name(tool_call["function"]["name"])[1],
                                input=tool_call["function"]["arguments"],
                            ),
                        ),
                    )
                    for tool_call in message["tool_calls"]
                ],
            )

        return content

    def __to_prompt_stack_delta_message_content(self, content_delta: ChatResponse) -> BaseDeltaMessageContent:
        message = content_delta["message"]
        if "content" in message and message["content"]:
            return TextDeltaMessageContent(message["content"])
        elif "tool_calls" in message and len(message["tool_calls"]):
            tool_calls = message["tool_calls"]

            # Ollama doesn't _really_ support Tool streaming. They provide the full tool call at once.
            # Multiple, parallel, Tool calls are provided as multiple content deltas.
            # Tracking here: https://github.com/ollama/ollama/issues/7886
            tool_call = tool_calls[0]

            return ActionCallDeltaMessageContent(
                tag=tool_call["function"]["name"],
                name=ToolAction.from_native_tool_name(tool_call["function"]["name"])[0],
                path=ToolAction.from_native_tool_name(tool_call["function"]["name"])[1],
                partial_input=json.dumps(tool_call["function"]["arguments"]),
            )
        else:
            return TextDeltaMessageContent("")
