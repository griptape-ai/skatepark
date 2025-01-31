from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from attrs import Factory, define, field

from griptape.artifacts import ActionArtifact, TextArtifact
from griptape.artifacts.list_artifact import ListArtifact
from griptape.common import (
    ActionCallMessageContent,
    ActionResultMessageContent,
    BaseDeltaMessageContent,
    BaseMessageContent,
    DeltaMessage,
    Message,
    PromptStack,
    TextDeltaMessageContent,
    TextMessageContent,
    ToolAction,
    observable,
)
from griptape.common.prompt_stack.contents.action_call_delta_message_content import ActionCallDeltaMessageContent
from griptape.configs import Defaults
from griptape.drivers.prompt import BasePromptDriver
from griptape.tokenizers import BaseTokenizer, CohereTokenizer
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cohere import AssistantMessageResponse, ChatResponse, ClientV2, StreamedChatResponseV2

    from griptape.tools import BaseTool

logger = logging.getLogger(Defaults.logging_config.logger_name)


@define(kw_only=True)
class CoherePromptDriver(BasePromptDriver):
    """Cohere Prompt Driver.

    Attributes:
        api_key: Cohere API key.
        model: 	Cohere model name.
        client: Custom `cohere.Client`.
    """

    api_key: str = field(metadata={"serializable": False})
    model: str = field(metadata={"serializable": True})
    force_single_step: bool = field(default=False, kw_only=True, metadata={"serializable": True})
    use_native_tools: bool = field(default=True, kw_only=True, metadata={"serializable": True})
    _client: ClientV2 = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})
    tokenizer: BaseTokenizer = field(
        default=Factory(lambda self: CohereTokenizer(model=self.model, client=self.client), takes_self=True),
    )

    @lazy_property()
    def client(self) -> ClientV2:
        return import_optional_dependency("cohere").ClientV2(self.api_key, log_warning_experimental_features=False)

    @observable
    def try_run(self, prompt_stack: PromptStack) -> Message:
        params = self._base_params(prompt_stack)

        logger.debug(params)

        result: ChatResponse = self.client.chat(**params)
        logger.debug(result.model_dump())

        return Message(
            content=self.__to_prompt_stack_message_content(result.message),
            role=Message.ASSISTANT_ROLE,
            usage=Message.Usage(
                input_tokens=result.usage.tokens.input_tokens if result.usage and result.usage.tokens else 0,
                output_tokens=result.usage.tokens.output_tokens if result.usage and result.usage.tokens else 0,
            ),
        )

    @observable
    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]:
        params = self._base_params(prompt_stack)
        logger.debug(params)
        result: Iterator[StreamedChatResponseV2] = self.client.chat_stream(**params)

        for event in result:
            if event.type == "stream-end":
                usage = event.response.meta.tokens

                yield DeltaMessage(
                    usage=DeltaMessage.Usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens),
                )
            elif event.type in ("tool-plan-delta", "content-delta", "tool-call-start", "tool-call-delta"):
                yield DeltaMessage(content=self.__to_prompt_stack_delta_message_content(event))

    def _base_params(self, prompt_stack: PromptStack) -> dict:
        tool_results = []

        messages = self.__to_cohere_messages(prompt_stack.messages)

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop_sequences": self.tokenizer.stop_sequences,
            "max_tokens": self.max_tokens,
            **({"tool_results": tool_results} if tool_results else {}),
            **self.extra_params,
        }

        if prompt_stack.output_schema is not None and self.structured_output_strategy == "native":
            params["response_format"] = {"type": "json_object", "schema": prompt_stack.to_output_json_schema()}

        if prompt_stack.tools and self.use_native_tools:
            params["tools"] = self.__to_cohere_tools(prompt_stack.tools)

        return params

    def __to_cohere_messages(self, messages: list[Message]) -> list[dict]:
        cohere_messages = []

        for message in messages:
            # If the message only contains textual content we can send it as a single content.
            if message.is_text():
                cohere_messages.append({"role": self.__to_cohere_role(message), "content": message.to_text()})
            # Action results must be sent as separate messages.
            elif message.has_any_content_type(ActionResultMessageContent):
                cohere_messages.extend(
                    {
                        "role": self.__to_cohere_role(message, action_result),
                        "content": self.__to_cohere_message_content(action_result),
                        "tool_call_id": action_result.action.tag,
                    }
                    for action_result in message.get_content_type(ActionResultMessageContent)
                )

                if message.has_any_content_type(TextMessageContent):
                    cohere_messages.append({"role": self.__to_cohere_role(message), "content": message.to_text()})
            else:
                cohere_message = {
                    "role": self.__to_cohere_role(message),
                    "content": [
                        self.__to_cohere_message_content(content)
                        for content in [
                            content for content in message.content if not isinstance(content, ActionCallMessageContent)
                        ]
                    ],
                }

                # Action calls must be attached to the message, not sent as content.
                action_call_content = [
                    content for content in message.content if isinstance(content, ActionCallMessageContent)
                ]
                if action_call_content:
                    cohere_message["tool_calls"] = [
                        self.__to_cohere_message_content(action_call) for action_call in action_call_content
                    ]

                cohere_messages.append(cohere_message)

        return cohere_messages

    def __to_cohere_message_content(self, content: BaseMessageContent) -> str | dict | list[dict]:
        if isinstance(content, ActionCallMessageContent):
            action = content.artifact.value

            return {
                "type": "function",
                "id": action.tag,
                "function": {
                    "name": action.to_native_tool_name(),
                    "arguments": json.dumps(action.input),
                },
            }
        elif isinstance(content, ActionResultMessageContent):
            artifact = content.artifact

            if isinstance(artifact, ListArtifact):
                message_content = [{"type": "text", "text": artifact.to_text()} for artifact in artifact.value]
            else:
                message_content = {"type": "text", "text": artifact.to_text()}

            return message_content
        else:
            return {"type": "text", "text": content.artifact.to_text()}

    def __to_cohere_role(self, message: Message, message_content: Optional[BaseMessageContent] = None) -> str:
        if message.is_system():
            return "system"
        elif message.is_assistant():
            return "assistant"
        else:
            if isinstance(message_content, ActionResultMessageContent):
                return "tool"
            else:
                return "user"

    def __to_cohere_tools(self, tools: list[BaseTool]) -> list[dict]:
        return [
            {
                "function": {
                    "name": tool.to_native_tool_name(activity),
                    "description": tool.activity_description(activity),
                    "parameters": tool.to_activity_json_schema(activity, "Parameters Schema"),
                },
                "type": "function",
            }
            for tool in tools
            for activity in tool.activities()
        ]

    def __to_prompt_stack_message_content(self, response: AssistantMessageResponse) -> list[BaseMessageContent]:
        content = []
        if response.content:
            content.extend([TextMessageContent(TextArtifact(content.text)) for content in response.content])
        if response.tool_plan:
            content.append(TextMessageContent(TextArtifact(response.tool_plan)))
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
                    if tool_call.id is not None
                    and tool_call.function is not None
                    and tool_call.function.name is not None
                    and tool_call.function.arguments is not None
                ],
            )

        return content

    def __to_prompt_stack_delta_message_content(self, event: Any) -> BaseDeltaMessageContent:
        if event.type == "content-delta":
            return TextDeltaMessageContent(event.delta.message.content.text, index=0)
        elif event.type == "tool-plan-delta":
            return TextDeltaMessageContent(event.delta.message["tool_plan"])
        elif event.type == "tool-call-start":
            tool_call_delta = event.delta.message["tool_calls"]
            name, path = ToolAction.from_native_tool_name(tool_call_delta["function"]["name"])

            return ActionCallDeltaMessageContent(tag=tool_call_delta["id"], name=name, path=path)
        elif event.type == "tool-call-delta":
            tool_call_delta = event.delta.message["tool_calls"]["function"]

            return ActionCallDeltaMessageContent(partial_input=tool_call_delta["arguments"])
        else:
            raise ValueError(f"Unsupported event type: {event.type}")
