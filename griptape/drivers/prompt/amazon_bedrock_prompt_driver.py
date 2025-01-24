from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from attrs import Attribute, Factory, define, field

from griptape.artifacts import (
    ActionArtifact,
    BaseArtifact,
    ErrorArtifact,
    ImageArtifact,
    InfoArtifact,
    ListArtifact,
    TextArtifact,
)
from griptape.common import (
    ActionCallDeltaMessageContent,
    ActionCallMessageContent,
    ActionResultMessageContent,
    BaseDeltaMessageContent,
    BaseMessageContent,
    DeltaMessage,
    ImageMessageContent,
    Message,
    TextDeltaMessageContent,
    TextMessageContent,
    ToolAction,
    observable,
)
from griptape.configs import Defaults
from griptape.drivers.prompt import BasePromptDriver
from griptape.tokenizers import AmazonBedrockTokenizer, BaseTokenizer
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from collections.abc import Iterator

    import boto3

    from griptape.common import PromptStack
    from griptape.drivers.prompt.base_prompt_driver import StructuredOutputStrategy
    from griptape.tools import BaseTool

logger = logging.getLogger(Defaults.logging_config.logger_name)


@define
class AmazonBedrockPromptDriver(BasePromptDriver):
    session: boto3.Session = field(default=Factory(lambda: import_optional_dependency("boto3").Session()), kw_only=True)
    additional_model_request_fields: dict = field(default=Factory(dict), kw_only=True)
    tokenizer: BaseTokenizer = field(
        default=Factory(lambda self: AmazonBedrockTokenizer(model=self.model), takes_self=True),
        kw_only=True,
    )
    use_native_tools: bool = field(default=True, kw_only=True, metadata={"serializable": True})
    structured_output_strategy: StructuredOutputStrategy = field(
        default="tool", kw_only=True, metadata={"serializable": True}
    )
    tool_choice: dict = field(default=Factory(lambda: {"auto": {}}), kw_only=True, metadata={"serializable": True})
    _client: Any = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @structured_output_strategy.validator  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    def validate_structured_output_strategy(self, _: Attribute, value: str) -> str:
        if value == "native":
            raise ValueError(f"{__class__.__name__} does not support `{value}` structured output strategy.")

        return value

    @lazy_property()
    def client(self) -> Any:
        return self.session.client("bedrock-runtime")

    @observable
    def try_run(self, prompt_stack: PromptStack) -> Message:
        params = self._base_params(prompt_stack)
        logger.debug(params)
        response = self.client.converse(**params)
        logger.debug(response)

        usage = response["usage"]
        output_message = response["output"]["message"]

        return Message(
            content=[self.__to_prompt_stack_message_content(content) for content in output_message["content"]],
            role=Message.ASSISTANT_ROLE,
            usage=Message.Usage(input_tokens=usage["inputTokens"], output_tokens=usage["outputTokens"]),
        )

    @observable
    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]:
        params = self._base_params(prompt_stack)
        logger.debug(params)
        response = self.client.converse_stream(**params)

        stream = response.get("stream")
        if stream is not None:
            for event in stream:
                logger.debug(event)
                if "contentBlockDelta" in event or "contentBlockStart" in event:
                    yield DeltaMessage(content=self.__to_prompt_stack_delta_message_content(event))
                elif "metadata" in event:
                    usage = event["metadata"]["usage"]
                    yield DeltaMessage(
                        usage=DeltaMessage.Usage(
                            input_tokens=usage["inputTokens"],
                            output_tokens=usage["outputTokens"],
                        ),
                    )
        else:
            raise Exception("model response is empty")

    def _base_params(self, prompt_stack: PromptStack) -> dict:
        system_messages = [{"text": message.to_text()} for message in prompt_stack.system_messages]
        messages = self.__to_bedrock_messages([message for message in prompt_stack.messages if not message.is_system()])

        params = {
            "modelId": self.model,
            "messages": messages,
            "system": system_messages,
            "inferenceConfig": {
                "temperature": self.temperature,
                **({"maxTokens": self.max_tokens} if self.max_tokens is not None else {}),
            },
            "additionalModelRequestFields": self.additional_model_request_fields,
            **self.extra_params,
        }

        if prompt_stack.tools and self.use_native_tools:
            params["toolConfig"] = {
                "tools": [],
                "toolChoice": self.tool_choice,
            }

            if prompt_stack.output_schema is not None and self.structured_output_strategy == "tool":
                params["toolConfig"]["toolChoice"] = {"any": {}}

            params["toolConfig"]["tools"] = self.__to_bedrock_tools(prompt_stack.tools)

        return params

    def __to_bedrock_messages(self, messages: list[Message]) -> list[dict]:
        return [
            {
                "role": self.__to_bedrock_role(message),
                "content": [self.__to_bedrock_message_content(content) for content in message.content],
            }
            for message in messages
        ]

    def __to_bedrock_role(self, message: Message) -> str:
        if message.is_assistant():
            return "assistant"
        else:
            return "user"

    def __to_bedrock_tools(self, tools: list[BaseTool]) -> list[dict]:
        return [
            {
                "toolSpec": {
                    "name": tool.to_native_tool_name(activity),
                    "description": tool.activity_description(activity),
                    "inputSchema": {
                        "json": tool.to_activity_json_schema(activity, "http://json-schema.org/draft-07/schema#"),
                    },
                },
            }
            for tool in tools
            for activity in tool.activities()
        ]

    def __to_bedrock_message_content(self, content: BaseMessageContent) -> dict:
        if isinstance(content, TextMessageContent):
            return {"text": content.artifact.to_text()}
        elif isinstance(content, ImageMessageContent):
            artifact = content.artifact

            return {"image": {"format": artifact.format, "source": {"bytes": artifact.value}}}
        elif isinstance(content, ActionCallMessageContent):
            action_call = content.artifact.value

            return {
                "toolUse": {
                    "toolUseId": action_call.tag,
                    "name": f"{action_call.name}_{action_call.path}",
                    "input": action_call.input,
                },
            }
        elif isinstance(content, ActionResultMessageContent):
            artifact = content.artifact

            if isinstance(artifact, ListArtifact):
                message_content = [self.__to_bedrock_tool_use_content(artifact) for artifact in artifact.value]
            else:
                message_content = [self.__to_bedrock_tool_use_content(artifact)]

            return {
                "toolResult": {
                    "toolUseId": content.action.tag,
                    "content": message_content,
                    "status": "error" if isinstance(artifact, ErrorArtifact) else "success",
                },
            }
        else:
            return content.artifact.value

    def __to_bedrock_tool_use_content(self, artifact: BaseArtifact) -> dict:
        if isinstance(artifact, ImageArtifact):
            return {"image": {"format": artifact.format, "source": {"bytes": artifact.value}}}
        elif isinstance(artifact, (TextArtifact, ErrorArtifact, InfoArtifact)):
            return {"text": artifact.to_text()}
        else:
            raise ValueError(f"Unsupported artifact type: {type(artifact)}")

    def __to_prompt_stack_message_content(self, content: dict) -> BaseMessageContent:
        if "text" in content:
            return TextMessageContent(TextArtifact(content["text"]))
        elif "toolUse" in content:
            name, path = ToolAction.from_native_tool_name(content["toolUse"]["name"])
            return ActionCallMessageContent(
                artifact=ActionArtifact(
                    value=ToolAction(
                        tag=content["toolUse"]["toolUseId"],
                        name=name,
                        path=path,
                        input=content["toolUse"]["input"],
                    ),
                ),
            )
        else:
            raise ValueError(f"Unsupported message content type: {content}")

    def __to_prompt_stack_delta_message_content(self, event: dict) -> BaseDeltaMessageContent:
        if "contentBlockStart" in event:
            content_block = event["contentBlockStart"]["start"]

            if "toolUse" in content_block:
                name, path = ToolAction.from_native_tool_name(content_block["toolUse"]["name"])

                return ActionCallDeltaMessageContent(
                    index=event["contentBlockStart"]["contentBlockIndex"],
                    tag=content_block["toolUse"]["toolUseId"],
                    name=name,
                    path=path,
                )
            elif "text" in content_block:
                return TextDeltaMessageContent(
                    content_block["text"],
                    index=event["contentBlockStart"]["contentBlockIndex"],
                )
            else:
                raise ValueError(f"Unsupported message content type: {event}")
        elif "contentBlockDelta" in event:
            content_block_delta = event["contentBlockDelta"]

            if "text" in content_block_delta["delta"]:
                return TextDeltaMessageContent(
                    content_block_delta["delta"]["text"],
                    index=content_block_delta["contentBlockIndex"],
                )
            elif "toolUse" in content_block_delta["delta"]:
                return ActionCallDeltaMessageContent(
                    index=content_block_delta["contentBlockIndex"],
                    partial_input=content_block_delta["delta"]["toolUse"]["input"],
                )
            else:
                raise ValueError(f"Unsupported message content type: {event}")
        else:
            raise ValueError(f"Unsupported message content type: {event}")
