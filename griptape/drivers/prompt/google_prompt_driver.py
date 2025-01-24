from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from attrs import Attribute, Factory, define, field

from griptape.artifacts import ActionArtifact, TextArtifact
from griptape.common import (
    ActionCallDeltaMessageContent,
    ActionCallMessageContent,
    ActionResultMessageContent,
    BaseDeltaMessageContent,
    BaseMessageContent,
    DeltaMessage,
    GenericMessageContent,
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
from griptape.tokenizers import BaseTokenizer, GoogleTokenizer
from griptape.utils import import_optional_dependency, remove_key_in_dict_recursively
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    from collections.abc import Iterator

    from google.generativeai.generative_models import GenerativeModel
    from google.generativeai.protos import Part
    from google.generativeai.types import ContentDict, ContentsType, GenerateContentResponse

    from griptape.drivers.prompt.base_prompt_driver import StructuredOutputStrategy
    from griptape.tools import BaseTool

logger = logging.getLogger(Defaults.logging_config.logger_name)


@define
class GooglePromptDriver(BasePromptDriver):
    """Google Prompt Driver.

    Attributes:
        api_key: Google API key.
        model: Google model name.
        client: Custom `GenerativeModel` client.
        top_p: Optional value for top_p.
        top_k: Optional value for top_k.
    """

    api_key: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": False})
    model: str = field(kw_only=True, metadata={"serializable": True})
    tokenizer: BaseTokenizer = field(
        default=Factory(lambda self: GoogleTokenizer(api_key=self.api_key, model=self.model), takes_self=True),
        kw_only=True,
    )
    top_p: Optional[float] = field(default=None, kw_only=True, metadata={"serializable": True})
    top_k: Optional[int] = field(default=None, kw_only=True, metadata={"serializable": True})
    use_native_tools: bool = field(default=True, kw_only=True, metadata={"serializable": True})
    structured_output_strategy: StructuredOutputStrategy = field(
        default="tool", kw_only=True, metadata={"serializable": True}
    )
    tool_choice: str = field(default="auto", kw_only=True, metadata={"serializable": True})
    _client: GenerativeModel = field(default=None, kw_only=True, alias="client", metadata={"serializable": False})

    @structured_output_strategy.validator  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    def validate_structured_output_strategy(self, _: Attribute, value: str) -> str:
        if value == "native":
            raise ValueError(f"{__class__.__name__} does not support `{value}` structured output strategy.")

        return value

    @lazy_property()
    def client(self) -> GenerativeModel:
        genai = import_optional_dependency("google.generativeai")
        genai.configure(api_key=self.api_key)

        return genai.GenerativeModel(self.model)

    @observable
    def try_run(self, prompt_stack: PromptStack) -> Message:
        messages = self.__to_google_messages(prompt_stack)
        params = self._base_params(prompt_stack)
        logging.debug((messages, params))
        response: GenerateContentResponse = self.client.generate_content(messages, **params)
        logging.debug(response.to_dict())

        usage_metadata = response.usage_metadata

        return Message(
            content=[self.__to_prompt_stack_message_content(part) for part in response.parts],
            role=Message.ASSISTANT_ROLE,
            usage=Message.Usage(
                input_tokens=usage_metadata.prompt_token_count,
                output_tokens=usage_metadata.candidates_token_count,
            ),
        )

    @observable
    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]:
        messages = self.__to_google_messages(prompt_stack)
        params = {**self._base_params(prompt_stack), "stream": True}
        logging.debug((messages, params))
        response: GenerateContentResponse = self.client.generate_content(
            messages,
            **params,
        )

        prompt_token_count = None
        for chunk in response:
            logger.debug(chunk.to_dict())
            usage_metadata = chunk.usage_metadata

            content = self.__to_prompt_stack_delta_message_content(chunk.parts[0]) if chunk.parts else None
            # Only want to output the prompt token count once since it is static each chunk
            if prompt_token_count is None:
                prompt_token_count = usage_metadata.prompt_token_count
                yield DeltaMessage(
                    content=content,
                    usage=DeltaMessage.Usage(
                        input_tokens=usage_metadata.prompt_token_count,
                        output_tokens=usage_metadata.candidates_token_count,
                    ),
                )
            else:
                yield DeltaMessage(
                    content=content,
                    usage=DeltaMessage.Usage(output_tokens=usage_metadata.candidates_token_count),
                )

    def _base_params(self, prompt_stack: PromptStack) -> dict:
        types = import_optional_dependency("google.generativeai.types")
        protos = import_optional_dependency("google.generativeai.protos")

        system_messages = prompt_stack.system_messages
        if system_messages:
            self.client._system_instruction = types.ContentDict(
                role="system",
                parts=[protos.Part(text=system_message.to_text()) for system_message in system_messages],
            )

        params = {
            "generation_config": types.GenerationConfig(
                **{
                    # For some reason, providing stop sequences when streaming breaks native functions
                    # https://github.com/google-gemini/generative-ai-python/issues/446
                    "stop_sequences": [] if self.stream and self.use_native_tools else self.tokenizer.stop_sequences,
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    **self.extra_params,
                },
            ),
        }

        if prompt_stack.tools and self.use_native_tools:
            params["tool_config"] = {"function_calling_config": {"mode": self.tool_choice}}

            if prompt_stack.output_schema is not None and self.structured_output_strategy == "tool":
                params["tool_config"]["function_calling_config"]["mode"] = "auto"

            params["tools"] = self.__to_google_tools(prompt_stack.tools)

        return params

    def __to_google_messages(self, prompt_stack: PromptStack) -> ContentsType:
        types = import_optional_dependency("google.generativeai.types")

        return [
            types.ContentDict(
                {
                    "role": self.__to_google_role(message),
                    "parts": [self.__to_google_message_content(content) for content in message.content],
                },
            )
            for message in prompt_stack.messages
            if not message.is_system()
        ]

    def __to_google_role(self, message: Message) -> str:
        if message.is_assistant():
            return "model"
        else:
            return "user"

    def __to_google_tools(self, tools: list[BaseTool]) -> list[dict]:
        types = import_optional_dependency("google.generativeai.types")

        tool_declarations = []
        for tool in tools:
            for activity in tool.activities():
                schema = tool.to_activity_json_schema(activity, "Parameters Schema")

                if "values" in schema["properties"]:
                    schema = schema["properties"]["values"]

                schema = remove_key_in_dict_recursively(schema, "additionalProperties")
                tool_declaration = types.FunctionDeclaration(
                    name=tool.to_native_tool_name(activity),
                    description=tool.activity_description(activity),
                    **(
                        {
                            "parameters": {
                                "type": schema["type"],
                                "properties": schema["properties"],
                                "required": schema.get("required", []),
                            }
                        }
                        if schema.get("properties")
                        else {}
                    ),
                )

                tool_declarations.append(tool_declaration)

        return tool_declarations

    def __to_google_message_content(self, content: BaseMessageContent) -> ContentDict | Part | str:
        types = import_optional_dependency("google.generativeai.types")
        protos = import_optional_dependency("google.generativeai.protos")

        if isinstance(content, TextMessageContent):
            return content.artifact.to_text()
        elif isinstance(content, ImageMessageContent):
            return types.ContentDict(mime_type=content.artifact.mime_type, data=content.artifact.value)
        elif isinstance(content, ActionCallMessageContent):
            action = content.artifact.value

            return protos.Part(function_call=protos.FunctionCall(name=action.tag, args=action.input))
        elif isinstance(content, ActionResultMessageContent):
            artifact = content.artifact

            return protos.Part(
                function_response=protos.FunctionResponse(
                    name=content.action.to_native_tool_name(),
                    response=artifact.to_dict(),
                ),
            )
        elif isinstance(content, GenericMessageContent):
            return content.artifact.value
        else:
            raise ValueError(f"Unsupported prompt stack content type: {type(content)}")

    def __to_prompt_stack_message_content(self, content: Part) -> BaseMessageContent:
        json_format = import_optional_dependency("google.protobuf.json_format")

        if content.text:
            return TextMessageContent(TextArtifact(content.text))
        elif content.function_call:
            function_call = content.function_call

            name, path = ToolAction.from_native_tool_name(function_call.name)

            args = json_format.MessageToDict(function_call._pb).get("args", {})
            return ActionCallMessageContent(
                artifact=ActionArtifact(value=ToolAction(tag=function_call.name, name=name, path=path, input=args)),
            )
        else:
            raise ValueError(f"Unsupported message content type {content}")

    def __to_prompt_stack_delta_message_content(self, content: Part) -> BaseDeltaMessageContent:
        json_format = import_optional_dependency("google.protobuf.json_format")

        if content.text:
            return TextDeltaMessageContent(content.text)
        elif content.function_call:
            function_call = content.function_call

            name, path = ToolAction.from_native_tool_name(function_call.name)

            args = json_format.MessageToDict(function_call._pb).get("args", {})
            return ActionCallDeltaMessageContent(
                tag=function_call.name,
                name=name,
                path=path,
                partial_input=json.dumps(args),
            )
        else:
            raise ValueError(f"Unsupported message content type {content}")
