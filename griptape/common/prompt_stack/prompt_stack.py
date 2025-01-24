from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from attrs import define, field
from pydantic import BaseModel
from schema import Schema

from griptape.artifacts import (
    ActionArtifact,
    BaseArtifact,
    GenericArtifact,
    ImageArtifact,
    ListArtifact,
    TextArtifact,
)
from griptape.common import (
    ActionCallMessageContent,
    ActionResultMessageContent,
    BaseMessageContent,
    GenericMessageContent,
    ImageMessageContent,
    Message,
    TextMessageContent,
)
from griptape.mixins.serializable_mixin import SerializableMixin
from griptape.utils.json_schema_utils import build_strict_schema

if TYPE_CHECKING:
    from griptape.tools import BaseTool


@define
class PromptStack(SerializableMixin):
    messages: list[Message] = field(factory=list, kw_only=True, metadata={"serializable": True})
    tools: list[BaseTool] = field(factory=list, kw_only=True)
    output_schema: Optional[Union[Schema, type[BaseModel]]] = field(default=None, kw_only=True)

    @property
    def system_messages(self) -> list[Message]:
        return [message for message in self.messages if message.is_system()]

    @property
    def user_messages(self) -> list[Message]:
        return [message for message in self.messages if message.is_user()]

    @property
    def assistant_messages(self) -> list[Message]:
        return [message for message in self.messages if message.is_assistant()]

    def add_message(self, artifact: str | BaseArtifact, role: str) -> Message:
        content = self.__to_message_content(artifact)

        self.messages.append(Message(content=content, role=role))

        return self.messages[-1]

    def add_system_message(self, artifact: str | BaseArtifact) -> Message:
        return self.add_message(artifact, Message.SYSTEM_ROLE)

    def add_user_message(self, artifact: str | BaseArtifact) -> Message:
        return self.add_message(artifact, Message.USER_ROLE)

    def add_assistant_message(self, artifact: str | BaseArtifact) -> Message:
        return self.add_message(artifact, Message.ASSISTANT_ROLE)

    def to_output_json_schema(self, schema_id: str = "Output Format") -> dict:
        if self.output_schema is None:
            raise ValueError("Output schema is not set")

        if isinstance(self.output_schema, Schema):
            json_schema = self.output_schema.json_schema(schema_id)
        elif isinstance(self.output_schema, type) and issubclass(self.output_schema, BaseModel):
            json_schema = build_strict_schema(self.output_schema.model_json_schema(), schema_id)
        else:
            raise ValueError(f"Unsupported output schema type: {type(self.output_schema)}")

        return json_schema

    @classmethod
    def from_artifact(cls, artifact: BaseArtifact) -> PromptStack:
        prompt_stack = cls()
        prompt_stack.add_user_message(artifact)

        return prompt_stack

    def __to_message_content(self, artifact: str | BaseArtifact) -> list[BaseMessageContent]:
        if isinstance(artifact, str):
            return [TextMessageContent(TextArtifact(artifact))]
        elif isinstance(artifact, TextArtifact):
            return [TextMessageContent(artifact)]
        elif isinstance(artifact, ImageArtifact):
            return [ImageMessageContent(artifact)]
        elif isinstance(artifact, GenericArtifact):
            return [GenericMessageContent(artifact)]
        elif isinstance(artifact, ActionArtifact):
            action = artifact.value
            output = action.output
            if output is None:
                return [ActionCallMessageContent(artifact)]
            else:
                return [ActionResultMessageContent(output, action=action)]
        elif isinstance(artifact, ListArtifact):
            processed_contents = [self.__to_message_content(artifact) for artifact in artifact.value]

            return [sub_content for processed_content in processed_contents for sub_content in processed_content]
        else:
            return [TextMessageContent(TextArtifact(artifact.to_text()))]
