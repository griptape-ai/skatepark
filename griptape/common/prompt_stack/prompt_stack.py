from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from attrs import define, field

from griptape.artifacts import (
    ActionArtifact,
    AudioArtifact,
    BaseArtifact,
    GenericArtifact,
    ImageArtifact,
    ListArtifact,
    TextArtifact,
)
from griptape.common import (
    ActionCallMessageContent,
    ActionResultMessageContent,
    AudioMessageContent,
    BaseMessageContent,
    GenericMessageContent,
    ImageMessageContent,
    Message,
    TextMessageContent,
)
from griptape.mixins.serializable_mixin import SerializableMixin

if TYPE_CHECKING:
    from schema import Schema

    from griptape.tools import BaseTool


@define
class PromptStack(SerializableMixin):
    messages: list[Message] = field(factory=list, kw_only=True, metadata={"serializable": True})
    tools: list[BaseTool] = field(factory=list, kw_only=True)
    output_schema: Optional[Schema] = field(default=None, kw_only=True)

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
        elif isinstance(artifact, AudioArtifact):
            return [AudioMessageContent(artifact)]
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
