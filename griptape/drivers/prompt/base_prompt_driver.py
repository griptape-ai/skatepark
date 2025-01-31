from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional

from attrs import Factory, define, field

from griptape.artifacts import BaseArtifact, TextArtifact
from griptape.common import (
    ActionCallDeltaMessageContent,
    ActionCallMessageContent,
    BaseDeltaMessageContent,
    DeltaMessage,
    Message,
    PromptStack,
    TextDeltaMessageContent,
    TextMessageContent,
    observable,
)
from griptape.events import (
    ActionChunkEvent,
    EventBus,
    FinishPromptEvent,
    StartPromptEvent,
    TextChunkEvent,
)
from griptape.mixins.exponential_backoff_mixin import ExponentialBackoffMixin
from griptape.mixins.serializable_mixin import SerializableMixin
from griptape.rules.json_schema_rule import JsonSchemaRule

if TYPE_CHECKING:
    from collections.abc import Iterator

    from griptape.tokenizers import BaseTokenizer

StructuredOutputStrategy = Literal["native", "tool", "rule"]


@define(kw_only=True)
class BasePromptDriver(SerializableMixin, ExponentialBackoffMixin, ABC):
    """Base class for the Prompt Drivers.

    Attributes:
        temperature: The temperature to use for the completion.
        max_tokens: The maximum number of tokens to generate. If not specified, the value will be automatically generated based by the tokenizer.
        prompt_stack_to_string: A function that converts a `PromptStack` to a string.
        ignored_exception_types: A tuple of exception types to ignore.
        model: The model name.
        tokenizer: An instance of `BaseTokenizer` to when calculating tokens.
        stream: Whether to stream the completion or not. `CompletionChunkEvent`s will be published to the `Structure` if one is provided.
        use_native_tools: Whether to use LLM's native function calling capabilities. Must be supported by the model.
        extra_params: Extra parameters to pass to the model.
    """

    temperature: float = field(default=0.1, metadata={"serializable": True})
    max_tokens: Optional[int] = field(default=None, metadata={"serializable": True})
    ignored_exception_types: tuple[type[Exception], ...] = field(default=Factory(lambda: (ImportError, ValueError)))
    model: str = field(metadata={"serializable": True})
    tokenizer: BaseTokenizer
    stream: bool = field(default=False, kw_only=True, metadata={"serializable": True})
    use_native_tools: bool = field(default=False, kw_only=True, metadata={"serializable": True})
    structured_output_strategy: StructuredOutputStrategy = field(
        default="rule", kw_only=True, metadata={"serializable": True}
    )
    extra_params: dict = field(factory=dict, kw_only=True, metadata={"serializable": True})

    def before_run(self, prompt_stack: PromptStack) -> None:
        self._init_structured_output(prompt_stack)
        EventBus.publish_event(StartPromptEvent(model=self.model, prompt_stack=prompt_stack))

    def after_run(self, result: Message) -> None:
        EventBus.publish_event(
            FinishPromptEvent(
                model=self.model,
                result=result.value,
                input_token_count=result.usage.input_tokens,
                output_token_count=result.usage.output_tokens,
            ),
        )

    @observable(tags=["PromptDriver.run()"])
    def run(self, prompt_input: PromptStack | BaseArtifact) -> Message:
        if isinstance(prompt_input, BaseArtifact):
            prompt_stack = PromptStack.from_artifact(prompt_input)
        else:
            prompt_stack = prompt_input

        for attempt in self.retrying():
            with attempt:
                self.before_run(prompt_stack)

                result = self.__process_stream(prompt_stack) if self.stream else self.__process_run(prompt_stack)

                self.after_run(result)

                return result
        else:
            raise Exception("prompt driver failed after all retry attempts")

    def prompt_stack_to_string(self, prompt_stack: PromptStack) -> str:
        """Converts a Prompt Stack to a string for token counting or model prompt_input.

        This base implementation is only a rough approximation, and should be overridden by subclasses with model-specific tokens.

        Args:
            prompt_stack: The Prompt Stack to convert to a string.

        Returns:
            A single string representation of the Prompt Stack.
        """
        prompt_lines = []

        for i in prompt_stack.messages:
            content = i.to_text()
            if i.is_user():
                prompt_lines.append(f"User: {content}")
            elif i.is_assistant():
                prompt_lines.append(f"Assistant: {content}")
            else:
                prompt_lines.append(content)

        prompt_lines.append("Assistant:")

        return "\n\n".join(prompt_lines)

    @abstractmethod
    def try_run(self, prompt_stack: PromptStack) -> Message: ...

    @abstractmethod
    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]: ...

    def _init_structured_output(self, prompt_stack: PromptStack) -> None:
        from griptape.tools import StructuredOutputTool

        if (output_schema := prompt_stack.output_schema) is not None:
            if self.structured_output_strategy == "tool":
                structured_output_tool = StructuredOutputTool(output_schema=output_schema)
                if structured_output_tool not in prompt_stack.tools:
                    prompt_stack.tools.append(structured_output_tool)
            elif self.structured_output_strategy == "rule":
                output_artifact = TextArtifact(JsonSchemaRule(prompt_stack.to_output_json_schema()).to_text())
                system_messages = prompt_stack.system_messages
                if system_messages:
                    last_system_message = prompt_stack.system_messages[-1]
                    last_system_message.content.extend(
                        [
                            TextMessageContent(TextArtifact("\n\n")),
                            TextMessageContent(output_artifact),
                        ]
                    )
                else:
                    prompt_stack.messages.insert(
                        0,
                        Message(
                            content=[TextMessageContent(output_artifact)],
                            role=Message.SYSTEM_ROLE,
                        ),
                    )

    def __process_run(self, prompt_stack: PromptStack) -> Message:
        return self.try_run(prompt_stack)

    def __process_stream(self, prompt_stack: PromptStack) -> Message:
        delta_contents: dict[int, list[BaseDeltaMessageContent]] = {}
        usage = DeltaMessage.Usage()

        # Aggregate all content deltas from the stream
        message_deltas = self.try_stream(prompt_stack)
        for message_delta in message_deltas:
            usage += message_delta.usage
            content = message_delta.content

            if content is not None:
                if content.index in delta_contents:
                    delta_contents[content.index].append(content)
                else:
                    delta_contents[content.index] = [content]
                if isinstance(content, TextDeltaMessageContent):
                    EventBus.publish_event(TextChunkEvent(token=content.text, index=content.index))
                elif isinstance(content, ActionCallDeltaMessageContent):
                    EventBus.publish_event(
                        ActionChunkEvent(
                            partial_input=content.partial_input,
                            tag=content.tag,
                            name=content.name,
                            path=content.path,
                            index=content.index,
                        ),
                    )

        # Build a complete content from the content deltas
        return self.__build_message(list(delta_contents.values()), usage)

    def __build_message(
        self, delta_contents: list[list[BaseDeltaMessageContent]], usage: DeltaMessage.Usage
    ) -> Message:
        content = []
        for delta_content in delta_contents:
            text_deltas = [delta for delta in delta_content if isinstance(delta, TextDeltaMessageContent)]
            action_deltas = [delta for delta in delta_content if isinstance(delta, ActionCallDeltaMessageContent)]

            if text_deltas:
                content.append(TextMessageContent.from_deltas(text_deltas))
            if action_deltas:
                content.append(ActionCallMessageContent.from_deltas(action_deltas))

        return Message(
            content=content,
            role=Message.ASSISTANT_ROLE,
            usage=Message.Usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens),
        )
