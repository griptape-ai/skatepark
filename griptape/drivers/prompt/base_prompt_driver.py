from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Callable, Tuple, Type
from attr import define, field, Factory
from griptape.events import (
    StartPromptEvent,
    FinishPromptEvent,
    CompletionChunkEvent,
)
from griptape.utils import PromptStack
from griptape.mixins import ExponentialBackoffMixin
from griptape.tokenizers import BaseTokenizer
from griptape.artifacts import TextArtifact

if TYPE_CHECKING:
    from griptape.structures import Structure


@define
class BasePromptDriver(ExponentialBackoffMixin, ABC):
    temperature: float = field(default=0.1, kw_only=True)
    max_tokens: Optional[int] = field(default=None, kw_only=True)
    structure: Optional[Structure] = field(default=None, kw_only=True)
    prompt_stack_to_string: Callable[[PromptStack], str] = field(
        default=Factory(
            lambda self: self.default_prompt_stack_to_string_converter,
            takes_self=True,
        ),
        kw_only=True,
    )
    ignored_exception_types: Tuple[Type[Exception], ...] = field(
        default=Factory(lambda: (ImportError)), kw_only=True
    )
    model: str
    tokenizer: BaseTokenizer
    stream: bool = field(default=False, kw_only=True)

    def max_output_tokens(self, text: str) -> int:
        tokens_left = self.tokenizer.tokens_left(text)

        if self.max_tokens:
            return min(self.max_tokens, tokens_left)
        else:
            return tokens_left

    def token_count(self, prompt_stack: PromptStack) -> int:
        return self.tokenizer.token_count(
            self.prompt_stack_to_string(prompt_stack)
        )

    def before_run(self, prompt_stack: PromptStack) -> None:
        if self.structure:
            self.structure.publish_event(
                StartPromptEvent(token_count=self.token_count(prompt_stack))
            )

    def after_run(self, result: TextArtifact) -> None:
        if self.structure:
            self.structure.publish_event(
                FinishPromptEvent(
                    token_count=result.token_count(self.tokenizer)
                )
            )

    def run(self, prompt_stack: PromptStack) -> TextArtifact:
        for attempt in self.retrying():
            with attempt:
                self.before_run(prompt_stack)

                if self.stream:
                    tokens = []
                    completion_chunks = self.try_stream(prompt_stack)
                    for chunk in completion_chunks:
                        self.structure.publish_event(
                            CompletionChunkEvent(token=chunk.value)
                        )
                        tokens.append(chunk.value)
                    result = TextArtifact(value="".join(tokens).strip())
                else:
                    result = self.try_run(prompt_stack)
                    result.value = result.value.strip()

                self.after_run(result)

                return result
        else:
            raise Exception("prompt driver failed after all retry attempts")

    def default_prompt_stack_to_string_converter(
        self, prompt_stack: PromptStack
    ) -> str:
        prompt_lines = []

        for i in prompt_stack.inputs:
            if i.is_user():
                prompt_lines.append(f"User: {i.content}")
            elif i.is_assistant():
                prompt_lines.append(f"Assistant: {i.content}")
            else:
                prompt_lines.append(i.content)

        prompt_lines.append("Assistant:")

        return "\n\n".join(prompt_lines)

    @abstractmethod
    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:
        ...

    @abstractmethod
    def try_stream(self, prompt_stack: PromptStack) -> Iterator[TextArtifact]:
        ...
