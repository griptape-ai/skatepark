from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
from attr import define, field, Factory

if TYPE_CHECKING:
    from griptape.structures import Structure
from griptape.utils.stream import Stream


@define(frozen=True)
class Chat:
    structure: Structure = field()
    exit_keywords: list[str] = field(default=["exit"], kw_only=True)
    exiting_text: str = field(default="exiting...", kw_only=True)
    processing_text: str = field(default="processing...", kw_only=True)
    intro_text: Optional[str] = field(default=None, kw_only=True)
    prompt_prefix: str = field(default="Q: ", kw_only=True)
    response_prefix: str = field(default="A: ", kw_only=True)
    streaming_output_fn: Callable[[str], None] = field(default=lambda x: print(x, end=""), kw_only=True)
    output_fn: Callable[[str], None] = field(default=Factory(lambda: print), kw_only=True)

    def start(self) -> None:
        if self.intro_text:
            self.output_fn(self.intro_text)
        while True:
            question = input(self.prompt_prefix)

            if question.lower() in self.exit_keywords:
                self.output_fn(self.exiting_text)
                break
            else:
                self.output_fn(self.processing_text)

            if self.structure.stream:
                stream = Stream(self.structure).run(question)
                first_chunk = next(stream)
                self.streaming_output_fn(self.response_prefix + first_chunk.value)
                for chunk in stream:
                    self.streaming_output_fn(chunk.value)
                self.streaming_output_fn("\n")
            else:
                self.output_fn(f"{self.response_prefix}{self.structure.run(question).output.to_text()}")
