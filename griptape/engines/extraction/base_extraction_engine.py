from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
from attrs import define, field, Factory
from griptape.chunkers import BaseChunker, TextChunker

if TYPE_CHECKING:
    from griptape.artifacts import ListArtifact, ErrorArtifact
    from griptape.rules import Ruleset
    from griptape.drivers import BasePromptDriver


@define
class BaseExtractionEngine(ABC):
    max_token_multiplier: float = field(default=0.5, kw_only=True)
    chunk_joiner: str = field(default="\n\n", kw_only=True)
    prompt_driver: BasePromptDriver = field(kw_only=True)
    chunker: BaseChunker = field(
        default=Factory(
            lambda self: TextChunker(tokenizer=self.prompt_driver.tokenizer, max_tokens=self.max_chunker_tokens),
            takes_self=True,
        ),
        kw_only=True,
    )

    @max_token_multiplier.validator  # pyright: ignore[reportAttributeAccessIssue]
    def validate_max_token_multiplier(self, _, max_token_multiplier: int) -> None:
        if max_token_multiplier > 1:
            raise ValueError("has to be less than or equal to 1")
        elif max_token_multiplier <= 0:
            raise ValueError("has to be greater than 0")

    @property
    def max_chunker_tokens(self) -> int:
        return round(self.prompt_driver.tokenizer.max_input_tokens * self.max_token_multiplier)

    @property
    def min_response_tokens(self) -> int:
        return round(
            self.prompt_driver.tokenizer.max_input_tokens
            - self.prompt_driver.tokenizer.max_input_tokens * self.max_token_multiplier
        )

    @abstractmethod
    def extract(
        self, text: str | ListArtifact, *, rulesets: Optional[list[Ruleset]] = None, **kwargs
    ) -> ListArtifact | ErrorArtifact: ...
