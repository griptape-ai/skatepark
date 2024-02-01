from __future__ import annotations
from attr import define, field, Factory
from .simple_tokenizer import SimpleTokenizer


@define(frozen=True)
class BedrockLlamaTokenizer(SimpleTokenizer):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-prepare.html#model-customization-prepare-finetuning
    DEFAULT_CHARACTERS_PER_TOKEN = 6
    MODEL_PREFIXES_TO_MAX_TOKENS = {"meta.llama2": 4096}

    model: str = field(kw_only=True)
    characters_per_token: int = field(default=DEFAULT_CHARACTERS_PER_TOKEN, kw_only=True)
    max_tokens: int = field(kw_only=True, default=Factory(lambda self: self.default_max_tokens(), takes_self=True))
    stop_sequences: list[str] = field(factory=list, kw_only=True)

    def default_max_tokens(self) -> int:
        tokens = next(v for k, v in self.MODEL_PREFIXES_TO_MAX_TOKENS.items() if self.model.startswith(k))

        return tokens
