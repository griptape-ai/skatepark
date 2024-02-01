from attr import define
from griptape.tokenizers import AnthropicTokenizer


@define(frozen=True)
class BedrockClaudeTokenizer(AnthropicTokenizer):
    MODEL_PREFIXES_TO_MAX_TOKENS = {"anthropic.claude-v2:1": 200000, "anthropic.claude": 100000}
