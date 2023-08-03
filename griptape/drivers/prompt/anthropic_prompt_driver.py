import anthropic
from attr import define, field, Factory
from griptape.artifacts import TextArtifact
from griptape.core import PromptStack
from griptape.drivers import BasePromptDriver
from griptape.tokenizers import AnthropicTokenizer


@define
class AnthropicPromptDriver(BasePromptDriver):
    prompt_prefix: str = field(default=anthropic.HUMAN_PROMPT, kw_only=True)
    prompt_suffix: str = field(default=anthropic.AI_PROMPT, kw_only=True)
    api_key: str = field(kw_only=True)
    model: str = field(default=AnthropicTokenizer.DEFAULT_MODEL, kw_only=True)
    tokenizer: AnthropicTokenizer = field(
        default=Factory(
            lambda self: AnthropicTokenizer(model=self.model), takes_self=True
        ),
        kw_only=True,
    )

    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:
        return self.__run_completion(prompt_stack)

    def prompt_stack_to_prompt(self, prompt_stack: PromptStack) -> str:
        prompt_lines = []

        for i in prompt_stack.inputs:
            if i.is_system():
                prompt_lines.append(f"\n\nHuman: {i.content}")
            elif i.is_user():
                prompt_lines.append(f"Human: {i.content}")
            elif i.is_assistant():
                prompt_lines.append(f"Assistant: {i.content}")

        prompt_lines.append("Assistant:")

        return "\n\n".join(prompt_lines)

    def __run_completion(self, prompt_stack: PromptStack) -> TextArtifact:
        client = anthropic.Client(self.api_key)
        prompt = self.prompt_stack_to_prompt(prompt_stack)

        # Anthropic requires specific prompt formatting: https://console.anthropic.com/docs/api
        response = client.completion(
            prompt=prompt,
            stop_sequences=self.tokenizer.stop_sequences,
            model=self.model,
            max_tokens_to_sample=self.tokenizer.tokens_left(prompt),
            temperature=self.temperature,
        )

        return TextArtifact(value=response["completion"])
