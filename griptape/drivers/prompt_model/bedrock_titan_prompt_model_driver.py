import json
from attr import define, field, Factory
from griptape.artifacts import TextArtifact
from griptape.utils import PromptStack
from griptape.drivers import BasePromptModelDriver
from griptape.tokenizers import AmazonBedrockTokenizer


@define
class BedrockTitanPromptModelDriver(BasePromptModelDriver):
    model: str = field(default="amazon.titan-tg1-large", kw_only=True)
    tokenizer: AmazonBedrockTokenizer = field(
        default=Factory(
            lambda self: AmazonBedrockTokenizer(model=self.model),
            takes_self=True
        ),
        kw_only=True,
    )
    top_p: float = field(default=0.9, kw_only=True)

    def prompt_stack_to_model_input(self, prompt_stack: PromptStack) -> dict:
        prompt_lines = []

        for i in prompt_stack.inputs:
            if i.is_assistant():
                prompt_lines.append(f"Bot: {i.content}")
            else:
                prompt_lines.append(f"User: {i.content}")

        prompt_lines.append("\nBot:")

        prompt = "\n".join(prompt_lines)
        return {"inputText": prompt}

    def prompt_stack_to_model_params(self, prompt_stack: PromptStack) -> dict:
        prompt = self.prompt_driver.prompt_stack_to_string(prompt_stack)

        return {
            "textGenerationConfig": {
                "maxTokenCount": self.prompt_driver.max_output_tokens(prompt),
                "stopSequences": self.tokenizer.stop_sequences,
                "temperature": self.prompt_driver.temperature,
                "topP": self.top_p,
            }
        }

    def process_output(self, response_body: str) -> TextArtifact:
        response_body = json.loads(response_body)

        return TextArtifact(response_body["results"][0]["outputText"])
