from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from attrs import Factory, define, field

from griptape.artifacts.text_artifact import TextArtifact
from griptape.configs import Defaults
from griptape.engines.rag.modules import BaseResponseRagModule
from griptape.mixins.rule_mixin import RuleMixin
from griptape.utils import J2

if TYPE_CHECKING:
    from griptape.artifacts import BaseArtifact
    from griptape.drivers.prompt import BasePromptDriver
    from griptape.engines.rag import RagContext


@define(kw_only=True)
class PromptResponseRagModule(BaseResponseRagModule, RuleMixin):
    prompt_driver: BasePromptDriver = field(default=Factory(lambda: Defaults.drivers_config.prompt_driver))
    answer_token_offset: int = field(default=400)
    metadata: Optional[str] = field(default=None)
    generate_system_template: Callable[[RagContext, list[TextArtifact]], str] = field(
        default=Factory(lambda self: self.default_generate_system_template, takes_self=True),
    )

    def run(self, context: RagContext) -> BaseArtifact:
        query = context.query
        tokenizer = self.prompt_driver.tokenizer
        included_chunks = []
        system_prompt = self.generate_system_template(context, included_chunks)

        for artifact in context.text_chunks:
            included_chunks.append(artifact)

            system_prompt = self.generate_system_template(context, included_chunks)
            message_token_count = self.prompt_driver.tokenizer.count_tokens(
                self.prompt_driver.prompt_stack_to_string(self.generate_prompt_stack(system_prompt, query)),
            )

            if message_token_count + self.answer_token_offset >= tokenizer.max_input_tokens:
                included_chunks.pop()

                system_prompt = self.generate_system_template(context, included_chunks)

                break

        output = self.prompt_driver.run(self.generate_prompt_stack(system_prompt, query)).to_artifact()

        if isinstance(output, TextArtifact):
            return output
        else:
            raise ValueError("Prompt driver did not return a TextArtifact")

    def default_generate_system_template(self, context: RagContext, artifacts: list[TextArtifact]) -> str:
        params: dict[str, Any] = {"text_chunks": [c.to_text() for c in artifacts]}

        if len(self.all_rulesets) > 0:
            params["rulesets"] = J2("rulesets/rulesets.j2").render(rulesets=self.all_rulesets)

        if self.metadata is not None:
            params["metadata"] = J2("engines/rag/modules/response/metadata/system.j2").render(metadata=self.metadata)

        return J2("engines/rag/modules/response/prompt/system.j2").render(**params)
