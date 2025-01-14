from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import Factory, define, field
from schema import Literal, Or, Schema

from griptape.artifacts import ErrorArtifact, ListArtifact, TextArtifact
from griptape.configs import Defaults
from griptape.engines.rag import RagEngine
from griptape.engines.rag.modules import (
    PromptResponseRagModule,
)
from griptape.engines.rag.rag_context import RagContext
from griptape.engines.rag.stages import ResponseRagStage
from griptape.mixins.rule_mixin import RuleMixin
from griptape.tools.base_tool import BaseTool
from griptape.utils.decorators import activity

if TYPE_CHECKING:
    from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver


@define(kw_only=True)
class QueryTool(BaseTool, RuleMixin):
    """Tool for performing a query against data."""

    prompt_driver: BasePromptDriver = field(default=Factory(lambda: Defaults.drivers_config.prompt_driver))

    _rag_engine: RagEngine = field(
        default=Factory(
            lambda self: RagEngine(
                response_stage=ResponseRagStage(
                    response_modules=[
                        PromptResponseRagModule(prompt_driver=self.prompt_driver, rulesets=self.all_rulesets)
                    ],
                ),
            ),
            takes_self=True,
        ),
        alias="_rag_engine",
    )

    @activity(
        config={
            "description": "Can be used to search through textual content.",
            "schema": Schema(
                {
                    Literal("query", description="A natural language search query"): str,
                    Literal("content"): Or(
                        str,
                        Schema(
                            {
                                "memory_name": str,
                                "artifact_namespace": str,
                            }
                        ),
                    ),
                }
            ),
        },
    )
    def query(self, params: dict) -> ListArtifact | ErrorArtifact:
        query = params["values"]["query"]
        content = params["values"]["content"]

        if isinstance(content, str):
            text_artifacts = [TextArtifact(content)]
        else:
            memory = self.find_input_memory(content["memory_name"])
            artifact_namespace = content["artifact_namespace"]

            if memory is not None:
                artifacts = memory.load_artifacts(artifact_namespace)
            else:
                return ErrorArtifact("memory not found")

            text_artifacts = [artifact for artifact in artifacts if isinstance(artifact, TextArtifact)]

        outputs = self._rag_engine.process(RagContext(query=query, text_chunks=text_artifacts)).outputs

        if len(outputs) > 0:
            return ListArtifact(outputs)
        else:
            return ErrorArtifact("query output is empty")
