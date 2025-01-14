from __future__ import annotations

from attrs import Factory, define, field
from schema import Literal, Or, Schema

from griptape.artifacts import BaseArtifact, ErrorArtifact, ListArtifact, TextArtifact
from griptape.engines import PromptSummaryEngine
from griptape.mixins.rule_mixin import RuleMixin
from griptape.tools import BaseTool
from griptape.utils.decorators import activity


@define(kw_only=True)
class PromptSummaryTool(BaseTool, RuleMixin):
    """Tool for using a Prompt Summary Engine.

    Attributes:
        prompt_summary_engine: `PromptSummaryEngine`.
    """

    prompt_summary_engine: PromptSummaryEngine = field(kw_only=True, default=Factory(lambda: PromptSummaryEngine()))

    @activity(
        config={
            "description": "Can be used to summarize text content.",
            "schema": Schema(
                {
                    Literal("summary"): Or(
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
    def summarize(self, params: dict) -> BaseArtifact:
        summary = params["values"]["summary"]

        if isinstance(summary, str):
            artifacts = ListArtifact([TextArtifact(summary)])
        else:
            memory = self.find_input_memory(summary["memory_name"])
            artifact_namespace = summary["artifact_namespace"]

            if memory is not None:
                artifacts = memory.load_artifacts(artifact_namespace)
            else:
                return ErrorArtifact("memory not found")

        return self.prompt_summary_engine.summarize_artifacts(artifacts, rulesets=self.all_rulesets)
