from __future__ import annotations

from typing import TYPE_CHECKING, Union

from attrs import define, field

from griptape.artifacts import BaseArtifact, JsonArtifact
from griptape.tools import BaseTool
from griptape.utils.decorators import activity

if TYPE_CHECKING:
    from pydantic import BaseModel
    from schema import Schema


@define
class StructuredOutputTool(BaseTool):
    output_schema: Union[Schema, type[BaseModel]] = field(kw_only=True)

    @activity(
        config={
            "description": "Used to provide the final response which ends this conversation.",
            "schema": lambda _self: _self.output_schema,
        }
    )
    def provide_output(self, params: dict) -> BaseArtifact:
        return JsonArtifact(params["values"])
