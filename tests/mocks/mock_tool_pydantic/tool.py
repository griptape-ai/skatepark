from attrs import define
from pydantic import BaseModel, Field, create_model

from griptape.artifacts import BaseArtifact, TextArtifact
from griptape.tools import BaseTool
from griptape.utils.decorators import activity


class TestModel(BaseModel):
    foo: str = Field(description="Test input")


@define
class MockToolPydantic(BaseTool):
    @activity(config={"description": "test description", "schema": TestModel})
    def test(self) -> BaseArtifact:
        return TextArtifact("ack")

    @activity(
        config={
            "description": "test description",
            "schema": lambda _self: _self.build_custom_schema(),
        }
    )
    def test_callable_schema(self) -> TextArtifact:
        return TextArtifact("ack")

    def build_custom_schema(self) -> type[BaseModel]:
        return create_model("TestModel", foo=(str, Field(description="Test input")))
