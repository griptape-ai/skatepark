from attrs import define, field

from griptape.common import BasePromptStackContent
from griptape.artifacts import TextArtifact


@define
class TextPromptStackContent(BasePromptStackContent):
    artifact: TextArtifact = field(metadata={"serializable": True})