from typing import Optional
from attr import define, field
from griptape.artifacts import TextArtifact


@define(kw_only=True)
class RagContext:
    initial_query: str = field()
    expanded_queries: list[str] = field(factory=list)
    before_query: list[str] = field(factory=list)
    after_query: list[str] = field(factory=list)
    text_chunks: list[TextArtifact] = field(factory=list)
    output: Optional[str] = field(default=None)

    @property
    def all_queries(self) -> list[str]:
        return [self.initial_query] + self.expanded_queries
