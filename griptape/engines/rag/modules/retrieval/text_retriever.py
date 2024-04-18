from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Optional
from attr import define, field
from griptape import utils
from griptape.artifacts import TextArtifact
from griptape.engines.rag import RagContext
from griptape.engines.rag.modules.retrieval import BaseRetrievalModule

if TYPE_CHECKING:
    from griptape.drivers import BaseVectorStoreDriver


@define
class TextRetriever(BaseRetrievalModule):
    namespace: Optional[str] = field(default=None, kw_only=True)
    top_n: Optional[int] = field(default=None, kw_only=True)
    vector_store_driver: BaseVectorStoreDriver = field(kw_only=True)

    def run(self, context: RagContext) -> list[TextArtifact]:
        results = utils.execute_futures_list(
            [self.futures_executor.submit(
                self.vector_store_driver.query, q, self.top_n, self.namespace, False
            ) for q in context.all_queries]
        )

        return [
            artifact
            for artifact in [r.to_artifact() for r in list(itertools.chain.from_iterable(results))]
            if isinstance(artifact, TextArtifact)
        ]
