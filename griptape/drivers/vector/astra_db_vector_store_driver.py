from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Optional

from attrs import define, field

from griptape.drivers import BaseVectorStoreDriver
from griptape.utils import import_optional_dependency

if TYPE_CHECKING:
    from astrapy import Collection
    from astrapy.authentication import TokenProvider

GRIPTAPE_VERSION: Optional[str]
try:
    from importlib import metadata

    GRIPTAPE_VERSION = metadata.version("griptape")
except Exception:
    GRIPTAPE_VERSION = None

logging.basicConfig(level=logging.WARNING)


COLLECTION_INDEXING = {"deny": ["meta.artifact"]}


@define
class AstraDBVectorStoreDriver(BaseVectorStoreDriver):
    """A Vector Store Driver for Astra DB.

    Attributes:
        embedding_driver: a `griptape.drivers.BaseEmbeddingDriver` for embedding computations within the store
        api_endpoint: the "API Endpoint" for the Astra DB instance.
        token: a Database Token ("AstraCS:...") secret to access Astra DB. An instance of `astrapy.authentication.TokenProvider` is also accepted.
        collection_name: the name of the collection on Astra DB.
        environment: the environment ("prod", "hcd", ...) hosting the target Data API.
            It can be omitted for production Astra DB targets. See `astrapy.constants.Environment` for allowed values.
        dimension: the number of components for embedding vectors. If not provided, it will be guessed from the embedding driver.
        metric: the similarity metric to use, one of "dot_product", "euclidean" or "cosine".
            If omitted, the server default ("cosine") will be used. See also values of `astrapy.constants.VectorMetric`.
            If the vectors are normalized to unit norm, choosing "dot_product" over cosine yields up to 2x speedup in searches.
        astra_db_namespace: optional specification of the namespace (in the Astra database) for the data.
            *Note*: not to be confused with the "namespace" mentioned elsewhere, which is a grouping within this vector store.
    """

    api_endpoint: str = field(kw_only=True, metadata={"serializable": True})
    token: Optional[str | TokenProvider] = field(kw_only=True, default=None, metadata={"serializable": False})
    collection_name: str = field(kw_only=True, metadata={"serializable": True})
    environment: Optional[str] = field(kw_only=True, default=None, metadata={"serializable": True})
    dimension: Optional[int] = field(kw_only=True, default=None, metadata={"serializable": True})
    metric: Optional[str] = field(kw_only=True, default=None, metadata={"serializable": True})
    astra_db_namespace: Optional[str] = field(default=None, kw_only=True, metadata={"serializable": True})

    collection: Collection = field(init=False)

    def __attrs_post_init__(self) -> None:
        astrapy = import_optional_dependency("astrapy")
        if not self.dimension:
            # auto-compute dimension from the embedding
            self.dimension = len(self.embedding_driver.embed_string("This is a sample text."))
        self.collection = (
            astrapy.DataAPIClient(
                caller_name="griptape",
                caller_version=GRIPTAPE_VERSION,
                environment=self.environment,
            )
            .get_database(
                self.api_endpoint,
                token=self.token,
                namespace=self.astra_db_namespace,
            )
            .create_collection(
                name=self.collection_name,
                dimension=self.dimension,
                metric=self.metric,
                indexing=COLLECTION_INDEXING,
                check_exists=False,
            )
        )

    def delete_vector(self, vector_id: str) -> None:
        """Delete a vector from Astra DB store.

        The method succeeds regardless of whether a vector with the provided ID
        was actually stored or not in the first place.

        Args:
            vector_id: ID of the vector to delete.
        """
        self.collection.delete_one({"_id": vector_id})

    def upsert_vector(
        self,
        vector: list[float],
        *,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        """Write a vector to the Astra DB store.

        In case the provided ID exists already, an overwrite will take place.

        Args:
            vector: the vector to be upserted.
            vector_id: the ID for the vector to store. If omitted, a server-provided new ID will be employed.
            namespace: a namespace (a grouping within the vector store) to assign the vector to.
            meta: a metadata dictionary associated to the vector.
            kwargs: additional keyword arguments. Currently none is used: if they are passed, they will be ignored with a warning.

        Returns:
            the ID of the written vector (str).
        """
        if kwargs:
            warnings.warn(
                "Unhandled keyword argument(s) provided to AstraDBVectorStore.upsert_vector: "
                f"'{','.join(sorted(kwargs.keys()))}'. These will be ignored.",
                stacklevel=2,
            )
        document = {
            k: v
            for k, v in {"$vector": vector, "_id": vector_id, "namespace": namespace, "meta": meta}.items()
            if v is not None
        }
        if vector_id is not None:
            self.collection.find_one_and_replace({"_id": vector_id}, document, upsert=True)
            return vector_id
        else:
            insert_result = self.collection.insert_one(document)
            return insert_result.inserted_id

    def load_entry(self, vector_id: str, *, namespace: Optional[str] = None) -> Optional[BaseVectorStoreDriver.Entry]:
        """Load a single vector entry from the Astra DB store given its ID.

        Args:
            vector_id: the ID of the required vector.
            namespace: a namespace, within the vector store, to constrain the search.

        Returns:
            The vector entry (a `BaseVectorStoreDriver.Entry`) if found, otherwise None.
        """
        find_filter = {k: v for k, v in {"_id": vector_id, "namespace": namespace}.items() if v is not None}
        match = self.collection.find_one(filter=find_filter, projection={"*": 1})
        if match:
            return BaseVectorStoreDriver.Entry(
                id=match["_id"], vector=match.get("$vector"), meta=match.get("meta"), namespace=match.get("namespace")
            )
        else:
            return None

    def load_entries(self, *, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        """Load entries from the Astra DB store.

        Args:
            namespace: a namespace, within the vector store, to constrain the search.

        Returns:
            A list of vector (`BaseVectorStoreDriver.Entry`) entries.
        """
        find_filter: dict[str, str] = {} if namespace is None else {"namespace": namespace}
        return [
            BaseVectorStoreDriver.Entry(
                id=match["_id"], vector=match.get("$vector"), meta=match.get("meta"), namespace=match.get("namespace")
            )
            for match in self.collection.find(filter=find_filter, projection={"*": 1})
        ]

    def query(
        self,
        query: str,
        *,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        **kwargs: Any,
    ) -> list[BaseVectorStoreDriver.Entry]:
        """Run a similarity search on the Astra DB store, based on a query string.

        Args:
            query: the query string.
            count: the maximum number of results to return. If omitted, defaults will apply.
            namespace: the namespace to filter results by.
            include_vectors: whether to include vector data in the results.
            kwargs: additional keyword arguments. Currently only the free-form dict `filter`
                is recognized (and goes straight to the Data API query);
                others will generate a warning and be ignored.

        Returns:
            A list of vector (`BaseVectorStoreDriver.Entry`) entries,
            with their `score` attribute set to the vector similarity to the query.
        """
        query_filter: Optional[dict[str, Any]] = kwargs.pop("filter", None)
        if kwargs:
            warnings.warn(
                "Unhandled keyword argument(s) provided to AstraDBVectorStore.query: "
                f"'{','.join(sorted(kwargs.keys()))}'. These will be ignored.",
                stacklevel=2,
            )
        find_filter_ns: dict[str, Any] = {} if namespace is None else {"namespace": namespace}
        find_filter = {**(query_filter or {}), **find_filter_ns}
        find_projection: Optional[dict[str, int]] = {"*": 1} if include_vectors else None
        vector = self.embedding_driver.embed_string(query)
        ann_limit = count or BaseVectorStoreDriver.DEFAULT_QUERY_COUNT
        matches = self.collection.find(
            filter=find_filter,
            sort={"$vector": vector},
            limit=ann_limit,
            projection=find_projection,
            include_similarity=True,
        )
        return [
            BaseVectorStoreDriver.Entry(
                id=match["_id"],
                vector=match.get("$vector"),
                score=match["$similarity"],
                meta=match.get("meta"),
                namespace=match.get("namespace"),
            )
            for match in matches
        ]
