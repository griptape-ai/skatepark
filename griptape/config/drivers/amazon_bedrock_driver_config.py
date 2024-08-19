from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import Factory, define, field

from griptape.config.drivers import DriverConfig
from griptape.drivers import (
    AmazonBedrockImageGenerationDriver,
    AmazonBedrockImageQueryDriver,
    AmazonBedrockPromptDriver,
    AmazonBedrockTitanEmbeddingDriver,
    BedrockClaudeImageQueryModelDriver,
    BedrockTitanImageGenerationModelDriver,
    LocalVectorStoreDriver,
)
from griptape.utils import import_optional_dependency
from griptape.utils.decorators import lazy_property

if TYPE_CHECKING:
    import boto3


@define
class AmazonBedrockDriverConfig(DriverConfig):
    session: boto3.Session = field(
        default=Factory(lambda: import_optional_dependency("boto3").Session()),
        kw_only=True,
        metadata={"serializable": False},
    )

    @lazy_property()
    def prompt(self) -> AmazonBedrockPromptDriver:
        return AmazonBedrockPromptDriver(session=self.session, model="anthropic.claude-3-5-sonnet-20240620-v1:0")

    @lazy_property()
    def embedding(self) -> AmazonBedrockTitanEmbeddingDriver:
        return AmazonBedrockTitanEmbeddingDriver(session=self.session, model="amazon.titan-embed-text-v1")

    @lazy_property()
    def image_generation(self) -> AmazonBedrockImageGenerationDriver:
        return AmazonBedrockImageGenerationDriver(
            session=self.session,
            model="amazon.titan-image-generator-v1",
            image_generation_model_driver=BedrockTitanImageGenerationModelDriver(),
        )

    @lazy_property()
    def image_query(self) -> AmazonBedrockImageQueryDriver:
        return AmazonBedrockImageQueryDriver(
            session=self.session,
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            image_query_model_driver=BedrockClaudeImageQueryModelDriver(),
        )

    @lazy_property()
    def vector_store(self) -> LocalVectorStoreDriver:
        return LocalVectorStoreDriver(
            embedding_driver=AmazonBedrockTitanEmbeddingDriver(session=self.session, model="amazon.titan-embed-text-v1")
        )
