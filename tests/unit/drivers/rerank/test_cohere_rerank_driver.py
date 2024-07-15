import pytest
from cohere import RerankResponseResultsItem, RerankResponseResultsItemDocument

from griptape.artifacts import TextArtifact
from griptape.drivers import CohereRerankDriver


class TestCohereRerankDriver:
    @pytest.fixture
    def mock_client(self, mocker):
        mock_client = mocker.patch("cohere.Client").return_value
        mock_client.rerank.return_value.results = [
            RerankResponseResultsItem(
                index=1, relevance_score=1.0, document=RerankResponseResultsItemDocument(text="foo")
            ),
            RerankResponseResultsItem(
                index=2, relevance_score=0.5, document=RerankResponseResultsItemDocument(text="bar")
            ),
        ]

        return mock_client

    def test_run(self, mock_client):
        driver = CohereRerankDriver(api_key="api-key")
        result = driver.run("hello", artifacts=[TextArtifact("foo"), TextArtifact("bar")])

        assert len(result) == 2
