import pytest

from griptape.configs.drivers import AzureOpenAiDriversConfig


class TestAzureOpenAiDriversConfig:
    @pytest.fixture(autouse=True)
    def mock_openai(self, mocker):
        return mocker.patch("openai.AzureOpenAI")

    @pytest.fixture()
    def config(self):
        return AzureOpenAiDriversConfig(
            azure_endpoint="http://localhost:8080",
            azure_ad_token="test-token",
            azure_ad_token_provider=lambda: "test-provider",
        )

    def test_to_dict(self, config):
        assert config.to_dict() == {
            "type": "AzureOpenAiDriversConfig",
            "azure_endpoint": "http://localhost:8080",
            "prompt_driver": {
                "type": "AzureOpenAiChatPromptDriver",
                "base_url": None,
                "model": "gpt-4o",
                "azure_deployment": "gpt-4o",
                "azure_endpoint": "http://localhost:8080",
                "audio": {"format": "pcm16", "voice": "alloy"},
                "modalities": ["text"],
                "api_version": "2023-05-15",
                "organization": None,
                "parallel_tool_calls": True,
                "response_format": None,
                "seed": None,
                "temperature": 0.1,
                "max_tokens": None,
                "stream": False,
                "user": "",
                "use_native_tools": True,
                "structured_output_strategy": "native",
                "extra_params": {},
            },
            "conversation_memory_driver": {
                "type": "LocalConversationMemoryDriver",
                "persist_file": None,
            },
            "embedding_driver": {
                "base_url": None,
                "model": "text-embedding-3-small",
                "api_version": "2023-05-15",
                "azure_deployment": "text-embedding-3-small",
                "azure_endpoint": "http://localhost:8080",
                "organization": None,
                "type": "AzureOpenAiEmbeddingDriver",
            },
            "image_generation_driver": {
                "api_version": "2024-02-01",
                "base_url": None,
                "image_size": "512x512",
                "model": "dall-e-2",
                "azure_deployment": "dall-e-2",
                "azure_endpoint": "http://localhost:8080",
                "organization": None,
                "quality": "standard",
                "response_format": "b64_json",
                "style": None,
                "type": "AzureOpenAiImageGenerationDriver",
            },
            "vector_store_driver": {
                "embedding_driver": {
                    "base_url": None,
                    "model": "text-embedding-3-small",
                    "api_version": "2023-05-15",
                    "azure_deployment": "text-embedding-3-small",
                    "azure_endpoint": "http://localhost:8080",
                    "organization": None,
                    "type": "AzureOpenAiEmbeddingDriver",
                },
                "type": "LocalVectorStoreDriver",
            },
            "text_to_speech_driver": {
                "base_url": None,
                "format": "mp3",
                "model": "tts",
                "api_version": "2024-07-01-preview",
                "azure_deployment": "tts",
                "azure_endpoint": "http://localhost:8080",
                "organization": None,
                "type": "AzureOpenAiTextToSpeechDriver",
                "voice": "alloy",
            },
            "audio_transcription_driver": {"type": "DummyAudioTranscriptionDriver"},
            "ruleset_driver": {
                "type": "LocalRulesetDriver",
                "raise_not_found": True,
                "persist_dir": None,
            },
        }
