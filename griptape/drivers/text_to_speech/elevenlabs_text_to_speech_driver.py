from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any

from attr import define, field, Factory

from griptape.artifacts.audio_artifact import AudioArtifact
from griptape.drivers import BaseTextToSpeechDriver

if TYPE_CHECKING:
    from elevenlabs.client import ElevenLabs


@define
class ElevenLabsTextToSpeechDriver(BaseTextToSpeechDriver):
    api_key: str = field(kw_only=True, metadata={"serializable": True})
    client: Any = field(
        default=Factory(lambda self: ElevenLabs(api_key=self.api_key), takes_self=True),
        kw_only=True,
        metadata={"serializable": True},
    )
    voice: str = field(kw_only=True, metadata={"serializable": True})
    output_format: str = field(default="mp3_44100_128", kw_only=True, metadata={"serializable": True})

    def try_text_to_audio(self, prompts: list[str], negative_prompts: Optional[list[str]] = None) -> AudioArtifact:
        audio = self.client.generate(
            text=". ".join(prompts), voice=self.voice, model=self.model, output_format=self.output_format
        )

        content = b""
        for chunk in audio:
            content += chunk

        # All ElevenLabs audio format strings have the following structure:
        # {format}_{sample_rate}_{bitrate}
        artifact_format = self.output_format.split("_")[0]

        return AudioArtifact(value=content, format=artifact_format)