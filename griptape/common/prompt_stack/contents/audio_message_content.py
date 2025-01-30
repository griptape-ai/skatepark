from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from attrs import define, field

from griptape.artifacts import AudioArtifact
from griptape.common import (
    AudioDeltaMessageContent,
    BaseDeltaMessageContent,
    BaseMessageContent,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@define
class AudioMessageContent(BaseMessageContent):
    artifact: AudioArtifact = field(metadata={"serializable": True})

    @classmethod
    def from_deltas(cls, deltas: Sequence[BaseDeltaMessageContent]) -> AudioMessageContent:
        audio_deltas = [delta for delta in deltas if isinstance(delta, AudioDeltaMessageContent)]
        audio_data = [delta.data for delta in audio_deltas if delta.data is not None]
        transcript_data = [delta.transcript for delta in audio_deltas if delta.transcript is not None]
        expires_at = next(delta.expires_at for delta in audio_deltas if delta.expires_at is not None)
        audio_id = next(delta.id for delta in audio_deltas if delta.id is not None)

        audio_transcript = "".join(data for data in transcript_data)

        artifact = AudioArtifact(
            value=b"".join(base64.b64decode(data) for data in audio_data),
            format="wav",
            meta={
                "audio_id": audio_id,
                "expires_at": expires_at,
                "transcript": audio_transcript,
            },
        )

        return cls(artifact=artifact)
