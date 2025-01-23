from __future__ import annotations

from typing import Optional

from attrs import define, field

from griptape.common import BaseDeltaMessageContent


@define(kw_only=True)
class AudioDeltaMessageContent(BaseDeltaMessageContent):
    """A delta message content for audio data.

    Attributes:
        id: The ID of the audio data.
        data: Base64 encoded audio data.
        transcript: The transcript of the audio data.
        expires_at: The Unix timestamp (in seconds) for when this audio data will no longer be accessible.
    """

    id: Optional[str] = field(default=None, metadata={"serializable": True})
    data: Optional[str] = field(default=None, metadata={"serializable": True})
    transcript: Optional[str] = field(default=None, metadata={"serializable": True})
    expires_at: Optional[int] = field(default=None, metadata={"serializable": True})
