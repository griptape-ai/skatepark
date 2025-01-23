from attrs import define, field

from griptape.events.base_chunk_event import BaseChunkEvent


@define
class AudioChunkEvent(BaseChunkEvent):
    """Stores a chunk of audio data.

    Attributes:
        data: Base64 encoded audio data.
    """

    data: str = field(kw_only=True, metadata={"serializable": True})

    def __str__(self) -> str:
        return self.data
