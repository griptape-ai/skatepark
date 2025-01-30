from griptape.events.audio_chunk_event import AudioChunkEvent


class TestAudioChunkEvent:
    def test___str__(self):
        assert str(AudioChunkEvent(data="audio-data")) == "audio-data"
