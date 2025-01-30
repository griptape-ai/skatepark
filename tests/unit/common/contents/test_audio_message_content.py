import base64
import time

from griptape.artifacts.audio_artifact import AudioArtifact
from griptape.common import AudioDeltaMessageContent, AudioMessageContent


class TestAudioMessageContent:
    def test_init(self):
        assert AudioMessageContent(AudioArtifact(b"foo", format="wav")).artifact.value == b"foo"

    def test_from_deltas(self):
        content = AudioMessageContent.from_deltas(
            [
                AudioDeltaMessageContent(
                    id="foo-id",
                ),
                AudioDeltaMessageContent(
                    data=base64.b64encode(b"foo").decode(),
                ),
                AudioDeltaMessageContent(
                    data=base64.b64encode(b"bar").decode(),
                ),
                AudioDeltaMessageContent(expires_at=int(time.time())),
                AudioDeltaMessageContent(transcript="foobar"),
            ]
        )

        assert content.artifact.value == b"foobar"
        assert content.artifact.meta == {
            "audio_id": "foo-id",
            "expires_at": int(time.time()),
            "transcript": "foobar",
        }
