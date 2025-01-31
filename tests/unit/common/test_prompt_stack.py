from contextlib import nullcontext

import pytest
from pydantic import create_model
from schema import Schema

from griptape.artifacts import ActionArtifact, GenericArtifact, ImageArtifact, ListArtifact, TextArtifact
from griptape.artifacts.error_artifact import ErrorArtifact
from griptape.common import (
    ActionCallMessageContent,
    ActionResultMessageContent,
    GenericMessageContent,
    ImageMessageContent,
    PromptStack,
    TextMessageContent,
    ToolAction,
)


class TestPromptStack:
    @pytest.fixture()
    def prompt_stack(self):
        return PromptStack()

    def test_init(self):
        assert PromptStack()

    def test_add_message(self, prompt_stack):
        prompt_stack.add_message("foo", "role")
        prompt_stack.add_message(TextArtifact("foo"), "role")
        prompt_stack.add_message(ImageArtifact(b"foo", format="png", width=100, height=100), "role")
        prompt_stack.add_message(ListArtifact([TextArtifact("foo"), TextArtifact("bar")]), "role")
        prompt_stack.add_message(
            ListArtifact(
                [TextArtifact("foo"), ActionArtifact(ToolAction(tag="foo", name="bar", path="baz", input={}))]
            ),
            "role",
        )
        prompt_stack.add_message(
            ListArtifact(
                [
                    TextArtifact("foo"),
                    ActionArtifact(ToolAction(tag="foo", name="bar", path="baz", input={}, output=TextArtifact("qux"))),
                ]
            ),
            "role",
        )
        prompt_stack.add_message(
            GenericArtifact("foo"),
            "role",
        )

        prompt_stack.add_message(ErrorArtifact("foo"), "role")

        assert prompt_stack.messages[0].role == "role"
        assert isinstance(prompt_stack.messages[0].content[0], TextMessageContent)
        assert prompt_stack.messages[0].content[0].artifact.value == "foo"

        assert prompt_stack.messages[1].role == "role"
        assert isinstance(prompt_stack.messages[1].content[0], TextMessageContent)
        assert prompt_stack.messages[1].content[0].artifact.value == "foo"

        assert prompt_stack.messages[2].role == "role"
        assert isinstance(prompt_stack.messages[2].content[0], ImageMessageContent)
        assert prompt_stack.messages[2].content[0].artifact.value == b"foo"

        assert prompt_stack.messages[3].role == "role"
        assert isinstance(prompt_stack.messages[3].content[0], TextMessageContent)
        assert prompt_stack.messages[3].content[0].artifact.value == "foo"
        assert isinstance(prompt_stack.messages[3].content[1], TextMessageContent)
        assert prompt_stack.messages[3].content[1].artifact.value == "bar"

        assert prompt_stack.messages[4].role == "role"
        assert isinstance(prompt_stack.messages[4].content[0], TextMessageContent)
        assert prompt_stack.messages[4].content[0].artifact.value == "foo"
        assert isinstance(prompt_stack.messages[4].content[1], ActionCallMessageContent)
        assert prompt_stack.messages[4].content[1].artifact.value.to_dict() == {
            "tag": "foo",
            "name": "bar",
            "path": "baz",
            "input": {},
            "type": "ToolAction",
        }

        assert prompt_stack.messages[5].role == "role"
        assert isinstance(prompt_stack.messages[5].content[0], TextMessageContent)
        assert prompt_stack.messages[5].content[0].artifact.value == "foo"
        assert isinstance(prompt_stack.messages[5].content[1], ActionResultMessageContent)
        assert prompt_stack.messages[5].content[1].artifact.value == "qux"

        assert prompt_stack.messages[6].role == "role"
        assert isinstance(prompt_stack.messages[6].content[0], GenericMessageContent)
        assert prompt_stack.messages[6].content[0].artifact.value == "foo"

        assert prompt_stack.messages[7].role == "role"
        assert isinstance(prompt_stack.messages[7].content[0], TextMessageContent)

    def test_add_system_message(self, prompt_stack):
        prompt_stack.add_system_message("foo")

        assert prompt_stack.messages[0].role == "system"
        assert prompt_stack.messages[0].content[0].artifact.value == "foo"

    def test_add_user_message(self, prompt_stack):
        prompt_stack.add_user_message("foo")
        assert prompt_stack.messages[0].role == "user"
        assert prompt_stack.messages[0].content[0].artifact.value == "foo"

    def test_add_assistant_message(self, prompt_stack):
        prompt_stack.add_assistant_message("foo")

        assert prompt_stack.messages[0].role == "assistant"
        assert prompt_stack.messages[0].content[0].artifact.value == "foo"

    def test_from_artifact(self):
        prompt_stack = PromptStack.from_artifact(TextArtifact("foo"))

        assert prompt_stack.messages[0].role == "user"
        assert prompt_stack.messages[0].content[0].artifact.value == "foo"

    @pytest.mark.parametrize(
        ("output_schema", "expected_output", "expected_exception"),
        [
            (
                Schema({"foo": Schema({})}),
                {
                    "$id": "foo",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "foo": {
                            "additionalProperties": False,
                            "properties": {},
                            "required": [],
                            "type": "object",
                        },
                    },
                    "additionalProperties": False,
                    "required": ["foo"],
                },
                nullcontext(),
            ),
            (
                create_model("OutputSchema", foo=(create_model("NestedOutputSchema"), ...)),
                {
                    "$id": "foo",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "foo": {
                            "additionalProperties": False,
                            "properties": {},
                            "title": "NestedOutputSchema",
                            "type": "object",
                        },
                    },
                    "required": ["foo"],
                    "title": "OutputSchema",
                    "type": "object",
                },
                nullcontext(),
            ),
            (None, None, pytest.raises(ValueError, match="Output schema is not set")),
            ("str", None, pytest.raises(ValueError, match="Unsupported output schema type")),
        ],
    )
    def test_to_output_json_schema(self, prompt_stack, output_schema, expected_output, expected_exception):
        prompt_stack.output_schema = output_schema

        with expected_exception:
            output_json_schema = prompt_stack.to_output_json_schema("foo")
            assert output_json_schema == expected_output
