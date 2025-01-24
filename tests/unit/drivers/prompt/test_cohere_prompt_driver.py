import json
from unittest.mock import Mock

import pytest
from schema import Schema

from griptape.artifacts.action_artifact import ActionArtifact
from griptape.artifacts.list_artifact import ListArtifact
from griptape.artifacts.text_artifact import TextArtifact
from griptape.common import ActionCallDeltaMessageContent, PromptStack, TextDeltaMessageContent, ToolAction
from griptape.drivers.prompt.cohere import CoherePromptDriver
from tests.mocks.mock_tool.tool import MockTool


class TestCoherePromptDriver:
    COHERE_STRUCTURED_OUTPUT_SCHEMA = {
        "$id": "Output Format",
        "$schema": "http://json-schema.org/draft-07/schema#",
        "additionalProperties": False,
        "properties": {"foo": {"type": "string"}},
        "required": ["foo"],
        "type": "object",
    }
    COHERE_TOOLS = [
        {
            "function": {
                "description": "test description: foo",
                "name": "MockTool_test",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "additionalProperties": False,
                            "description": "Test input",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "type": "object",
                        }
                    },
                    "required": ["values"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "name": "MockTool_test_callable_schema",
                "description": "test description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": {
                            "description": "Test input",
                            "type": "object",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "additionalProperties": False,
                        }
                    },
                    "required": ["values"],
                    "additionalProperties": False,
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description: foo",
                "name": "MockTool_test_error",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "additionalProperties": False,
                            "description": "Test input",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "type": "object",
                        }
                    },
                    "required": ["values"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description: foo",
                "name": "MockTool_test_exception",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "additionalProperties": False,
                            "description": "Test input",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "type": "object",
                        }
                    },
                    "required": ["values"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description",
                "name": "MockTool_test_list_output",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description",
                "name": "MockTool_test_no_schema",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description: foo",
                "name": "MockTool_test_str_output",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "additionalProperties": False,
                            "description": "Test input",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "type": "object",
                        }
                    },
                    "required": ["values"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description",
                "name": "MockTool_test_without_default_memory",
                "parameters": {
                    "$id": "Parameters Schema",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "additionalProperties": False,
                            "description": "Test input",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                            "type": "object",
                        }
                    },
                    "required": ["values"],
                    "type": "object",
                },
            },
            "type": "function",
        },
    ]

    @pytest.fixture()
    def mock_client(self, mocker):
        mock_client = mocker.patch("cohere.ClientV2").return_value
        mock_tool_call = Mock(id="mock-id", function=Mock(arguments=json.dumps({"foo": "bar"})))
        mock_tool_call.function.name = "MockTool_test"
        mock_client.chat.return_value = Mock(
            message=Mock(
                tool_plan="tool-plan",
                content=[Mock(text="model-output")],
                tool_calls=[mock_tool_call],
            ),
            usage=Mock(tokens=Mock(input_tokens=5, output_tokens=10)),
        )

        return mock_client

    @pytest.fixture()
    def mock_stream_client(self, mocker):
        mock_client = mocker.patch("cohere.ClientV2").return_value

        mock_tool_call_delta_header = Mock(
            message={
                "tool_calls": {
                    "id": "mock-id",
                    "function": {
                        "name": "MockTool_test",
                    },
                }
            },
        )

        mock_tool_call_delta_body = Mock(
            message={
                "tool_calls": {
                    "function": {
                        "arguments": '{"foo": "bar"}',
                    },
                }
            },
        )
        mock_client.chat_stream.return_value = iter(
            [
                Mock(type="content-delta", delta=Mock(message=Mock(content=Mock(text="model-output")))),
                Mock(type="tool-plan-delta", delta=Mock(message={"tool_plan": "tool-plan"})),
                Mock(type="tool-call-start", delta=mock_tool_call_delta_header),
                Mock(type="tool-call-delta", delta=mock_tool_call_delta_body),
                Mock(type="stream-end", response=Mock(meta=Mock(tokens=Mock(input_tokens=5, output_tokens=10)))),
            ]
        )

        return mock_client

    @pytest.fixture(autouse=True)
    def mock_tokenizer(self, mocker):
        return mocker.patch("griptape.tokenizers.CohereTokenizer").return_value

    @pytest.fixture()
    def prompt_stack(self):
        prompt_stack = PromptStack()
        prompt_stack.output_schema = Schema({"foo": str})
        prompt_stack.tools = [MockTool()]
        prompt_stack.add_system_message("system-input")
        prompt_stack.add_user_message("user-input")
        prompt_stack.add_assistant_message("assistant-input")
        prompt_stack.add_assistant_message(
            ListArtifact(
                [
                    TextArtifact("thought"),
                    ActionArtifact(ToolAction(tag="MockTool_test", name="MockTool", path="test", input={"foo": "bar"})),
                ]
            )
        )
        prompt_stack.add_user_message(
            ListArtifact(
                [
                    TextArtifact("keep-going"),
                    ActionArtifact(
                        ToolAction(
                            tag="MockTool_test",
                            name="MockTool",
                            path="test",
                            input={"foo": "bar"},
                            output=TextArtifact("tool-output"),
                        )
                    ),
                ]
            )
        )
        return prompt_stack

    @pytest.fixture()
    def messages(self):
        return [
            {"role": "system", "content": "system-input"},
            {"role": "user", "content": "user-input"},
            {"role": "assistant", "content": "assistant-input"},
            {
                "content": [
                    {"type": "text", "text": "thought"},
                ],
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {"arguments": '{"foo": "bar"}', "name": "MockTool_test"},
                        "id": "MockTool_test",
                        "type": "function",
                    }
                ],
            },
            {
                "content": {
                    "type": "text",
                    "text": "tool-output",
                },
                "role": "tool",
                "tool_call_id": "MockTool_test",
            },
            {"content": "keep-going", "role": "user"},
        ]

    def test_init(self):
        assert CoherePromptDriver(model="command", api_key="foobar")

    @pytest.mark.parametrize("use_native_tools", [True, False])
    @pytest.mark.parametrize("structured_output_strategy", ["native", "tool", "foo"])
    def test_try_run(
        self,
        mock_client,
        prompt_stack,
        messages,
        use_native_tools,
        structured_output_strategy,
    ):
        # Given
        driver = CoherePromptDriver(
            model="command",
            api_key="api-key",
            use_native_tools=use_native_tools,
            structured_output_strategy=structured_output_strategy,
            extra_params={"foo": "bar"},
        )

        # When
        message = driver.try_run(prompt_stack)

        # Then
        mock_client.chat.assert_called_once_with(
            model="command",
            messages=messages,
            max_tokens=None,
            **{"tools": self.COHERE_TOOLS} if use_native_tools else {},
            **{
                "response_format": {
                    "type": "json_object",
                    "schema": self.COHERE_STRUCTURED_OUTPUT_SCHEMA,
                }
            }
            if structured_output_strategy == "native"
            else {},
            stop_sequences=[],
            temperature=0.1,
            foo="bar",
        )

        assert isinstance(message.value[0], TextArtifact)
        assert message.value[0].value == "model-output"
        assert isinstance(message.value[1], TextArtifact)
        assert message.value[1].value == "tool-plan"
        assert isinstance(message.value[2], ActionArtifact)
        assert message.value[2].value.tag == "mock-id"
        assert message.value[2].value.name == "MockTool"
        assert message.value[2].value.path == "test"
        assert message.value[2].value.input == {"foo": "bar"}

        assert message.usage.input_tokens == 5
        assert message.usage.output_tokens == 10

    @pytest.mark.parametrize("use_native_tools", [True, False])
    @pytest.mark.parametrize("structured_output_strategy", ["native", "tool", "foo"])
    def test_try_stream_run(
        self,
        mock_stream_client,
        prompt_stack,
        messages,
        use_native_tools,
        structured_output_strategy,
    ):
        # Given
        driver = CoherePromptDriver(
            model="command",
            api_key="api-key",
            stream=True,
            use_native_tools=use_native_tools,
            structured_output_strategy=structured_output_strategy,
            extra_params={"foo": "bar"},
        )

        # When
        stream = driver.try_stream(prompt_stack)
        event = next(stream)

        # Then
        mock_stream_client.chat_stream.assert_called_once_with(
            model="command",
            messages=messages,
            max_tokens=None,
            **{"tools": self.COHERE_TOOLS} if use_native_tools else {},
            **{
                "response_format": {
                    "type": "json_object",
                    "schema": self.COHERE_STRUCTURED_OUTPUT_SCHEMA,
                }
            }
            if structured_output_strategy == "native"
            else {},
            stop_sequences=[],
            temperature=0.1,
            foo="bar",
        )

        assert isinstance(event.content, TextDeltaMessageContent)
        assert event.content.text == "model-output"

        event = next(stream)
        assert isinstance(event.content, TextDeltaMessageContent)
        assert event.content.text == "tool-plan"

        event = next(stream)
        assert isinstance(event.content, ActionCallDeltaMessageContent)
        assert event.content.tag == "mock-id"
        assert event.content.name == "MockTool"
        assert event.content.path == "test"

        event = next(stream)
        assert isinstance(event.content, ActionCallDeltaMessageContent)
        assert event.content.partial_input == '{"foo": "bar"}'

        event = next(stream)
        assert event.usage.input_tokens == 5
        assert event.usage.output_tokens == 10
