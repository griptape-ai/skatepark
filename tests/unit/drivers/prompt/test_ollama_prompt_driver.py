import json

import pytest
from schema import Schema

from griptape.artifacts import ActionArtifact, ImageArtifact, ListArtifact, TextArtifact
from griptape.common import ActionCallDeltaMessageContent, PromptStack, TextDeltaMessageContent, ToolAction
from griptape.drivers.prompt.ollama import OllamaPromptDriver
from tests.mocks.mock_tool.tool import MockTool


class TestOllamaPromptDriver:
    OLLAMA_STRUCTURED_OUTPUT_SCHEMA = {
        "$id": "Output Format",
        "$schema": "http://json-schema.org/draft-07/schema#",
        "additionalProperties": False,
        "properties": {"foo": {"type": "string"}},
        "required": ["foo"],
        "type": "object",
    }
    OLLAMA_TOOLS = [
        {
            "function": {
                "description": "test description: foo",
                "name": "MockTool_test",
                "parameters": {
                    "additionalProperties": False,
                    "description": "Test input",
                    "properties": {"test": {"type": "string"}},
                    "required": ["test"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description",
                "name": "MockTool_test_callable_schema",
                "parameters": {
                    "additionalProperties": False,
                    "description": "Test input",
                    "properties": {"test": {"type": "string"}},
                    "required": ["test"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description: foo",
                "name": "MockTool_test_error",
                "parameters": {
                    "additionalProperties": False,
                    "description": "Test input",
                    "properties": {"test": {"type": "string"}},
                    "required": ["test"],
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
                    "additionalProperties": False,
                    "description": "Test input",
                    "properties": {"test": {"type": "string"}},
                    "required": ["test"],
                    "type": "object",
                },
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description",
                "name": "MockTool_test_list_output",
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description",
                "name": "MockTool_test_no_schema",
            },
            "type": "function",
        },
        {
            "function": {
                "description": "test description: foo",
                "name": "MockTool_test_str_output",
                "parameters": {
                    "additionalProperties": False,
                    "description": "Test input",
                    "properties": {"test": {"type": "string"}},
                    "required": ["test"],
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
                    "additionalProperties": False,
                    "description": "Test input",
                    "properties": {"test": {"type": "string"}},
                    "required": ["test"],
                    "type": "object",
                },
            },
            "type": "function",
        },
    ]

    @pytest.fixture()
    def mock_client(self, mocker):
        mock_client = mocker.patch("ollama.Client")

        mock_response = mocker.MagicMock()

        data = {
            "message": {
                "content": "model-output",
                "tool_calls": [
                    {
                        "function": {
                            "name": "MockTool_test",
                            "arguments": {"foo": "bar"},
                        }
                    },
                ],
            },
        }

        mock_response.__getitem__.side_effect = lambda key: data[key]
        mock_response.model_dump.return_value = data
        mock_client.return_value.chat.return_value = mock_response

        return mock_client

    @pytest.fixture()
    def mock_stream_client(self, mocker):
        mock_stream_client = mocker.patch("ollama.Client")
        mock_stream_client.return_value.chat.return_value = iter(
            [
                {"message": {"content": "model-output"}},
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "MockTool_test",
                                    "arguments": {"foo": "bar"},
                                }
                            },
                        ],
                    }
                },
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "MockTool_test",
                                    "arguments": {"foo": "bar"},
                                }
                            },
                            {
                                "function": {
                                    "name": "MockTool_test",
                                    "arguments": {"foo": "bar"},
                                }
                            },
                        ],
                    }
                },
                {"message": {}},
            ]
        )

        return mock_stream_client

    @pytest.fixture()
    def prompt_stack(self):
        prompt_stack = PromptStack()
        prompt_stack.output_schema = Schema({"foo": str})
        prompt_stack.tools = [MockTool()]
        prompt_stack.add_system_message("system-input")
        prompt_stack.add_user_message("user-input")
        prompt_stack.add_user_message(
            ListArtifact(
                [TextArtifact("user-input"), ImageArtifact(value=b"image-data", format="png", width=100, height=100)]
            )
        )
        prompt_stack.add_assistant_message("assistant-input")
        prompt_stack.add_assistant_message(
            ListArtifact(
                [
                    TextArtifact(""),
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
            {
                "role": "user",
                "content": "user-input",
                "images": ["aW1hZ2UtZGF0YQ=="],
            },
            {"role": "assistant", "content": "assistant-input"},
            {
                "content": "",
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {"arguments": {"foo": "bar"}, "name": "MockTool_test"},
                        "id": "MockTool_test",
                        "type": "function",
                    }
                ],
            },
            {"content": "tool-output", "role": "tool"},
            {"content": "keep-going", "role": "user"},
        ]

    def test_init(self):
        assert OllamaPromptDriver(model="llama")

    @pytest.mark.parametrize("use_native_tools", [True, False])
    @pytest.mark.parametrize("structured_output_strategy", ["native", "tool", "rule", "foo"])
    def test_try_run(
        self,
        mock_client,
        prompt_stack,
        messages,
        use_native_tools,
        structured_output_strategy,
    ):
        # Given
        driver = OllamaPromptDriver(
            model="llama",
            use_native_tools=use_native_tools,
            structured_output_strategy=structured_output_strategy,
            extra_params={"foo": "bar"},
        )

        # When
        message = driver.try_run(prompt_stack)

        # Then
        mock_client.return_value.chat.assert_called_once_with(
            messages=messages,
            model=driver.model,
            options={
                "temperature": driver.temperature,
                "stop": [],
                "num_predict": driver.max_tokens,
            },
            **{
                "tools": self.OLLAMA_TOOLS,
            }
            if use_native_tools
            else {},
            **{"format": self.OLLAMA_STRUCTURED_OUTPUT_SCHEMA} if structured_output_strategy == "native" else {},
            foo="bar",
        )
        assert isinstance(message.value[0], TextArtifact)
        assert message.value[0].value == "model-output"
        assert isinstance(message.value[1], ActionArtifact)
        assert message.value[1].value.tag == "MockTool_test"
        assert message.value[1].value.name == "MockTool"
        assert message.value[1].value.path == "test"
        assert message.value[1].value.input == {"foo": "bar"}

    @pytest.mark.parametrize("use_native_tools", [True, False])
    @pytest.mark.parametrize("structured_output_strategy", ["native", "tool", "rule", "foo"])
    def test_try_stream_run(
        self,
        mock_stream_client,
        prompt_stack,
        messages,
        use_native_tools,
        structured_output_strategy,
    ):
        # Given
        driver = OllamaPromptDriver(
            model="llama",
            stream=True,
            use_native_tools=use_native_tools,
            structured_output_strategy=structured_output_strategy,
            extra_params={"foo": "bar"},
        )

        # When
        stream = driver.try_stream(prompt_stack)
        event = next(stream)

        # Then
        mock_stream_client.return_value.chat.assert_called_once_with(
            messages=messages,
            model=driver.model,
            options={"temperature": driver.temperature, "stop": [], "num_predict": driver.max_tokens},
            **{
                "tools": self.OLLAMA_TOOLS,
            }
            if use_native_tools
            else {},
            **{"format": self.OLLAMA_STRUCTURED_OUTPUT_SCHEMA} if structured_output_strategy == "native" else {},
            stream=True,
            foo="bar",
        )
        assert isinstance(event.content, TextDeltaMessageContent)
        assert event.content.text == "model-output"

        event = next(stream)
        assert isinstance(event.content, ActionCallDeltaMessageContent)
        assert event.content.index == 0
        assert event.content.tag == "MockTool_test"
        assert event.content.name == "MockTool"
        assert event.content.path == "test"
        assert event.content.partial_input == json.dumps({"foo": "bar"})

        event = next(stream)
        assert isinstance(event.content, ActionCallDeltaMessageContent)
        assert event.content.index == 1
        assert event.content.tag == "MockTool_test"
        assert event.content.name == "MockTool"
        assert event.content.path == "test"
        assert event.content.partial_input == json.dumps({"foo": "bar"})

        event = next(stream)
        assert isinstance(event.content, TextDeltaMessageContent)
        assert event.content.text == ""
