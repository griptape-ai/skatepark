import json

import pytest
import schema
from pydantic import create_model

from griptape.artifacts import ActionArtifact, ListArtifact, TextArtifact
from griptape.artifacts.error_artifact import ErrorArtifact
from griptape.artifacts.json_artifact import JsonArtifact
from griptape.common import ToolAction
from griptape.structures import Agent
from griptape.tasks import ActionsSubtask, PromptTask, ToolkitTask
from tests.mocks.mock_tool.tool import MockTool
from tests.mocks.mock_tool_pydantic.tool import MockToolPydantic


class TestActionsSubtask:
    def test_prompt_input(self):
        valid_input = TextArtifact(
            "Thought: need to test\n"
            'Actions: [{"tag": "foo", "name": "MockTool", "path": "test", "input": {"values": {"test": "value"}}}]\n'
            "<|Response|>: test observation\n"
            "Answer: test output",
            meta={"is_react_prompt": True},
        )

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))
        json_dict = json.loads(subtask.actions_to_json())

        assert json_dict[0]["name"] == "MockTool"
        assert json_dict[0]["path"] == "test"
        assert json_dict[0]["input"] == {"values": {"test": "value"}}
        assert subtask.thought == "need to test"
        assert subtask.output is None

    def test_artifact_input(self):
        valid_input = ListArtifact(
            [
                TextArtifact("need to test"),
                ActionArtifact(
                    ToolAction(tag="foo", name="MockTool", path="test", input={"values": {"test": "value"}})
                ),
                TextArtifact("answer"),
            ],
        )
        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))
        json_dict = json.loads(subtask.actions_to_json())

        assert json_dict[0]["name"] == "MockTool"
        assert json_dict[0]["path"] == "test"
        assert json_dict[0]["input"] == {"values": {"test": "value"}}
        assert subtask.thought == "need to test"
        assert subtask.output is None

    def test_artifact_action_and_thought_input(self):
        valid_input = ListArtifact(
            [
                TextArtifact("thought"),
                ActionArtifact(
                    ToolAction(tag="foo", name="MockTool", path="test", input={"values": {"test": "value"}})
                ),
            ],
        )
        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))
        json_dict = json.loads(subtask.actions_to_json())

        assert subtask.thought == "thought"
        assert json_dict[0]["name"] == "MockTool"
        assert json_dict[0]["path"] == "test"
        assert json_dict[0]["input"] == {"values": {"test": "value"}}

    @pytest.mark.parametrize(("is_react_prompt", "expected"), [(True, "test output"), (False, "Answer: test output")])
    def test_prompt_answer(self, is_react_prompt, expected):
        valid_input = TextArtifact("Answer: test output", meta={"is_react_prompt": is_react_prompt})

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))

        assert subtask.thought is None
        assert subtask.actions == []
        assert subtask.output is not None
        assert subtask.output.value == expected

    @pytest.mark.parametrize("is_react_prompt", [True, False])
    def test_prompt_implicit_answer(self, is_react_prompt):
        valid_input = TextArtifact("test output", meta={"use_native_tools": is_react_prompt})

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))

        assert subtask.thought is None
        assert subtask.actions == []
        assert subtask.output is not None
        assert subtask.output.value == "test output"

    def test_artifact_answer(self):
        valid_input = ListArtifact(
            [
                TextArtifact("answer"),
            ]
        )
        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))

        assert subtask.thought is None
        assert subtask.actions == []
        assert subtask.output is not None
        assert subtask.output.value == "answer"

    def test_callable_input(self):
        valid_input = ListArtifact(
            [
                TextArtifact("thought"),
                ActionArtifact(
                    ToolAction(tag="foo", name="MockTool", path="test", input={"values": {"test": "value"}})
                ),
            ]
        )
        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(lambda task: valid_input))
        json_dict = json.loads(subtask.actions_to_json())

        assert subtask.thought == "thought"
        assert json_dict[0]["name"] == "MockTool"
        assert json_dict[0]["path"] == "test"
        assert json_dict[0]["input"] == {"values": {"test": "value"}}

    def test_input_with_multiline_actions(self):
        valid_input = TextArtifact(
            "Thought: need to test\n"
            'Actions:\nFoobarfoobar baz}!@#$%^&*()123(*!378934)\n\n```json\n[{"tag": "foo", "name": "MockTool",\n"path": "test",\n\n"input": {"values":\n{"test":\n"test\n\ninput\n\nwith\nnewlines"}}}]```!@#$%^&*()123(*!378934)'
            "Response: test response\n"
            "Answer: test output",
            meta={"is_react_prompt": True},
        )

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))
        json_dict = json.loads(subtask.actions_to_json())

        assert json_dict[0]["name"] == "MockTool"
        assert json_dict[0]["path"] == "test"
        assert json_dict[0]["input"] == {"values": {"test": "test\n\ninput\n\nwith\nnewlines"}}

    def test_input(self):
        assert ActionsSubtask("{{ hello }}").input.value == "{{ hello }}"

    def test_with_no_action_input(self):
        valid_input = TextArtifact(
            "Thought: need to test\n"
            'Actions: [{"tag": "foo", "name": "MockTool", "path": "test_no_schema"}]\n'
            "<|Response|>: test observation\n"
            "Answer: test output",
            meta={"is_react_prompt": True},
        )

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))
        json_dict = json.loads(subtask.actions_to_json())

        assert json_dict[0]["name"] == "MockTool"
        assert json_dict[0]["path"] == "test_no_schema"
        assert json_dict[0].get("input") is None

    def test_no_actions(self):
        valid_input = TextArtifact(
            "Thought: need to test\n<|Response|>: test observation\nAnswer: test output",
            meta={"is_react_prompt": True},
        )

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))
        json_dict = json.loads(subtask.actions_to_json())

        assert len(json_dict) == 0

    def test_empty_actions(self):
        valid_input = TextArtifact(
            "Thought: need to test\nActions: []\n<|Response|>: test observation\nAnswer: test output"
        )

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))
        json_dict = json.loads(subtask.actions_to_json())

        assert len(json_dict) == 0

    def test_invalid_actions(self):
        invalid_input = TextArtifact(
            "Thought: need to test\nActions: [{,{]\n<|Response|>: test observation\nAnswer: test output",
            meta={"is_react_prompt": True},
        )

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(invalid_input))

        assert isinstance(subtask.output, ErrorArtifact)
        assert "Actions JSON decoding error" in subtask.output.value
        assert subtask.thought == "need to test"
        assert subtask.actions == []

    def test_implicit_values(self):
        valid_input = TextArtifact(
            "Thought: need to test\n"
            'Actions:[{"tag": "foo", "name": "MockTool","path": "test","input": {"test":\n"value"}}]'
            "Response: test response\n"
            "Answer: test output",
            meta={"is_react_prompt": True},
        )

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))
        json_dict = json.loads(subtask.actions_to_json())

        assert json_dict[0]["name"] == "MockTool"
        assert json_dict[0]["path"] == "test"
        assert json_dict[0]["input"] == {"values": {"test": "value"}}

    def test_execute_tool(self):
        valid_input = TextArtifact(
            "Thought: need to test\n"
            'Actions:[{"tag": "foo", "name": "MockTool","path": "test","input": {"values": {"test": "value"}}}]',
            meta={"is_react_prompt": True},
        )

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))
        subtask.run()

        assert isinstance(subtask.output, ListArtifact)
        assert isinstance(subtask.output.value[0], TextArtifact)
        assert subtask.output.value[0].value == "ack value"

    @pytest.mark.parametrize(
        "invalid_input",
        [
            ListArtifact(
                [
                    ActionArtifact(
                        ToolAction(tag="foo", name="MockTool", path="test", input={"values": {"test1234": "value"}})
                    ),
                ]
            ),
            ListArtifact(
                [
                    ActionArtifact(ToolAction(tag="foo", name="MockTool", path="test", input={"values": {}})),
                ]
            ),
            ListArtifact(
                [
                    ActionArtifact(
                        ToolAction(
                            tag="foo",
                            name="MockTool",
                            path="test",
                            input={
                                "values": {
                                    "test": "value",
                                    "extra": "extra value",
                                }
                            },
                        )
                    ),
                ]
            ),
        ],
    )
    @pytest.mark.parametrize(
        "tool",
        [
            MockTool(),
            MockToolPydantic(name="MockTool"),
        ],
    )
    def test_action_validation_error(self, invalid_input, tool):
        task = PromptTask(tools=[tool])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(invalid_input))
        subtask.run()

        assert isinstance(subtask.output, ErrorArtifact)
        assert "Activity input JSON validation error" in subtask.output.value

    def test_execute_tool_exception(self):
        valid_input = TextArtifact(
            "Thought: need to test\n"
            'Actions:[{"tag": "foo", "name": "MockTool","path": "test_exception","input": {"values": {"test": "value"}}}]',
            meta={"is_react_prompt": True},
        )

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))
        subtask.run()

        assert isinstance(subtask.output, ListArtifact)
        assert isinstance(subtask.output.value[0], ErrorArtifact)
        assert subtask.output.value[0].value == "error value"

    def test_origin_task(self):
        valid_input = TextArtifact(
            "Thought: need to test\n"
            'Actions:[{"tag": "foo", "name": "MockTool","path": "test","input": {"values": {"test": "value"}}}]',
            meta={"is_react_prompt": True},
        )

        task = PromptTask(tools=[MockTool()])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(valid_input))

        assert subtask.origin_task == task

        with pytest.raises(Exception, match="ActionSubtask has no origin task."):
            assert ActionsSubtask("test").origin_task

    @pytest.mark.parametrize(
        "output_schema",
        [
            schema.Schema({"test": str}),
            create_model("TestModel", test=(str, ...)),
        ],
    )
    def test_structured_output_tool(self, output_schema):
        from griptape.tools.structured_output.tool import StructuredOutputTool

        actions = ListArtifact(
            [
                ActionArtifact(
                    ToolAction(
                        tag="foo",
                        name="StructuredOutputTool",
                        path="provide_output",
                        input={"values": {"test": "value"}},
                    )
                ),
            ]
        )

        task = ToolkitTask(tools=[StructuredOutputTool(output_schema=output_schema)])
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(actions))

        assert isinstance(subtask.output, JsonArtifact)
        assert subtask.output.value == {"test": "value"}

    @pytest.mark.parametrize(
        "output_schemas",
        [
            (schema.Schema({"test1": str}), schema.Schema({"test2": str})),
            (create_model("TestModel1", test1=(str, ...)), create_model("TestModel2", test2=(str, ...))),
        ],
    )
    def test_structured_output_tool_multiple(self, output_schemas):
        from griptape.tools.structured_output.tool import StructuredOutputTool

        actions = ListArtifact(
            [
                ActionArtifact(
                    ToolAction(
                        tag="foo",
                        name="StructuredOutputTool1",
                        path="provide_output",
                        input={"values": {"test1": "value"}},
                    )
                ),
                ActionArtifact(
                    ToolAction(
                        tag="foo",
                        name="StructuredOutputTool2",
                        path="provide_output",
                        input={"values": {"test2": "value"}},
                    )
                ),
            ]
        )

        task = ToolkitTask(
            tools=[
                StructuredOutputTool(name="StructuredOutputTool1", output_schema=output_schemas[0]),
                StructuredOutputTool(name="StructuredOutputTool2", output_schema=output_schemas[1]),
            ]
        )
        Agent().add_task(task)
        subtask = task.add_subtask(ActionsSubtask(actions))

        assert isinstance(subtask.output, ListArtifact)
        assert len(subtask.output.value) == 2
        assert subtask.output.value[0].value == {"test1": "value"}
        assert subtask.output.value[1].value == {"test2": "value"}
