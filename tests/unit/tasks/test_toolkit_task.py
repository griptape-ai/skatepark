import pytest

from griptape.artifacts import ErrorArtifact, TextArtifact
from griptape.common import ToolAction
from griptape.structures import Agent
from griptape.tasks import ActionsSubtask, ToolkitTask
from tests.mocks.mock_tool.tool import MockTool
from tests.utils import defaults


class TestToolkitSubtask:
    TARGET_TOOLS_SCHEMA = {
        "description": "JSON schema for an array of actions.",
        "type": "array",
        "items": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description: foo", "const": "test"},
                        "input": {
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
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description", "const": "test_callable_schema"},
                        "input": {
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
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description: foo", "const": "test_error"},
                        "input": {
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
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description: foo", "const": "test_exception"},
                        "input": {
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
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description", "const": "test_list_output"},
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                        "input": {"additionalProperties": False, "properties": {}, "required": [], "type": "object"},
                    },
                    "required": ["name", "path", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description", "const": "test_no_schema"},
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                        "input": {"additionalProperties": False, "properties": {}, "required": [], "type": "object"},
                    },
                    "required": ["name", "path", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description: foo", "const": "test_str_output"},
                        "input": {
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
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "MockTool"},
                        "path": {"description": "test description", "const": "test_without_default_memory"},
                        "input": {
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
                        },
                        "tag": {"description": "Unique tag name for action execution.", "type": "string"},
                    },
                    "required": ["name", "path", "input", "tag"],
                    "additionalProperties": False,
                },
            ]
        },
        "$id": "Actions Schema",
        "$schema": "http://json-schema.org/draft-07/schema#",
    }

    def test_init(self):
        assert len(ToolkitTask("test", tools=[MockTool(name="Tool1"), MockTool(name="Tool2")]).tools) == 2

        try:
            ToolkitTask("test", tools=[MockTool(), MockTool()])
            raise AssertionError()
        except ValueError:
            assert True

    def test_run(self, mock_config):
        output = """Answer: done"""
        mock_config.drivers_config.prompt_driver.mock_output = output

        task = ToolkitTask("test", tools=[MockTool(name="Tool1"), MockTool(name="Tool2")])
        agent = Agent()

        agent.add_task(task)

        result = agent.run()

        assert len(task.tools) == 2
        assert len(task.subtasks) == 1
        assert result.output_task.output.to_text() == "done"

    def test_run_max_subtasks(self, mock_config):
        output = 'Actions: [{"tag": "foo", "name": "Tool1", "path": "test", "input": {"values": {"test": "value"}}}]'
        mock_config.drivers_config.prompt_driver.mock_output = output

        task = ToolkitTask("test", tools=[MockTool(name="Tool1")], max_subtasks=3)
        agent = Agent()

        agent.add_task(task)

        agent.run()

        assert len(task.subtasks) == 3
        assert isinstance(task.output, ErrorArtifact)

    def test_run_invalid_react_prompt(self, mock_config):
        output = """foo bar"""
        mock_config.drivers_config.prompt_driver.mock_output = output

        task = ToolkitTask("test", tools=[MockTool(name="Tool1")], max_subtasks=3)
        agent = Agent()

        agent.add_task(task)

        result = agent.run()

        assert len(task.subtasks) == 1
        assert result.output_task.output.to_text() == "foo bar"

    def test_init_from_prompt_1(self):
        valid_input = (
            "Thought: need to test\n"
            'Actions: [{"tag": "foo", "name": "Tool1", "path": "test", "input": {"values": {"test": "value"}}}]\n'
            "<|Response|>: test observation\n"
            "Answer: test output"
        )
        task = ToolkitTask("test", tools=[MockTool(name="Tool1")])

        Agent().add_task(task)

        subtask = task.add_subtask(ActionsSubtask(valid_input))

        assert subtask.thought == "need to test"
        assert subtask.actions[0].tag == "foo"
        assert subtask.actions[0].name == "Tool1"
        assert subtask.actions[0].path == "test"
        assert subtask.actions[0].input == {"values": {"test": "value"}}
        assert subtask.output is None

    def test_init_from_prompt_2(self):
        valid_input = """Thought: need to test\nObservation: test
        observation\nAnswer: test output"""
        task = ToolkitTask("test", tools=[MockTool(name="Tool1")])

        Agent().add_task(task)

        subtask = task.add_subtask(ActionsSubtask(valid_input))

        assert subtask.thought == "need to test"
        assert subtask.actions == []
        assert subtask.output.to_text() == "test output"

    def test_add_subtask(self):
        task = ToolkitTask("test", tools=[MockTool(name="Tool1")])
        subtask1 = ActionsSubtask(
            "test1", actions=[ToolAction(tag="foo", name="test", path="test", input={"values": {"f": "b"}})]
        )
        subtask2 = ActionsSubtask(
            "test2", actions=[ToolAction(tag="foo", name="test", path="test", input={"values": {"f": "b"}})]
        )

        Agent().add_task(task)

        task.add_subtask(subtask1)
        task.add_subtask(subtask2)

        assert len(task.subtasks) == 2

        assert len(subtask1.children) == 1
        assert len(subtask1.parents) == 0
        assert subtask1.children[0] == subtask2

        assert len(subtask2.children) == 0
        assert len(subtask2.parents) == 1
        assert subtask2.parents[0] == subtask1

    def test_find_subtask(self):
        task = ToolkitTask("test", tools=[MockTool(name="Tool1")])
        subtask1 = ActionsSubtask(
            "test1", actions=[ToolAction(tag="foo", name="test", path="test", input={"values": {"f": "b"}})]
        )
        subtask2 = ActionsSubtask(
            "test2", actions=[ToolAction(tag="foo", name="test", path="test", input={"values": {"f": "b"}})]
        )

        Agent().add_task(task)

        task.add_subtask(subtask1)
        task.add_subtask(subtask2)

        assert task.find_subtask(subtask1.id) == subtask1
        assert task.find_subtask(subtask2.id) == subtask2

    def test_find_tool(self):
        tool = MockTool()
        task = ToolkitTask("test", tools=[tool])

        Agent().add_task(task)

        assert task.find_tool(tool.name) == tool

    def test_find_memory(self):
        m1 = defaults.text_task_memory("Memory1")
        m2 = defaults.text_task_memory("Memory2")

        tool = MockTool(name="Tool1", output_memory={"test": [m1, m2]}, off_prompt=True)
        task = ToolkitTask("test", tools=[tool])

        Agent().add_task(task)

        assert task.find_memory("Memory1") == m1
        assert task.find_memory("Memory2") == m2

    def test_memory(self):
        tool1 = MockTool(
            name="Tool1",
            output_memory={"test": [defaults.text_task_memory("Memory1"), defaults.text_task_memory("Memory2")]},
            off_prompt=True,
        )

        tool2 = MockTool(
            name="Tool2",
            output_memory={"test": [defaults.text_task_memory("Memory1"), defaults.text_task_memory("Memory3")]},
            off_prompt=True,
        )

        task = ToolkitTask(tools=[tool1, tool2])

        Agent().add_task(task)

        assert len(task.tool_output_memory) == 3
        assert task.tool_output_memory[0].name == "Memory1"
        assert task.tool_output_memory[1].name == "Memory2"
        assert task.tool_output_memory[2].name == "Memory3"

    def test_meta_memory(self):
        memory = defaults.text_task_memory("TestMemory")
        subtask = ActionsSubtask()
        agent = Agent(task_memory=memory)

        subtask.structure = agent

        memory.process_output(MockTool().test, subtask, TextArtifact("foo"))

        task = ToolkitTask(tools=[MockTool()])

        agent.add_task(task)

        system_template = task.generate_system_template(ToolkitTask())

        assert "You have access to additional contextual information" in system_template

    def test_actions_schema(self):
        tool = MockTool()
        task = ToolkitTask("test", tools=[tool])

        Agent().add_task(task)

        assert task.actions_schema().json_schema("Actions Schema") == self.TARGET_TOOLS_SCHEMA

    def test_to_dict(self):
        tool = MockTool()
        task = ToolkitTask("test", tools=[tool])

        assert task.to_dict() == {
            "type": "ToolkitTask",
            "id": task.id,
            "state": "State.PENDING",
            "parent_ids": [],
            "child_ids": [],
            "max_meta_memory_entries": 20,
            "context": {},
            "rulesets": [],
            "rules": [],
            "prompt_driver": {
                "extra_params": {},
                "max_tokens": None,
                "stream": False,
                "temperature": 0.1,
                "type": "MockPromptDriver",
                "use_native_tools": False,
                "structured_output_strategy": "rule",
            },
            "tools": [
                {
                    "type": "MockTool",
                    "name": "MockTool",
                    "input_memory": None,
                    "output_memory": None,
                    "install_dependencies_on_init": True,
                    "dependencies_install_directory": None,
                    "verbose": False,
                    "off_prompt": False,
                }
            ],
            "max_subtasks": 20,
        }

    def test_from_dict(self):
        tool = MockTool()
        task = ToolkitTask("test", tools=[tool])
        serialized_task = task.to_dict()
        serialized_task["tools"][0]["module_name"] = "tests.mocks.mock_tool.tool"
        serialized_task["prompt_driver"]["module_name"] = "tests.mocks.mock_prompt_driver"

        assert ToolkitTask.from_dict(serialized_task).to_dict() == task.to_dict()

    def test_deprecation_warning(self):
        task = ToolkitTask("test")

        with pytest.warns(DeprecationWarning):
            task.run()
